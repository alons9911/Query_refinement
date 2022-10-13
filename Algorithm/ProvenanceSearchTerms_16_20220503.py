"""
executable but only for relaxation
Pure relaxations and refinements are treated differently.
Difference from 15:
In relaxation: Try to drop duplicated rows for delta table

In bidirectional: implementing


"""

import copy
from typing import List, Any
import numpy as np
import pandas as pd
import time
from intbitset import intbitset
import json
from Algorithm import LatticeTraversal_4_20220901 as lt


def subtract_provenance(data, selected_attributes, sensitive_attributes, fairness_constraints,
                        numeric_attributes, categorical_attributes, selection_numeric_attributes,
                        selection_categorical_attributes):
    """
Get provenance expressions
    :param all_sensitive_attributes: list of all att involved in fairness constraints
    :param fairness_constraints: [{'Gender': 'F', 'symbol': '>=', 'number': 3}]
    :param data: dataframe
    :param selected_attributes: attributes in selection conditions
    :return: a list of dictionaries
    """
    fairness_constraints_provenance_greater_than = []
    fairness_constraints_provenance_smaller_than = []
    data['protected_greater_than'] = 0
    data['protected_smaller_than'] = 0
    data['satisfy'] = 0

    # whether it's single-direction
    only_smaller_than = True
    only_greater_than = True
    for fc in fairness_constraints:
        if fc['symbol'] == ">=" or fc['symbol'] == ">":
            only_smaller_than = False
        else:
            only_greater_than = False
        if (not only_greater_than) and (not only_smaller_than):
            break

    # if one direction, evaluate whether a row satisfies selection conditions
    def test_satisfying_rows(row):
        terms = row[selected_attributes].to_dict()
        for k in terms:
            if k in selection_numeric_attributes:
                if not eval(
                        str(terms[k]) + selection_numeric_attributes[k][0] + str(selection_numeric_attributes[k][1])):
                    return 0
            else:
                if terms[k] not in selection_categorical_attributes[k]:
                    return 0
        return 1

    if only_greater_than:
        data['satisfy'] = data.apply(test_satisfying_rows, axis=1)
        all_relevant_attributes = sensitive_attributes + selected_attributes + \
                                  ['protected_greater_than', 'protected_smaller_than', 'satisfy']
        data = data[all_relevant_attributes]
        data = data.groupby(all_relevant_attributes, dropna=False, sort=False).size().reset_index(name='occurrence')

        def get_provenance_relax_only(row, fc_dic, fc):
            sensitive_att_of_fc = list(fc["sensitive_attributes"].keys())
            sensitive_values_of_fc = {k: fc["sensitive_attributes"][k] for k in sensitive_att_of_fc}
            fairness_value_of_row = row[sensitive_att_of_fc]
            if sensitive_values_of_fc == fairness_value_of_row.to_dict():
                if row['satisfy'] == 1:
                    fc_dic['number'] -= row['occurrence']
                else:
                    terms = row[selected_attributes]
                    term_dic = terms.to_dict()
                    term_dic['occurrence'] = row['occurrence']
                    fc_dic['provenance_expression'].append(term_dic)
                    row['protected_greater_than'] = 1
            return row

        for fc in fairness_constraints:
            fc_dic = dict()
            fc_dic['symbol'] = fc['symbol']
            fc_dic['number'] = fc['number']
            fc_dic['provenance_expression'] = []
            data = data[all_relevant_attributes + ['occurrence']].apply(get_provenance_relax_only,
                                                                        args=(fc_dic, fc),
                                                                        axis=1)
            fairness_constraints_provenance_greater_than.append(fc_dic)
        data = data[data['satisfy'] == 0]
        data_rows_greater_than = data[data['protected_greater_than'] == 1]
        data_rows_smaller_than = data[data['protected_smaller_than'] == 1]
        return fairness_constraints_provenance_greater_than, fairness_constraints_provenance_smaller_than, \
               data_rows_greater_than, data_rows_smaller_than, only_greater_than, only_smaller_than

    elif only_smaller_than:
        data['satisfy'] = data.apply(test_satisfying_rows, axis=1)

    all_relevant_attributes = sensitive_attributes + selected_attributes + \
                              ['protected_greater_than', 'protected_smaller_than', 'satisfy']
    data = data[all_relevant_attributes]
    data = data.groupby(all_relevant_attributes, dropna=False, sort=False).size().reset_index(name='occurrence')

    def get_provenance(row, fc_dic, fc, protected_greater_than):
        sensitive_att_of_fc = list(fc["sensitive_attributes"].keys())
        sensitive_values_of_fc = {k: fc["sensitive_attributes"][k] for k in sensitive_att_of_fc}
        fairness_value_of_row = row[sensitive_att_of_fc]
        if sensitive_values_of_fc == fairness_value_of_row.to_dict():
            if only_greater_than and row['satisfy'] == 1:
                fc_dic['number'] -= row['occurrence']
            elif not (only_smaller_than and row['satisfy'] == 0):
                terms = row[selected_attributes]
                term_dic = terms.to_dict()
                term_dic['occurrence'] = row['occurrence']
                fc_dic['provenance_expression'].append(term_dic)
            if protected_greater_than:
                row['protected_greater_than'] = 1
            else:
                row['protected_smaller_than'] = 1
        return row

    for fc in fairness_constraints:
        fc_dic = dict()
        fc_dic['symbol'] = fc['symbol']
        fc_dic['number'] = fc['number']
        fc_dic['provenance_expression'] = []
        data = data[all_relevant_attributes + ['occurrence']].apply(get_provenance,
                                                                    args=(fc_dic,
                                                                          fc,
                                                                          (fc['symbol'] == ">" or
                                                                           fc[
                                                                               'symbol'] == ">=")),
                                                                    axis=1)

        if fc_dic['symbol'] == "<" or fc_dic['symbol'] == "<=":
            fairness_constraints_provenance_smaller_than.append(fc_dic)
        else:
            fairness_constraints_provenance_greater_than.append(fc_dic)
    # if one direction, remove rows that already satisfy/dissatisfy selection conditions
    if only_smaller_than:
        data = data[data['satisfy'] == 1]
    # only get rows that are envolved w.r.t. fairness constraint
    data_rows_greater_than = data[data['protected_greater_than'] == 1]
    data_rows_smaller_than = data[data['protected_smaller_than'] == 1]

    return fairness_constraints_provenance_greater_than, fairness_constraints_provenance_smaller_than, \
           data_rows_greater_than, data_rows_smaller_than, only_greater_than, only_smaller_than


def compute_delta_numeric_att(value_of_data, symbol, threshold, greater_than=True):
    if greater_than:
        if symbol == ">":
            if value_of_data > threshold:
                return 0
            else:
                return threshold - value_of_data + 1
        elif symbol == ">=":
            if value_of_data >= threshold:
                return 0
            else:
                return threshold - value_of_data
    else:
        if symbol == ">":
            if value_of_data > threshold:
                return threshold - value_of_data + 1
            else:
                return 0
        elif symbol == ">=":
            if value_of_data >= threshold:
                return threshold - value_of_data
            else:
                return 0


def build_sorted_table_relax_only(data, selected_attributes, numeric_attributes,
                                  categorical_attributes, selection_numeric, selection_categorical,
                                  sensitive_attributes, fairness_constraints,
                                  fairness_constraints_provenance_greater_than,
                                  fairness_constraints_provenance_smaller_than,
                                  data_rows_greater_than, data_rows_smaller_than
                                  ):
    """
    to build the sorted table
    :param fairness_constraints_provenance_greater_than:
    :param fairness_constraints_provenance_smaller_than:
    :param data_rows_smaller_than:
    :param data_rows_greater_than:
    :param selected_attributes:
    :param data: dataframe
    :param numeric_attributes: list of names of numeric attributes [city, major, state]
    :param categorical_attributes: dictionary: {city: [domain of city], major: [domain of major]}
    :param selection_numeric: dictionary: {grade:[80, >], age:[30, <], hours: [100, <=]}
    :param selection_categorical: dictionary: {city: [accepted cities], major: [accepted majors]}
    :return: return the whole sorted table, including rows that already satisfy the selection conditions;
            also return delta table
    """

    columns_delta_table = numeric_attributes.copy()
    for att, domain in categorical_attributes.items():
        for value in domain:
            if value in selection_categorical[att]:
                continue
            else:
                columns_delta_table.append(att + "_" + value)

    list_of_rows_delta_table = []

    # build delta table
    def iterrow(row, greater_than=True):  # greater than is symbol in fairness constraint (relaxation term)
        delta_values = []  # for delta table where relaxation term doesn't have negative delta values recorded
        for att in numeric_attributes:
            delta_v_zero = 0
            if greater_than:  # relaxation term
                if selection_numeric[att][0] == ">":
                    delta_v = selection_numeric[att][1] - row[att] + selection_numeric[att][2]
                    if row[att] > selection_numeric[att][1]:
                        delta_v_zero = 0
                    else:
                        delta_v_zero = delta_v
                elif selection_numeric[att][0] == ">=":
                    delta_v = selection_numeric[att][1] - row[att]
                    if row[att] >= selection_numeric[att][1]:
                        delta_v_zero = 0
                    else:
                        delta_v_zero = delta_v
                elif selection_numeric[att][0] == "<":
                    delta_v = row[att] - selection_numeric[att][1] + selection_numeric[att][2]
                    if row[att] < selection_numeric[att][1]:
                        delta_v_zero = 0
                    else:
                        delta_v_zero = delta_v
                else:
                    delta_v = row[att] - selection_numeric[att][1]
                    if row[att] <= selection_numeric[att][1]:
                        delta_v_zero = 0
                    else:
                        delta_v_zero = delta_v
            else:
                if selection_numeric[att][0] == ">":
                    if row[att] <= selection_numeric[att][1]:
                        delta_v = 0
                    else:
                        delta_v = - (row[att] - selection_numeric[att][1])
                elif selection_numeric[att][0] == ">=":
                    if row[att] < selection_numeric[att][1]:
                        delta_v = 0
                    else:
                        delta_v = - (row[att] - selection_numeric[att][1] + selection_numeric[att][2])
                elif selection_numeric[att][0] == "<":
                    if row[att] >= selection_numeric[att][1]:
                        delta_v = 0
                    else:
                        delta_v = - (selection_numeric[att][1] - row[att])
                else:  # selection_numeric[att][0] == "<=":
                    if row[att] > selection_numeric[att][1]:
                        delta_v = 0
                    else:
                        delta_v = - (selection_numeric[att][1] - row[att] + selection_numeric[att][2])

            if greater_than:
                delta_values.append(delta_v_zero)
            else:
                delta_values.append(delta_v)
        for att, domain in categorical_attributes.items():
            for value in domain:
                if value in selection_categorical[att]:
                    continue
                if greater_than:
                    if row[att] == value:
                        delta_values.append(1)
                    else:
                        delta_values.append(0)
                else:
                    if row[att] == value:
                        delta_values.append(-1)
                    else:
                        delta_values.append(0)
        # FIXME: should I remove all-zero rows here?
        # remove all-zeros columns when constraints are single-direction
        if not all(v == 0 for v in delta_values):
            list_of_rows_delta_table.append(delta_values)

    data_rows_greater_than = data_rows_greater_than.drop_duplicates(
        subset=selected_attributes,
        keep='first').reset_index(drop=True)
    data_rows_greater_than.apply(iterrow, args=(True,), axis=1)

    delta_table = pd.DataFrame(list_of_rows_delta_table, columns=columns_delta_table)

    delta_table = delta_table.drop_duplicates().reset_index(drop=True)
    categorical_att_columns = [item for item in columns_delta_table if item not in numeric_attributes]
    return delta_table, columns_delta_table, categorical_att_columns


def build_sorted_table_bidirectional(data, selected_attributes, numeric_attributes,
                                     categorical_attributes, selection_numeric, selection_categorical,
                                     sensitive_attributes, fairness_constraints,
                                     fairness_constraints_provenance_greater_than,
                                     fairness_constraints_provenance_smaller_than,
                                     data_rows_greater_than, data_rows_smaller_than
                                     ):
    """
    to build the sorted table
    :param fairness_constraints_provenance_greater_than:
    :param fairness_constraints_provenance_smaller_than:
    :param data_rows_smaller_than:
    :param data_rows_greater_than:
    :param selected_attributes:
    :param data: dataframe
    :param numeric_attributes: list of names of numeric attributes [city, major, state]
    :param categorical_attributes: dictionary: {city: [domain of city], major: [domain of major]}
    :param selection_numeric: dictionary: {grade:[80, >], age:[30, <], hours: [100, <=]}
    :param selection_categorical: dictionary: {city: [accepted cities], major: [accepted majors]}
    :return: return the whole sorted table, including rows that already satisfy the selection conditions;
            also return delta table
    """

    columns_delta_table = numeric_attributes.copy()
    for att, domain in categorical_attributes.items():
        for value in domain:
            columns_delta_table.append(att + "_" + value)

    list_of_rows_delta_table = []
    list_of_rows_delta_table_multifunctional = []
    row_idx = 0

    # build delta table
    def iterrow(row,
                greater_than=True):  # greater than is symbol in fairness constraint is greater than (relaxation term)
        delta_values_multifunctional = []  # for delta table where relaxation term has negative delta values recorded
        delta_values = []  # for delta table where relaxation term doesn't have negative delta values recorded
        for att in numeric_attributes:
            delta_v_zero = 0
            if greater_than:  # relaxation term
                if selection_numeric[att][0] == ">":
                    delta_v = selection_numeric[att][1] - row[att] + selection_numeric[att][2]
                    if row[att] > selection_numeric[att][1]:
                        delta_v_zero = 0
                    else:
                        delta_v_zero = delta_v
                elif selection_numeric[att][0] == ">=":
                    delta_v = selection_numeric[att][1] - row[att]
                    if row[att] >= selection_numeric[att][1]:
                        delta_v_zero = 0
                    else:
                        delta_v_zero = delta_v
                elif selection_numeric[att][0] == "<":
                    delta_v = row[att] - selection_numeric[att][1] + selection_numeric[att][2]
                    if row[att] < selection_numeric[att][1]:
                        delta_v_zero = 0
                    else:
                        delta_v_zero = delta_v
                else:
                    delta_v = row[att] - selection_numeric[att][1]
                    if row[att] <= selection_numeric[att][1]:
                        delta_v_zero = 0
                    else:
                        delta_v_zero = delta_v
            else:
                if selection_numeric[att][0] == ">":
                    if row[att] <= selection_numeric[att][1]:
                        delta_v = 0
                    else:
                        delta_v = - (row[att] - selection_numeric[att][1])
                elif selection_numeric[att][0] == ">=":
                    if row[att] < selection_numeric[att][1]:
                        delta_v = 0
                    else:
                        delta_v = - (row[att] - selection_numeric[att][1] + selection_numeric[att][2])
                elif selection_numeric[att][0] == "<":
                    if row[att] >= selection_numeric[att][1]:
                        delta_v = 0
                    else:
                        delta_v = - (selection_numeric[att][1] - row[att])
                else:  # selection_numeric[att][0] == "<=":
                    if row[att] > selection_numeric[att][1]:
                        delta_v = 0
                    else:
                        delta_v = - (selection_numeric[att][1] - row[att] + selection_numeric[att][2])
            delta_values_multifunctional.append(delta_v)
            if greater_than:
                delta_values.append(delta_v_zero)
            else:
                delta_values.append(delta_v)
        for att, domain in categorical_attributes.items():
            for value in domain:
                if greater_than:
                    if row[att] in selection_categorical[att]:
                        delta_values.append(0)
                        delta_values_multifunctional.append(0)
                    elif row[att] == value:
                        delta_values.append(1)
                        delta_values_multifunctional.append(1)
                    else:
                        delta_values.append(0)
                        delta_values_multifunctional.append(0)
                else:
                    if row[att] not in selection_categorical[att]:
                        delta_values.append(0)
                        delta_values_multifunctional.append(0)
                    elif row[att] == value:
                        delta_values.append(-1)
                        delta_values_multifunctional.append(-1)
                    else:  # FIXME: +-100 or 0?
                        delta_values.append(0)
                        delta_values_multifunctional.append(0)
        # FIXME: should I remove all-zero rows here?
        # remove all-zeros or non-all-zeros columns when constraints are single-direction
        delta_values.append(greater_than)
        delta_values_multifunctional.append(greater_than)
        nonlocal row_idx
        list_of_rows_delta_table.append(delta_values)
        list_of_rows_delta_table_multifunctional.append(delta_values_multifunctional)
        row_idx += 1

    data_rows_greater_than = data_rows_greater_than.drop_duplicates(
        subset=selected_attributes,
        keep='first').reset_index(drop=True)
    data_rows_greater_than.apply(iterrow, args=(True,), axis=1)
    data_rows_smaller_than = data_rows_smaller_than.drop_duplicates(
        subset=selected_attributes,
        keep='first').reset_index(drop=True)
    data_rows_smaller_than.apply(iterrow, args=(False,), axis=1)

    delta_table = pd.DataFrame(list_of_rows_delta_table, columns=columns_delta_table + ['relaxation_term'])

    delta_table_multifunctional = pd.DataFrame(list_of_rows_delta_table_multifunctional,
                                               columns=columns_delta_table + ['relaxation_term'])
    # FIXME: should I drop duplicates for delta table here?
    # sorted_table_by_column = dict()
    #
    # # sort first column
    # # s = delta_table[columns_delta_table].abs().to_records(index=False)
    # dtypes = [(columns_delta_table[i], delta_table.dtypes[columns_delta_table[i]])
    #           for i in range(len(columns_delta_table))]
    # s2 = list(delta_table[columns_delta_table].abs().itertuples(index=False))
    # s3 = np.array(s2, dtype=dtypes)
    # sorted_att_idx = np.argsort(s3, order=columns_delta_table)
    # sorted_table_by_column[columns_delta_table[0]] = sorted_att_idx
    #
    # # tiebreaker_col = delta_table[columns_delta_table[0]].abs()
    # tiebreaker_col = [0] * len(sorted_att_idx)
    # for k, v in enumerate(sorted_att_idx):
    #     tiebreaker_col[v] = k
    #
    # tiebreaker_dtype = delta_table.dtypes[columns_delta_table[0]]
    # for att in columns_delta_table[1:]:
    #     values_in_col = delta_table[att]
    #     s = np.array(list(zip(values_in_col.abs(),
    #                           tiebreaker_col)),
    #                  dtype=[('value', delta_table.dtypes[att]),
    #                         ('tiebreaker', tiebreaker_dtype)])
    #     sorted_att_idx = np.argsort(s, order=['value', 'tiebreaker'])
    #     sorted_table_by_column[att] = sorted_att_idx
    #
    # sorted_table = pd.DataFrame(data=sorted_table_by_column)
    # print("sorted_table:\n", sorted_table)
    categorical_att_columns = [item for item in columns_delta_table if item not in numeric_attributes]
    return delta_table, delta_table_multifunctional, columns_delta_table, categorical_att_columns


def assign_to_provenance_relax_only(value_assignment, numeric_attributes, categorical_attributes, selection_numeric,
                                    selection_categorical, columns_delta_table, num_columns,
                                    fairness_constraints_provenance_greater_than,
                                    fairness_constraints_provenance_smaller_than):
    if value_assignment == [0, 20, 0, 0]:
        print("value assignment = {}".format(value_assignment))
    va_dict = dict(zip(columns_delta_table, value_assignment))
    # greater than
    for fc in fairness_constraints_provenance_greater_than:
        sum = 0
        satisfy_this_fairness_constraint = False
        for pe in fc['provenance_expression']:
            fail = False
            for att in pe:
                if att == 'occurrence':
                    continue
                if pd.isnull(pe[att]):
                    fail = True
                    break
                if att in numeric_attributes:
                    if selection_numeric[att][0] == ">=" or selection_numeric[att][0] == ">":
                        after_refinement = selection_numeric[att][1] - va_dict[att]
                        if eval(str(pe[att]) + selection_numeric[att][0] + str(after_refinement)):
                            continue
                        else:
                            fail = True
                            break
                    else:
                        after_refinement = selection_numeric[att][1] + va_dict[att]
                        if eval(str(pe[att]) + selection_numeric[att][0] + str(after_refinement)):
                            continue
                        else:
                            fail = True
                            break
                else:  # att in categorical
                    column_name = att + "_" + pe[att]
                    if column_name not in columns_delta_table:
                        continue
                    if pe[att] in selection_categorical[att]:
                        if 1 + va_dict[column_name] == 1:
                            continue
                        else:
                            fail = True
                            break
                    else:
                        if va_dict[column_name] == 1:
                            continue
                        else:
                            fail = True
                            break
            if not fail:
                sum += pe['occurrence']
                if eval(str(sum) + fc['symbol'] + str(fc['number'])):
                    satisfy_this_fairness_constraint = True
                    break
        if not satisfy_this_fairness_constraint:
            return False
    return True


def assign_to_provenance(value_assignment, numeric_attributes, categorical_attributes, selection_numeric,
                         selection_categorical, columns_delta_table, num_columns,
                         fairness_constraints_provenance_greater_than,
                         fairness_constraints_provenance_smaller_than):
    va_dict = dict(zip(columns_delta_table, value_assignment))
    if value_assignment == [0, 4, 1, 1, 0]:
        print("value_assignment = {}".format(value_assignment))
    # va_dict = value_assignment.to_dict()
    # greater than

    def assign(fairness_constraints_provenance, relaxation=True):
        for fc in fairness_constraints_provenance:
            sum = 0
            satisfy_this_fairness_constraint = not relaxation
            for pe in fc['provenance_expression']:
                fail = False
                for att in pe:
                    if att == 'occurrence':
                        continue
                    if pd.isnull(pe[att]):
                        fail = True
                        break
                    if att in numeric_attributes:
                        if selection_numeric[att][0] == ">=" or selection_numeric[att][0] == ">":
                            after_refinement = selection_numeric[att][1] - va_dict[att]
                            if eval(str(pe[att]) + selection_numeric[att][0] + str(after_refinement)):
                                continue
                            else:
                                fail = True
                                break
                        else:
                            after_refinement = selection_numeric[att][1] + va_dict[att]
                            if eval(str(pe[att]) + selection_numeric[att][0] + str(after_refinement)):
                                continue
                            else:
                                fail = True
                                break
                    else:  # att in categorical
                        column_name = att + "_" + pe[att]
                        if column_name not in columns_delta_table:
                            continue
                        if pe[att] in selection_categorical[att]:
                            if 1 + va_dict[column_name] == 1:
                                continue
                            else:
                                fail = True
                                break
                        else:
                            if va_dict[column_name] == 1:
                                continue
                            else:
                                fail = True
                                break
                if not fail:
                    sum += pe['occurrence']
                    if relaxation:
                        if eval(str(sum) + fc['symbol'] + str(fc['number'])):
                            satisfy_this_fairness_constraint = True
                            break
                    else:
                        if not eval(str(sum) + fc['symbol'] + str(fc['number'])):
                            satisfy_this_fairness_constraint = False
                            break
            if not satisfy_this_fairness_constraint:
                return False
        return True

    survive = assign(fairness_constraints_provenance_greater_than, relaxation=True)
    if not survive:
        return False
    survive = assign(fairness_constraints_provenance_smaller_than, relaxation=False)
    return survive


def get_relaxation(terms, delta_table):
    """
To get skyline of all terms in list terms
    :param terms: list of indices of terms. [1,3,5]
    :param delta_table: delta table returned by func build_sorted_table()
    :return: return Ture or false whether terms can have a legitimate value assignment .
    """
    column_names = delta_table.columns.tolist()
    value_assignment = delta_table[column_names].loc[terms].max().tolist()
    return value_assignment


def get_refinement(terms, delta_table, delta_table_multifunctional):
    """
To get skyline of all terms in list terms
    :param terms: list of indices of terms. [1,3,5]
    :param delta_table: delta table returned by func build_sorted_table()
    :return: return Ture or false whether terms can have a legitimate value assignment .
    """
    column_names = delta_table.columns.tolist()

    column_names.remove('relaxation_term')
    num_col = len(column_names) + 1

    value_assignments = [[]]
    mark_satisfied_terms = [0]

    rows_to_compare = delta_table.loc[terms]
    rows_to_compare_with_multifunctional = delta_table_multifunctional.loc[terms]
    rows_to_compare_indices = rows_to_compare.index.tolist()

    rows_to_compare.round(2)
    rows_to_compare_with_multifunctional.round(2)

    all_included = 0
    for i in rows_to_compare_indices:
        all_included = (1 << i) | all_included
    for i in range(num_col):
        if i == num_col - 1:
            num_result = len(value_assignments)
            j = 0
            while j < num_result:
                if mark_satisfied_terms[j] != all_included:
                    del value_assignments[j]
                    del mark_satisfied_terms[j]
                    num_result -= 1
                else:
                    j += 1
            return (len(value_assignments) != 0), value_assignments
        col = column_names[i]
        max_in_col = rows_to_compare[col].max()
        if max_in_col > 0:
            #  term_w_max = rows_to_compare[col].idxmax(axis=0)
            for va in value_assignments:
                va.append(max_in_col)
            positive_terms_int = 0
            indices = rows_to_compare[rows_to_compare[col] > 0].index.tolist()
            for d in indices:
                positive_terms_int = (1 << d) | positive_terms_int
            for k in range(len(mark_satisfied_terms)):
                mark_satisfied_terms[k] |= positive_terms_int
            continue
        unique_values = rows_to_compare[col].drop_duplicates()
        non_zeros = unique_values[unique_values < 0].index.tolist()
        relaxation_terms_idx = rows_to_compare[rows_to_compare['relaxation_term']].index.tolist()
        if len(non_zeros) == 0:
            for v in value_assignments:
                v.append(0)
            continue
        mark_satisfied_terms_new = []
        value_assignments_new = []
        for n in non_zeros:
            positive_terms_int = (1 << n)  # | positive_terms_int
            for k in range(len(mark_satisfied_terms)):
                if mark_satisfied_terms[k] & positive_terms_int == 0:
                    maximum_to_contract = rows_to_compare_with_multifunctional[col].loc[relaxation_terms_idx].min()
                    maximum_to_contract = round(maximum_to_contract, 2)
                    b = round(rows_to_compare.loc[n][col], 2)
                    if maximum_to_contract == 0 or maximum_to_contract <= b:
                        v_ = value_assignments[k].copy()
                        v_.append(rows_to_compare.loc[n][col])
                        value_assignments_new.append(v_)
                        covered = [x for x in non_zeros if rows_to_compare.loc[x][col] >= rows_to_compare.loc[n][col]]
                        to_append = mark_satisfied_terms[k]
                        for c in covered:
                            to_append |= 1 << c
                        mark_satisfied_terms_new.append(to_append)

        for s in value_assignments:
            s.append(0)
            value_assignments_new.append(s)
        mark_satisfied_terms_new = mark_satisfied_terms_new + mark_satisfied_terms
        value_assignments = value_assignments_new.copy()
        mark_satisfied_terms = mark_satisfied_terms_new.copy()

    return True, value_assignments


def dominate(a, b):
    """
    whether relaxation a dominates b. relaxation has all delta values. it is not a selection condition
    :param a: relaxation, format of sorted table
    :param b: relaxation
    :return: true if a dominates b (a is minimal compared to b), return true if equal
    """
    if a == b:
        return True
    non_zero_a = sum(x != 0 for x in a)
    non_zero_b = sum(x != 0 for x in b)
    if non_zero_a > non_zero_b:
        return False
    length = len(a)
    for i in range(length):
        if abs(a[i]) > abs(b[i]):
            return False
    return True


# for relaxation term, don't need to check terms with same value in this column again
# put 0 and 100 before +-1, stop line is the first +-1
def update_stop_line_relax_only(combo_w_t, stop_line, sorted_table, delta_table,
                                delta_table_multifunctional, columns_delta_table, value_assignment):
    column_idx = 0

    def itercol(column):
        nonlocal column_idx
        nonlocal sorted_table
        col_name = columns_delta_table[column_idx]
        col_idx = 0
        found = False
        for k, v in column.items():
            if value_assignment[column_idx] == delta_table[col_name].loc[v]:
                col_idx = k
                found = True
                break
        if not found:
            col_idx = len(delta_table) - 1
        column_idx += 1
        return col_idx

    new_stop_line = sorted_table.apply(itercol, axis=0)
    return new_stop_line


def update_stop_line_bidirectional(combo_w_t, stop_line, sorted_table, delta_table,
                                   delta_table_multifunctional, columns_delta_table, value_assignment):
    column_idx = 0
    rows_to_compare = delta_table.loc[combo_w_t]
    relaxation_terms = rows_to_compare[rows_to_compare['relaxation_term']].index.tolist()
    contraction_terms = [x for x in combo_w_t if x not in relaxation_terms]

    def itercol_relaxation(column):
        nonlocal column_idx
        if value_assignment[column_idx] <= 0:
            column_idx += 1
            return -1
        col_name = columns_delta_table[column_idx]
        col_idx = 0
        found = False
        for k, v in column.items():
            if value_assignment[column_idx] == delta_table[col_name].loc[v]:
                col_idx = k
                found = True
                break
        if not found:
            col_idx = len(delta_table) - 1
        column_idx += 1
        return col_idx

    def itercol_contraction(column):
        nonlocal column_idx
        col_name = columns_delta_table[column_idx]
        if value_assignment[column_idx] > 0:
            column_idx += 1
            return new_stop_line[col_name]
        to_compare = delta_table[col_name].loc[contraction_terms].min()
        col_idx = 0
        found = False
        for k, v in column.items():
            if - to_compare == abs(delta_table[col_name].loc[v]):
                col_idx = k
                found = True
                break
        if not found:
            col_idx = len(delta_table) - 1
        column_idx += 1
        return col_idx

    new_stop_line = sorted_table.apply(itercol_relaxation, axis=0)
    column_idx = 0
    new_stop_line = sorted_table.apply(itercol_contraction, axis=0)
    # print("new_stop_line:\n{}".format(new_stop_line))
    return new_stop_line


def dominated_by_minimal_set(minimal_added_relaxations, r):
    for mr in minimal_added_relaxations:
        if mr == r:
            return True
        if dominate(mr, r):
            return True
    return False


def update_minimal_relaxation(minimal_added_relaxations, r):
    dominated = []
    for mr in minimal_added_relaxations:
        if mr == r:
            return True, minimal_added_relaxations
        if dominate(mr, r):
            return False, minimal_added_relaxations
        elif dominate(r, mr):
            dominated.append(mr)
    if len(dominated) > 0:
        minimal_added_relaxations = [x for x in minimal_added_relaxations if x not in dominated]
    minimal_added_relaxations.append(r)
    return True, minimal_added_relaxations


# assume there is no stop line set yet
def set_stop_line_and_resort_relax_only(value_assignment, index_of_columns_remained, term_set, stop_line, sorted_table,
                                        delta_table, columns_delta_table, only_first_column_to_sort, row_num,
                                        columns_resort,
                                        minimal_added_relaxations,
                                        original_columns_delta_table,
                                        checked_satisfying_constraints,
                                        checked_unsatisfying_constraints,
                                        checked_assignments_satisfying,
                                        checked_assignments_unsatisfying,
                                        numeric_attributes,
                                        categorical_att_columns, categorical_attributes,
                                        selection_numeric, selection_categorical,
                                        fairness_constraints_provenance_greater_than,
                                        fairness_constraints_provenance_smaller_than
                                        ):
    value_assignment_remained_columns = [value_assignment[i] for i in
                                         index_of_columns_remained]
    stop_line = update_stop_line_relax_only(term_set, stop_line,
                                            sorted_table, delta_table[columns_delta_table],
                                            delta_table[columns_delta_table],
                                            columns_delta_table,
                                            value_assignment_remained_columns)
    # print("row_num: {}, value_assignment: {}".format(row_num, value_assignment))
    # print("stop line:\n{}".format(stop_line))
    if not only_first_column_to_sort:
        for col in columns_delta_table[1:]:
            if stop_line[col] > row_num + 1:
                columns_resort.add(col)
                terms_above_stop_line = list(sorted_table.loc[:stop_line[col] - 1, col])
                # new_columns = columns_delta_table.copy()
                # new_columns.remove(col)
                # index_of_columns_remained_without_col = [x for x in index_of_columns_remained
                #                              if original_columns_delta_table[x] != col]
                minimal_added_relaxations = resort_and_search_relax_only(terms_above_stop_line,
                                                                         delta_table.loc[terms_above_stop_line],
                                                                         columns_delta_table, index_of_columns_remained,
                                                                         minimal_added_relaxations,
                                                                         original_columns_delta_table,
                                                                         checked_satisfying_constraints,
                                                                         checked_unsatisfying_constraints,
                                                                         checked_assignments_satisfying,
                                                                         checked_assignments_unsatisfying,
                                                                         numeric_attributes,
                                                                         categorical_att_columns,
                                                                         categorical_attributes,
                                                                         selection_numeric, selection_categorical,
                                                                         fairness_constraints_provenance_greater_than,
                                                                         fairness_constraints_provenance_smaller_than)
        # for first column, don't need to resort since the rank won't change
        if stop_line[columns_delta_table[0]] > row_num + 1:
            for col in columns_delta_table[1:]:
                if stop_line[col] > row_num + 1:
                    stop_line[col] = 0
            return True, stop_line, minimal_added_relaxations
    return False, stop_line, minimal_added_relaxations


# TODO
# assume there is no stop line set yet
def set_stop_line_and_resort_bidirectional(value_assignment, index_of_columns_remained, term_set, stop_line,
                                           sorted_table,
                                           delta_table, delta_table_multifunctional, columns_delta_table,
                                           only_first_column_to_sort, row_num, columns_resort,
                                           minimal_added_refinements,
                                           original_columns_delta_table,
                                           checked_satisfying_constraints,
                                           checked_unsatisfying_constraints,
                                           checked_assignments_satisfying,
                                           checked_assignments_unsatisfying, checked_single_directional_combination,
                                           numeric_attributes,
                                           categorical_att_columns, categorical_attributes,
                                           selection_numeric, selection_categorical,
                                           fairness_constraints_provenance_greater_than,
                                           fairness_constraints_provenance_smaller_than
                                           ):
    value_assignment_remained_columns = [value_assignment[i] for i in
                                         index_of_columns_remained]
    stop_line = update_stop_line_bidirectional(term_set, stop_line,
                                               sorted_table, delta_table[columns_delta_table + ['relaxation_term']],
                                               delta_table_multifunctional[columns_delta_table + ['relaxation_term']],
                                               columns_delta_table,
                                               value_assignment_remained_columns)
    # print("row_num: {}, value_assignment: {}".format(row_num, value_assignment))
    # print("stop line:\n{}".format(stop_line))
    if not only_first_column_to_sort:
        for col in columns_delta_table[1:]:
            if stop_line[col] > row_num + 1:
                columns_resort.add(col)
                terms_above_stop_line = list(sorted_table.loc[:stop_line[col] - 1, col])
                # new_columns = columns_delta_table.copy()
                # new_columns.remove(col)
                # index_of_columns_remained_without_col = [x for x in index_of_columns_remained
                #                              if original_columns_delta_table[x] != col]
                minimal_added_refinements = resort_and_search_bidirectional(terms_above_stop_line,
                                                                            delta_table.loc[terms_above_stop_line],
                                                                            delta_table_multifunctional.loc[terms_above_stop_line],
                                                                            columns_delta_table,
                                                                            index_of_columns_remained,
                                                                            minimal_added_refinements,
                                                                            original_columns_delta_table,
                                                                            checked_satisfying_constraints,
                                                                            checked_unsatisfying_constraints,
                                                                            checked_assignments_satisfying,
                                                                            checked_assignments_unsatisfying,
                                                                            checked_single_directional_combination,
                                                                            numeric_attributes,
                                                                            categorical_att_columns,
                                                                            categorical_attributes,
                                                                            selection_numeric, selection_categorical,
                                                                            fairness_constraints_provenance_greater_than,
                                                                            fairness_constraints_provenance_smaller_than)
        # for first column, don't need to resort since the rank won't change
        if stop_line[columns_delta_table[0]] > row_num + 1:
            for col in columns_delta_table[1:]:
                if stop_line[col] > row_num + 1:
                    stop_line[col] = 0
            return True, stop_line, minimal_added_refinements
    return False, stop_line, minimal_added_refinements


def resort_and_search_bidirectional(terms, delta_table, delta_table_multifunctional, columns_delta_table,
                                    index_of_columns_remained,
                                    minimal_added_refinements, original_columns_delta_table,
                                    checked_satisfying_constraints,
                                    checked_unsatisfying_constraints, checked_assignments_satisfying,
                                    checked_assignments_unsatisfying, checked_single_directional_combination,
                                    numeric_attributes,
                                    categorical_att_columns, categorical_attributes, selection_numeric,
                                    selection_categorical,
                                    fairness_constraints_provenance_greater_than,
                                    fairness_constraints_provenance_smaller_than):
    # if all values in a column are the same, delete this column
    nunique = delta_table[columns_delta_table].nunique()

    print("delta_table:\n{}".format(delta_table))

    cols_to_drop = nunique[nunique == 1].index
    # delta_table.drop(cols_to_drop, axis=1, inplace=True)
    columns_delta_table = [x for x in columns_delta_table if x not in cols_to_drop]
    index_of_columns_remained = [i for i in index_of_columns_remained if
                                 original_columns_delta_table[i] not in cols_to_drop]
    # delta table drop duplicates
    # index of first of duplicates
    first_duplicates_idx = ~delta_table.duplicated(subset=columns_delta_table) & \
                           delta_table.duplicated(subset=columns_delta_table, keep=False)
    first_duplicates_idx = first_duplicates_idx[first_duplicates_idx].index
    if len(first_duplicates_idx) > 0:
        def get_min_negative_values(row):
            print("row {}".format(row))
            nonlocal delta_table
            nonlocal delta_table_multifunctional
            p = row[row > 0]
            dct = p.to_dict()
            rows_duplicate = delta_table_multifunctional[np.logical_and.reduce([delta_table_multifunctional[k] == v
                                                                                for k, v in dct.items()])]
            max_values = rows_duplicate.max()
            print(max_values)
            return max_values

        delta_table_multifunctional.loc[first_duplicates_idx] = \
            delta_table_multifunctional.loc[first_duplicates_idx].apply(get_min_negative_values, axis=1)

        duplicates_to_drop_idx = delta_table.duplicated(subset=columns_delta_table, keep="first")
        duplicates_to_drop_idx = duplicates_to_drop_idx[duplicates_to_drop_idx].index
        delta_table_drop_duplicates = delta_table.drop(duplicates_to_drop_idx)
        delta_table_multifunctional_drop_duplicates = delta_table_multifunctional.drop(duplicates_to_drop_idx)
    else:
        delta_table_drop_duplicates = delta_table
        delta_table_multifunctional_drop_duplicates = delta_table_multifunctional

    sorted_table_by_column = dict()
    # sort first column
    row_indices = delta_table_drop_duplicates.index.values
    # s = delta_table[columns_delta_table].to_records(index=False)

    dtypes = [(columns_delta_table[i], delta_table_drop_duplicates.dtypes[columns_delta_table[i]]) for i in
              range(len(columns_delta_table))]
    s2 = list(delta_table_drop_duplicates[columns_delta_table].abs().itertuples(index=False))
    s3 = np.array(s2, dtype=dtypes)

    sorted_att_idx = np.argsort(s3, order=columns_delta_table)
    sorted_delta_idx = row_indices[sorted_att_idx]
    sorted_table_by_column[columns_delta_table[0]] = sorted_delta_idx

    tiebreaker_col = [0] * len(sorted_att_idx)
    for k, v in enumerate(sorted_att_idx):
        tiebreaker_col[v] = k
    # tiebreaker_col = delta_table[columns_delta_table[0]]

    tiebreaker_dtype = delta_table_drop_duplicates.dtypes[columns_delta_table[0]]
    for att in columns_delta_table[1:]:
        values_in_col = delta_table_drop_duplicates[att].abs()
        s = np.array(list(zip(values_in_col,
                              tiebreaker_col)),
                     dtype=[('value', delta_table_drop_duplicates.dtypes[att]),
                            ('tiebreaker', tiebreaker_dtype)])
        sorted_att_idx = np.argsort(s, order=['value', 'tiebreaker'])
        sorted_delta_idx = row_indices[sorted_att_idx]
        sorted_table_by_column[att] = sorted_delta_idx
    sorted_table = pd.DataFrame(data=sorted_table_by_column)
    categorical_att_columns = [item for item in columns_delta_table if item not in numeric_attributes]
    row_num = 0
    only_first_column_to_sort = False
    set_stop_line = False
    num_columns = len(columns_delta_table)
    stop_line = pd.Series([len(sorted_table)] * num_columns, sorted_table.columns)
    columns_resort = set()

    print("resort_and_search_bidirectional, num of terms = {}".format(len(terms)))
    print("delta_table_drop_duplicates:\n{}".format(delta_table_drop_duplicates))
    print("delta_table_multifunctional_drop_duplicates:\n{}".format(delta_table_multifunctional_drop_duplicates))
    print("sorted_table:\n{}".format(sorted_table))

    def iterrow(row):
        global value_assignment
        nonlocal row_num
        nonlocal set_stop_line
        nonlocal stop_line
        nonlocal minimal_added_refinements
        nonlocal index_of_columns_remained
        nonlocal only_first_column_to_sort
        for i, t in row.items():
            if stop_line[i] <= row_num:
                continue
            if i in columns_resort:
                continue
            print("row_num={}".format(row_num))
            assign_successfully = False
            # if row_num == 7:
            #     print("stop here")
            terms_above = list(sorted_table.loc[:row_num - 1, i])
            combo_w_t = [t] + terms_above
            combo_str = intbitset(combo_w_t).strbits()
            # check whether all these terms satisfies constraints, if not, no need to check each smaller one
            # optimization 1: if all terms in the set are relaxation/contraction terms, skip
            terms_set = delta_table.loc[combo_w_t]
            # if terms_set['relaxation_term'].eq(True).all() or terms_set['relaxation_term'].eq(False).all():
            #     checked_single_directional_combination.add(combo_str)
            #     continue
            # if combo_str in checked_unsatisfying_constraints:
            #     continue
            # have_legitimate_value_assignment, value_assignments = \
            #     get_refinement(combo_w_t, delta_table_drop_duplicates, delta_table_multifunctional_drop_duplicates)
            # if not have_legitimate_value_assignment:
            #     continue
            # for value_assignment in value_assignments:
            #     if value_assignment in checked_assignments_satisfying:
            #         checked_satisfying_constraints.add(combo_str)
            #     elif value_assignment in checked_assignments_unsatisfying:
            #         checked_unsatisfying_constraints.add(combo_str)
            #         continue
            #     elif value_assignment in minimal_added_refinements:
            #         continue
            #     if value_assignment not in checked_assignments_satisfying:
            #         if not assign_to_provenance(value_assignment, numeric_attributes,
            #                                     categorical_attributes,
            #                                     selection_numeric, selection_categorical,
            #                                     original_columns_delta_table, num_columns,
            #                                     fairness_constraints_provenance_greater_than,
            #                                     fairness_constraints_provenance_smaller_than):
            #             checked_assignments_unsatisfying.append(value_assignment)
            #             checked_unsatisfying_constraints.add(combo_str)
            #             value_assignments.remove(value_assignment)
            #             continue
            #         else:
            #             checked_satisfying_constraints.add(combo_str)
            #             checked_assignments_satisfying.append(value_assignment)
            #     print("row_num = {}, terms above = {}".format(row_num, [t] + terms_above))
            #     print("terms above satisfy, value assignment = {}".format(value_assignment))
            #     checked_assignments_satisfying.append(value_assignment)
            #     this_is_minimal, minimal_added_refinements = \
            #         update_minimal_relaxation(minimal_added_refinements, value_assignment)
            #
            #     # check whether value assignment for this term and terms above is tight
            #     # if yes, no need to check subsets of them
            #     if whether_value_assignment_is_tight_minimal(delta_table_drop_duplicates.loc[t].values.tolist(),
            #                                                  value_assignment,
            #                                                  numeric_attributes, categorical_attributes,
            #                                                  selection_numeric, selection_categorical,
            #                                                  original_columns_delta_table, num_columns,
            #                                                  fairness_constraints_provenance_greater_than,
            #                                                  fairness_constraints_provenance_smaller_than,
            #                                                  minimal_added_refinements):
            #         value_assignments.remove(value_assignment)
            #
            # if len(value_assignments) == 0:
            #     print("no need to check subsets")
            #     continue
            # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
            #
            # if not set_stop_line:
            #     set_stop_line = True
            #     only_first_column_to_sort, stop_line, minimal_added_refinements = \
            #         set_stop_line_and_resort_bidirectional(value_assignment, index_of_columns_remained,
            #                                                [t] + terms_above,
            #                                                stop_line, sorted_table,
            #                                                delta_table_drop_duplicates,
            #                                                delta_table_multifunctional_drop_duplicates,
            #                                                columns_delta_table, only_first_column_to_sort,
            #                                                row_num, columns_resort,
            #                                                minimal_added_refinements,
            #                                                original_columns_delta_table,
            #                                                checked_satisfying_constraints,
            #                                                checked_unsatisfying_constraints,
            #                                                checked_assignments_satisfying,
            #                                                checked_assignments_unsatisfying,
            #                                                numeric_attributes,
            #                                                categorical_att_columns, categorical_attributes,
            #                                                selection_numeric, selection_categorical,
            #                                                fairness_constraints_provenance_greater_than,
            #                                                fairness_constraints_provenance_smaller_than
            #                                                )
            #     continue
            combo_list = [[y] for y in terms_above]
            while len(combo_list) > 0:
                combo: List[Any] = combo_list.pop(0)
                combo_str = intbitset(combo).strbits()
                if combo_str in checked_satisfying_constraints:
                    continue
                combo_w_t = [t] + combo
                combo_str = intbitset(combo_w_t).strbits()
                terms_set = delta_table.loc[combo_w_t]
                if terms_set['relaxation_term'].eq(True).all() or terms_set['relaxation_term'].eq(False).all():
                    checked_single_directional_combination.add(combo_str)
                    continue
                if combo_str in checked_satisfying_constraints:
                    continue
                if combo_str in checked_unsatisfying_constraints:
                    # generate children
                    idx = terms_above.index(combo[-1])
                    for x in terms_above[idx + 1:]:
                        combo_list.append(combo + [x])
                    continue

                have_legitimate_value_assignment, value_assignments = \
                    get_refinement(combo_w_t, delta_table_drop_duplicates, delta_table_multifunctional_drop_duplicates)
                if not have_legitimate_value_assignment:
                    continue
                satisfying_value_assignments = []
                for value_assignment in value_assignments:
                    if value_assignment in checked_assignments_satisfying:
                        checked_satisfying_constraints.add(combo_str)
                    elif value_assignment in checked_assignments_unsatisfying:
                        checked_unsatisfying_constraints.add(combo_str)
                        continue
                    elif value_assignment in minimal_added_refinements:
                        continue
                    if value_assignment not in checked_assignments_satisfying:
                        if not assign_to_provenance(value_assignment, numeric_attributes,
                                                    categorical_attributes,
                                                    selection_numeric, selection_categorical,
                                                    original_columns_delta_table, num_columns,
                                                    fairness_constraints_provenance_greater_than,
                                                    fairness_constraints_provenance_smaller_than):
                            checked_assignments_unsatisfying.append(value_assignment)
                            checked_unsatisfying_constraints.add(combo_str)
                            continue
                        else:
                            checked_satisfying_constraints.add(combo_str)
                            checked_assignments_satisfying.append(value_assignment)
                            this_is_minimal, minimal_added_refinements = \
                                update_minimal_relaxation(minimal_added_refinements, value_assignment)
                            satisfying_value_assignments.append(value_assignment)
                if len(satisfying_value_assignments) > 0:
                    if not set_stop_line:
                        set_stop_line = True
                        only_first_column_to_sort, stop_line, minimal_added_refinements = \
                            set_stop_line_and_resort_bidirectional(satisfying_value_assignments[0],
                                                                   index_of_columns_remained,
                                                                   combo_w_t,
                                                                   stop_line, sorted_table,
                                                                   delta_table_drop_duplicates,
                                                                   delta_table_multifunctional_drop_duplicates,
                                                                   columns_delta_table, only_first_column_to_sort,
                                                                   row_num, columns_resort,
                                                                   minimal_added_refinements,
                                                                   original_columns_delta_table,
                                                                   checked_satisfying_constraints,
                                                                   checked_unsatisfying_constraints,
                                                                   checked_assignments_satisfying,
                                                                   checked_assignments_unsatisfying,
                                                                   checked_single_directional_combination,
                                                                   numeric_attributes,
                                                                   categorical_att_columns, categorical_attributes,
                                                                   selection_numeric, selection_categorical,
                                                                   fairness_constraints_provenance_greater_than,
                                                                   fairness_constraints_provenance_smaller_than)
                        break
                else:  # if doesn't satisfy provenance constraints
                    checked_unsatisfying_constraints.add(combo_str)
                    checked_assignments_unsatisfying.append(value_assignment)
                    # generate children
                    idx = terms_above.index(combo[-1])
                    for x in terms_above[idx + 1:]:
                        combo_list.append(combo + [x])
            # end of while
        row_num += 1
        # end of for columns

    sorted_table.apply(iterrow, axis=1)
    return minimal_added_refinements


def resort_and_search_relax_only(terms, delta_table, columns_delta_table, index_of_columns_remained,
                                 minimal_added_relaxations, original_columns_delta_table,
                                 checked_satisfying_constraints,
                                 checked_unsatisfying_constraints, checked_assignments_satisfying,
                                 checked_assignments_unsatisfying,
                                 numeric_attributes,
                                 categorical_att_columns, categorical_attributes, selection_numeric,
                                 selection_categorical,
                                 fairness_constraints_provenance_greater_than,
                                 fairness_constraints_provenance_smaller_than):
    # if all values in a column are the same, delete this column
    nunique = delta_table[columns_delta_table].nunique()
    cols_to_drop = nunique[nunique == 1].index
    # delta_table.drop(cols_to_drop, axis=1, inplace=True)
    columns_delta_table = [x for x in columns_delta_table if x not in cols_to_drop]
    index_of_columns_remained = [i for i in index_of_columns_remained if
                                 original_columns_delta_table[i] not in cols_to_drop]

    # delta table drop duplicates
    delta_table_dropped_duplicates = delta_table.drop_duplicates(subset=columns_delta_table)

    sorted_table_by_column = dict()
    # sort first column
    row_indices = delta_table_dropped_duplicates.index.values
    # s = delta_table_dropped_duplicates[columns_delta_table].to_records(index=False)

    dtypes = [(columns_delta_table[i], delta_table_dropped_duplicates.dtypes[columns_delta_table[i]]) for i in
              range(len(columns_delta_table))]
    s2 = list(delta_table_dropped_duplicates[columns_delta_table].itertuples(index=False))
    s3 = np.array(s2, dtype=dtypes)

    sorted_att_idx = np.argsort(s3, order=columns_delta_table)
    sorted_delta_idx = row_indices[sorted_att_idx]
    sorted_table_by_column[columns_delta_table[0]] = sorted_delta_idx

    tiebreaker_col = [0] * len(sorted_att_idx)
    for k, v in enumerate(sorted_att_idx):
        tiebreaker_col[v] = k
    # tiebreaker_col = delta_table_dropped_duplicates[columns_delta_table[0]]

    tiebreaker_dtype = delta_table_dropped_duplicates.dtypes[columns_delta_table[0]]
    for att in columns_delta_table[1:]:
        values_in_col = delta_table_dropped_duplicates[att]
        s = np.array(list(zip(values_in_col,
                              tiebreaker_col)),
                     dtype=[('value', delta_table_dropped_duplicates.dtypes[att]),
                            ('tiebreaker', tiebreaker_dtype)])
        sorted_att_idx = np.argsort(s, order=['value', 'tiebreaker'])
        sorted_delta_idx = row_indices[sorted_att_idx]
        sorted_table_by_column[att] = sorted_delta_idx
    sorted_table = pd.DataFrame(data=sorted_table_by_column)
    categorical_att_columns = [item for item in columns_delta_table if item not in numeric_attributes]
    row_num = 0
    only_first_column_to_sort = False
    set_stop_line = False
    num_columns = len(columns_delta_table)
    stop_line = pd.Series([len(sorted_table)] * num_columns, sorted_table.columns)
    columns_resort = set()

    print("resort_and_search_relax_only, num of terms = {}".format(len(terms)))
    print("sorted_table = \n{}".format(sorted_table))
    print("delta_table = \n{}".format(delta_table))

    def iterrow(row):
        global value_assignment
        nonlocal row_num
        nonlocal set_stop_line
        nonlocal stop_line
        nonlocal minimal_added_relaxations
        nonlocal index_of_columns_remained
        nonlocal only_first_column_to_sort
        for i, t in row.items():
            if stop_line[i] <= row_num:
                continue
            if i in columns_resort:
                continue
            # print("row_num={}".format(row_num))
            assign_successfully = False
            t_str = '0' * t + '1' + '0' * (num_columns - 1 - t)
            if t_str not in checked_satisfying_constraints \
                    and t_str not in checked_unsatisfying_constraints:
                value_assignment = get_relaxation([t], delta_table_dropped_duplicates)
                if value_assignment in checked_assignments_satisfying:
                    checked_satisfying_constraints.add(t_str)
                    continue
                elif value_assignment in checked_assignments_unsatisfying:
                    checked_unsatisfying_constraints.add(t_str)
                else:
                    if assign_to_provenance_relax_only(value_assignment, numeric_attributes, categorical_attributes,
                                                       selection_numeric, selection_categorical,
                                                       original_columns_delta_table,
                                                       num_columns, fairness_constraints_provenance_greater_than,
                                                       fairness_constraints_provenance_smaller_than):
                        assign_successfully = True
                        checked_satisfying_constraints.add(t_str)
                        checked_assignments_satisfying.append(value_assignment)

                        this_is_minimal, minimal_added_relaxations = \
                            update_minimal_relaxation(minimal_added_relaxations, value_assignment)
                        # even if this is not minimal, it still gives a stop line
                        if not set_stop_line:
                            set_stop_line = True
                            only_first_column_to_sort, stop_line, minimal_added_relaxations = \
                                set_stop_line_and_resort_relax_only(value_assignment, index_of_columns_remained, [t],
                                                                    stop_line, sorted_table,
                                                                    delta_table_dropped_duplicates, columns_delta_table,
                                                                    only_first_column_to_sort,
                                                                    row_num, columns_resort,
                                                                    minimal_added_relaxations,
                                                                    original_columns_delta_table,
                                                                    checked_satisfying_constraints,
                                                                    checked_unsatisfying_constraints,
                                                                    checked_assignments_satisfying,
                                                                    checked_assignments_unsatisfying,
                                                                    numeric_attributes,
                                                                    categorical_att_columns, categorical_attributes,
                                                                    selection_numeric, selection_categorical,
                                                                    fairness_constraints_provenance_greater_than,
                                                                    fairness_constraints_provenance_smaller_than
                                                                    )
                    else:
                        checked_unsatisfying_constraints.add(t_str)
                        checked_assignments_unsatisfying.append(value_assignment)
            if assign_successfully:
                continue
            terms_above = list(sorted_table.loc[:row_num - 1, i])
            # check whether all these terms satisfies constraints, if not, no need to check each smaller one
            combo_w_t = [t] + terms_above
            combo_str = intbitset(combo_w_t).strbits()
            if combo_str in checked_unsatisfying_constraints:
                continue
            value_assignment = get_relaxation(combo_w_t, delta_table_dropped_duplicates)
            if value_assignment in checked_assignments_satisfying:
                checked_satisfying_constraints.add(combo_str)
            elif value_assignment in checked_assignments_unsatisfying:
                checked_unsatisfying_constraints.add(combo_str)
                continue
            elif value_assignment in minimal_added_relaxations:
                continue
            if value_assignment not in checked_assignments_satisfying:
                if not assign_to_provenance_relax_only(value_assignment, numeric_attributes,
                                                       categorical_attributes,
                                                       selection_numeric, selection_categorical,
                                                       original_columns_delta_table, num_columns,
                                                       fairness_constraints_provenance_greater_than,
                                                       fairness_constraints_provenance_smaller_than):
                    checked_assignments_unsatisfying.append(value_assignment)
                    checked_unsatisfying_constraints.add(combo_str)
                    print("terms above don't satisfy")
                    continue
                else:
                    checked_satisfying_constraints.add(combo_str)
                    checked_assignments_satisfying.append(value_assignment)
            print("row_num = {}, terms above = {}".format(row_num, [t] + terms_above))
            print("terms above satisfy, value assignment = {}".format(value_assignment))
            checked_assignments_satisfying.append(value_assignment)
            this_is_minimal, minimal_added_relaxations = \
                update_minimal_relaxation(minimal_added_relaxations, value_assignment)
            # check whether value assignment for this term and terms above is tight
            # if yes, no need to check subsets of them
            if whether_value_assignment_is_tight_minimal(delta_table_dropped_duplicates.loc[t].values.tolist(),
                                                         value_assignment,
                                                         numeric_attributes, categorical_attributes,
                                                         selection_numeric, selection_categorical,
                                                         original_columns_delta_table, num_columns,
                                                         fairness_constraints_provenance_greater_than,
                                                         fairness_constraints_provenance_smaller_than,
                                                         minimal_added_relaxations):
                print("{} is tight, no need to check subsets".format(value_assignment))
                if not set_stop_line:
                    set_stop_line = True
                    only_first_column_to_sort, stop_line, minimal_added_relaxations = \
                        set_stop_line_and_resort_relax_only(value_assignment, index_of_columns_remained,
                                                            [t] + terms_above,
                                                            stop_line, sorted_table,
                                                            delta_table_dropped_duplicates, columns_delta_table,
                                                            only_first_column_to_sort,
                                                            row_num, columns_resort,
                                                            minimal_added_relaxations,
                                                            original_columns_delta_table,
                                                            checked_satisfying_constraints,
                                                            checked_unsatisfying_constraints,
                                                            checked_assignments_satisfying,
                                                            checked_assignments_unsatisfying,
                                                            numeric_attributes,
                                                            categorical_att_columns, categorical_attributes,
                                                            selection_numeric, selection_categorical,
                                                            fairness_constraints_provenance_greater_than,
                                                            fairness_constraints_provenance_smaller_than
                                                            )
                continue
            combo_list = [[y] for y in terms_above]
            while len(combo_list) > 0:
                combo: List[Any] = combo_list.pop(0)
                combo_str = intbitset(combo).strbits()
                if combo_str in checked_satisfying_constraints:
                    continue
                combo_w_t = [t] + combo
                combo_str = intbitset(combo_w_t).strbits()
                # combo_str += '0' * (num_columns - len(combo) - 1)
                if combo_str in checked_satisfying_constraints:
                    continue
                if combo_str in checked_unsatisfying_constraints:
                    # generate children
                    idx = terms_above.index(combo[-1])
                    for x in terms_above[idx + 1:]:
                        combo_list.append(combo + [x])
                    continue
                value_assignment = get_relaxation(combo_w_t, delta_table_dropped_duplicates)
                print("check term set {}, value assignment {}".format(combo_w_t, value_assignment))
                if value_assignment in checked_assignments_satisfying:
                    checked_satisfying_constraints.add(combo_str)
                    continue
                elif value_assignment in checked_assignments_unsatisfying:
                    checked_unsatisfying_constraints.add(combo_str)
                    idx = terms_above.index(combo[-1])
                    for x in terms_above[idx + 1:]:
                        combo_list.append(combo + [x])
                    continue
                else:
                    # check larger term set if this one doesn't satisfy the fairness constraints
                    # if it does, whether minimal or not, don't check larger ones
                    if assign_to_provenance_relax_only(value_assignment, numeric_attributes, categorical_attributes,
                                                       selection_numeric, selection_categorical,
                                                       original_columns_delta_table, num_columns,
                                                       fairness_constraints_provenance_greater_than,
                                                       fairness_constraints_provenance_smaller_than):

                        checked_satisfying_constraints.add(combo_str)
                        checked_assignments_satisfying.append(value_assignment)

                        this_is_minimal, minimal_added_relaxations = \
                            update_minimal_relaxation(minimal_added_relaxations, value_assignment)

                        if not set_stop_line:
                            set_stop_line = True
                            only_first_column_to_sort, stop_line, minimal_added_relaxations = \
                                set_stop_line_and_resort_relax_only(value_assignment, index_of_columns_remained,
                                                                    combo_w_t,
                                                                    stop_line, sorted_table,
                                                                    delta_table_dropped_duplicates, columns_delta_table,
                                                                    only_first_column_to_sort,
                                                                    row_num, columns_resort,
                                                                    minimal_added_relaxations,
                                                                    original_columns_delta_table,
                                                                    checked_satisfying_constraints,
                                                                    checked_unsatisfying_constraints,
                                                                    checked_assignments_satisfying,
                                                                    checked_assignments_unsatisfying,
                                                                    numeric_attributes,
                                                                    categorical_att_columns, categorical_attributes,
                                                                    selection_numeric, selection_categorical,
                                                                    fairness_constraints_provenance_greater_than,
                                                                    fairness_constraints_provenance_smaller_than
                                                                    )

                        break
                    else:  # if doesn't satisfy provenance constraints
                        checked_unsatisfying_constraints.add(combo_str)
                        checked_assignments_unsatisfying.append(value_assignment)
                        # generate children
                        idx = terms_above.index(combo[-1])
                        for x in terms_above[idx + 1:]:
                            combo_list.append(combo + [x])
        row_num += 1

    sorted_table.apply(iterrow, axis=1)
    return minimal_added_relaxations


def whether_value_assignment_is_tight_minimal(must_included_term_delta_values, value_assignment, numeric_attributes,
                                              categorical_attributes,
                                              selection_numeric, selection_categorical,
                                              columns_delta_table, num_columns,
                                              fairness_constraints_provenance_greater_than,
                                              fairness_constraints_provenance_smaller_than,
                                              minimal_added_relaxations):
    value_idx = 0
    for v in value_assignment:
        if v == 0:
            value_idx += 1
            continue
        if v == must_included_term_delta_values[value_idx]:
            value_idx += 1
            continue
        smaller_value_assignment = value_assignment.copy()
        att = columns_delta_table[value_idx]
        if value_idx < len(numeric_attributes):  # numeric
            while True:
                if smaller_value_assignment[value_idx] == 0:
                    break
                if smaller_value_assignment[value_idx] < 0:
                    smaller_value_assignment[value_idx] += selection_numeric[att][2]
                    if smaller_value_assignment[value_idx] > 0:
                        break
                elif smaller_value_assignment[value_idx] > 0:
                    smaller_value_assignment[value_idx] -= selection_numeric[att][2]
                    if smaller_value_assignment[value_idx] < 0:
                        break
                if assign_to_provenance_relax_only(smaller_value_assignment, numeric_attributes, categorical_attributes,
                                                   selection_numeric, selection_categorical, columns_delta_table,
                                                   num_columns,
                                                   fairness_constraints_provenance_greater_than,
                                                   fairness_constraints_provenance_smaller_than):
                    if not dominated_by_minimal_set(minimal_added_relaxations, smaller_value_assignment):
                        return False
                else:
                    break
        else:  # categorical
            smaller_value_assignment[value_idx] = 0
            if assign_to_provenance_relax_only(smaller_value_assignment, numeric_attributes, categorical_attributes,
                                               selection_numeric, selection_categorical, columns_delta_table,
                                               num_columns,
                                               fairness_constraints_provenance_greater_than,
                                               fairness_constraints_provenance_smaller_than):
                if not dominated_by_minimal_set(minimal_added_relaxations, smaller_value_assignment):
                    return False
        value_idx += 1
    return True


def search_bidirectional(sorted_table, delta_table, delta_table_multifunctional, columns_delta_table,
                         numeric_attributes,
                         categorical_att_columns, categorical_attributes, selection_numeric, selection_categorical,
                         fairness_constraints_provenance_greater_than, fairness_constraints_provenance_smaller_than):
    minimal_added_relaxations = []  # relaxation to add to original selection conditions
    checked_invalid_combination = set()
    checked_single_directional_combination = set()
    checked_satisfying_constraints = set()  # set of bit arrays
    checked_unsatisfying_constraints = set()
    num_columns = len(sorted_table.columns)
    stop_line = pd.Series([len(sorted_table)] * num_columns, sorted_table.columns)
    set_stop_line = False
    row_num = 0
    checked_assignments_satisfying = []
    checked_assignments_unsatisfying = []

    def iterrow(row):
        nonlocal row_num
        nonlocal set_stop_line
        nonlocal stop_line
        nonlocal minimal_added_relaxations
        for i, t in row.items():
            if stop_line[i] <= row_num:
                continue
            # print("now I'm at row {}, col {}, term {}".format(row_num, i, t))
            terms_above = list(sorted_table.loc[:row_num - 1, i])
            combo_list = [[y] for y in terms_above]
            while len(combo_list) > 0:
                combo: List[Any] = combo_list.pop(0)
                combo_str = intbitset(combo).strbits()
                if combo_str in checked_invalid_combination or combo_str in checked_satisfying_constraints:
                    continue
                combo_w_t = [t] + combo
                combo_str = intbitset(combo_w_t).strbits()
                # combo_str += '0' * (num_columns - len(combo) - 1)
                if combo_str in checked_invalid_combination or combo_str in checked_satisfying_constraints \
                        or combo_str in checked_single_directional_combination:
                    continue
                if combo_str in checked_unsatisfying_constraints:
                    # generate children
                    idx = terms_above.index(combo[-1])
                    for x in terms_above[idx + 1:]:
                        combo_list.append(combo + [x])
                    continue

                # optimization 1: if all terms in the set are relaxation/contraction terms, skip
                terms_set = delta_table.loc[combo_w_t]
                if terms_set['relaxation_term'].eq(True).all() or terms_set['relaxation_term'].eq(False).all():
                    checked_single_directional_combination.add(combo_str)
                    continue
                have_legitimate_value_assignment, value_assignments = get_refinement(combo_w_t,
                                                                                     delta_table,
                                                                                     delta_table_multifunctional)
                if have_legitimate_value_assignment:
                    assign_successfully = False
                    for value_assignment in value_assignments:
                        if value_assignment in checked_assignments_satisfying:
                            checked_satisfying_constraints.add(combo_str)
                            continue
                        elif value_assignment in checked_assignments_unsatisfying:
                            idx = terms_above.index(combo[-1])
                            for x in terms_above[idx + 1:]:
                                combo_list.append(combo + [x])
                            continue
                        else:
                            # check larger term set if this one doesn't satisfy the fairness constraints
                            # if it does, whether minimal or not, don't check larger ones
                            if assign_to_provenance(value_assignment, numeric_attributes, categorical_attributes,
                                                    selection_numeric, selection_categorical,
                                                    columns_delta_table, num_columns,
                                                    fairness_constraints_provenance_greater_than,
                                                    fairness_constraints_provenance_smaller_than
                                                    ):
                                assign_successfully = True
                                # value_assignment = [x if isinstance(x, int) else round(x, 2) for x in value_assignment]
                                this_is_minimal, minimal_added_relaxations = \
                                    update_minimal_relaxation(minimal_added_relaxations, value_assignment)
                                checked_assignments_satisfying.append(value_assignment)

                                if this_is_minimal:
                                    print("terms: {}, value_assignment: {}".format(combo_w_t, value_assignment))
                                    if not set_stop_line:
                                        stop_line = update_stop_line_bidirectional(combo_w_t, stop_line,
                                                                                   minimal_added_relaxations,
                                                                                   sorted_table, delta_table,
                                                                                   delta_table_multifunctional,
                                                                                   columns_delta_table,
                                                                                   value_assignment,
                                                                                   numeric_attributes,
                                                                                   categorical_attributes,
                                                                                   categorical_att_columns)
                                        set_stop_line = True
                                        print("stop line {}".format(stop_line))
                            else:
                                checked_assignments_unsatisfying.append(value_assignment)
                    if assign_successfully:
                        checked_satisfying_constraints.add(combo_str)
                    else:
                        checked_unsatisfying_constraints.add(combo_str)
                        # generate children
                        idx = terms_above.index(combo[-1])
                        for x in terms_above[idx + 1:]:
                            combo_list.append(combo + [x])
                else:
                    checked_invalid_combination.add(combo_str)
        row_num += 1

    # sorted_table[:1].apply(iterrow, axis=1)
    # max_stop_line = stop_line.max()
    sorted_table.apply(iterrow, axis=1)
    return minimal_added_relaxations


def transform_to_refinement_format(minimal_added_refinements, numeric_attributes, selection_numeric_attributes,
                                   selection_categorical_attributes, columns_delta_table):
    minimal_refinements = []
    num_numeric_att = len(numeric_attributes)
    num_att = len(columns_delta_table)
    for ar in minimal_added_refinements:
        select_numeric = copy.deepcopy(selection_numeric_attributes)
        select_categorical = copy.deepcopy(selection_categorical_attributes)
        for att_idx in range(num_att):
            if att_idx < num_numeric_att:
                select_numeric[numeric_attributes[att_idx]][1] -= ar[att_idx]
            elif ar[att_idx] == 1:
                at, va = columns_delta_table[att_idx].rsplit('_', 1)
                select_categorical[at].append(va)
            elif ar[att_idx] == -1:
                at, va = columns_delta_table[att_idx].rsplit('_', 1)
                select_categorical[at].remove(va)
        minimal_refinements.append({'numeric': select_numeric, 'categorical': select_categorical})
    return minimal_refinements


########################################################################################################################


def FindMinimalRefinement(data_file, query_file, constraint_file):
    data = pd.read_csv(data_file)
    with open(query_file) as f:
        query_info = json.load(f)

    selection_numeric_attributes = query_info['selection_numeric_attributes']
    selection_categorical_attributes = query_info['selection_categorical_attributes']
    numeric_attributes = list(selection_numeric_attributes.keys())
    categorical_attributes = query_info['categorical_attributes']
    selected_attributes = numeric_attributes + [x for x in categorical_attributes]
    print("selected_attributes", selected_attributes)

    with open(constraint_file) as f:
        constraint_info = json.load(f)

    sensitive_attributes = constraint_info['all_sensitive_attributes']
    fairness_constraints = constraint_info['fairness_constraints']

    pd.set_option('display.float_format', '{:.2f}'.format)

    time1 = time.time()
    fairness_constraints_provenance_greater_than, fairness_constraints_provenance_smaller_than, \
    data_rows_greater_than, data_rows_smaller_than, only_greater_than, only_smaller_than \
        = subtract_provenance(data, selected_attributes, sensitive_attributes, fairness_constraints,
                              numeric_attributes, categorical_attributes, selection_numeric_attributes,
                              selection_categorical_attributes)
    time_provenance2 = time.time()
    provenance_time = time_provenance2 - time1
    # print("provenance_expressions")
    # print(*fairness_constraints_provenance_greater_than, sep="\n")
    # print(*fairness_constraints_provenance_smaller_than, sep="\n")

    if only_greater_than:
        time_table1 = time.time()
        delta_table, columns_delta_table, \
        categorical_att_columns = build_sorted_table_relax_only(data, selected_attributes,
                                                                numeric_attributes,
                                                                categorical_attributes,
                                                                selection_numeric_attributes,
                                                                selection_categorical_attributes,
                                                                sensitive_attributes,
                                                                fairness_constraints,
                                                                fairness_constraints_provenance_greater_than,
                                                                fairness_constraints_provenance_smaller_than,
                                                                data_rows_greater_than,
                                                                data_rows_smaller_than)
        time_table2 = time.time()
        table_time = time_table2 - time_table1
        # print("delta table:\n{}".format(delta_table))
        time_search1 = time.time()

        checked_satisfying_constraints = set()  # set of bit arrays
        checked_unsatisfying_constraints = set()
        checked_assignments_satisfying = []
        checked_assignments_unsatisfying = []

        minimal_added_refinements = resort_and_search_relax_only(delta_table.index.values, delta_table,
                                                                 columns_delta_table, range(len(columns_delta_table)),
                                                                 [], columns_delta_table,
                                                                 checked_satisfying_constraints,
                                                                 checked_unsatisfying_constraints,
                                                                 checked_assignments_satisfying,
                                                                 checked_assignments_unsatisfying, numeric_attributes,
                                                                 categorical_att_columns, categorical_attributes,
                                                                 selection_numeric_attributes,
                                                                 selection_categorical_attributes,
                                                                 fairness_constraints_provenance_greater_than,
                                                                 fairness_constraints_provenance_smaller_than)
        time_search2 = time.time()
        minimal_refinements = transform_to_refinement_format(minimal_added_refinements, numeric_attributes,
                                                             selection_numeric_attributes,
                                                             selection_categorical_attributes,
                                                             columns_delta_table)
        time2 = time.time()
        print("provenance time = {}".format(provenance_time))
        print("table time = {}".format(table_time))
        print("searching time = {}".format(time_search2 - time_search1))
        print("minimal_added_relaxations:{}".format(minimal_added_refinements))
        return minimal_refinements, minimal_added_refinements, time2 - time1

    delta_table, delta_table_multifunctional, columns_delta_table, \
    categorical_att_columns = build_sorted_table_bidirectional(data, selected_attributes,
                                                               numeric_attributes,
                                                               categorical_attributes,
                                                               selection_numeric_attributes,
                                                               selection_categorical_attributes,
                                                               sensitive_attributes,
                                                               fairness_constraints,
                                                               fairness_constraints_provenance_greater_than,
                                                               fairness_constraints_provenance_smaller_than,
                                                               data_rows_greater_than,
                                                               data_rows_smaller_than)

    checked_satisfying_constraints = set()  # set of bit arrays
    checked_unsatisfying_constraints = set()
    checked_single_directional_combination = set()
    checked_assignments_satisfying = []
    checked_assignments_unsatisfying = []

    minimal_added_refinements = resort_and_search_bidirectional(delta_table.index.values, delta_table,
                                                                delta_table_multifunctional,
                                                                columns_delta_table, range(len(columns_delta_table)),
                                                                [], columns_delta_table,
                                                                checked_satisfying_constraints,
                                                                checked_unsatisfying_constraints,
                                                                checked_assignments_satisfying,
                                                                checked_assignments_unsatisfying,
                                                                checked_single_directional_combination,
                                                                numeric_attributes,
                                                                categorical_att_columns, categorical_attributes,
                                                                selection_numeric_attributes,
                                                                selection_categorical_attributes,
                                                                fairness_constraints_provenance_greater_than,
                                                                fairness_constraints_provenance_smaller_than)

    delta_table.round(2)
    delta_table_multifunctional.round(2)
    print("delta table:\n{}".format(delta_table))
    print("delta_table_multifunctional:\n{}".format(delta_table_multifunctional))
    # print("sorted table:\n{}".format(sorted_table))

    time_search1 = time.time()
    # minimal_added_refinements = search_bidirectional(sorted_table, delta_table, delta_table_multifunctional,
    #                                                  columns_delta_table,
    #                                                  numeric_attributes, categorical_att_columns,
    #                                                  categorical_attributes, selection_numeric_attributes,
    #                                                  selection_categorical_attributes,
    #                                                  fairness_constraints_provenance_greater_than,
    #                                                  fairness_constraints_provenance_smaller_than)
    time_search2 = time.time()
    print("searching time = {}".format(time_search2 - time_search1))
    print("minimal_added_relaxations:{}".format(minimal_added_refinements))

    minimal_refinements = transform_to_refinement_format(minimal_added_refinements, numeric_attributes,
                                                         selection_numeric_attributes, selection_categorical_attributes,
                                                         columns_delta_table)
    time2 = time.time()

    return minimal_refinements, minimal_added_refinements, time2 - time1


# data_file = r"../InputData/Pipelines/healthcare/incomeK/before_selection_incomeK.csv"
# query_file = r"../InputData/Pipelines/healthcare/incomeK/relaxation/query4.json"
# constraint_file = r"../InputData/Pipelines/healthcare/incomeK/relaxation/constraint1.json"

data_file = r"toy_examples/example2.csv"
query_file = r"toy_examples/query2.json"
constraint_file = r"toy_examples/constraint3.json"


print("\nnaive algorithm:\n")

minimal_refinements2, minimal_added_refinements2, running_time2 = lt.FindMinimalRefinement(data_file, query_file,
                                                                                           constraint_file)

print(*minimal_refinements2, sep="\n")
print("running time = {}".format(running_time2))


print("\nour algorithm:\n")

minimal_refinements, minimal_added_refinements, running_time = FindMinimalRefinement(data_file, query_file,
                                                                                     constraint_file)

print(*minimal_refinements, sep="\n")
print("running time = {}".format(running_time))
