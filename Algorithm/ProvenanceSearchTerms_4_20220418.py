"""
executable but buggy since stop line should be below all terms with the same value in this column
based on ProvenanceSearch_3
add optimization 1: mark each term positive or negative.
If we have both directions, we need to build a refinement based on both positive and negative terms.
Otherwise it should be eliminated earlier.
"""

import copy
from typing import List, Any
import numpy as np
import pandas as pd
import time
from intbitset import intbitset
import json


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
    elif only_smaller_than:
        data['satisfy'] = data.apply(test_satisfying_rows, axis=1)


    def get_provenance(row, fc_dic, fc, protected_greater_than):
        sensitive_att_of_fc = list(fc["sensitive_attributes"].keys())
        sensitive_values_of_fc = {k: fc["sensitive_attributes"][k] for k in sensitive_att_of_fc}
        fairness_value_of_row = row[sensitive_att_of_fc]
        if sensitive_values_of_fc == fairness_value_of_row.to_dict():
            if only_greater_than and row['satisfy'] == 1:
                fc_dic['number'] -= 1
            elif not (only_smaller_than and row['satisfy'] == 0):
                terms = row[selected_attributes]
                fc_dic['provenance_expression'].append(terms.to_dict())
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
        data = data[selected_attributes + sensitive_attributes +
                    ['protected_greater_than', 'protected_smaller_than', 'satisfy']].apply(get_provenance,
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
    if only_greater_than:
        data = data[data['satisfy'] == 0]
    elif only_smaller_than:
        data = data[data['satisfy'] == 1]
    # only get rows that are envolved w.r.t. fairness constraint
    data_rows_greater_than = data[data['protected_greater_than'] == 1]
    data_rows_smaller_than = data[data['protected_smaller_than'] == 1]

    return fairness_constraints_provenance_greater_than, fairness_constraints_provenance_smaller_than, \
           data_rows_greater_than, data_rows_smaller_than



# # since when we change selection conditions, it may not satisfy
# # for such a term, we can only remove the corresponding row in delta table (and sorted table)
# def simplify(row, provenance_expressions, sensitive_attributes):
#     """
#     simplify each condition in provenance_expressions about row, since row already satisfies it
#     :param row: a row in sorted table but with all values zero
#     :param provenance_expressions: got from subtract_provenance()
#     """
#     for fc in provenance_expressions:
#         if fc['symbol'] == ">" or fc['symbol'] == ">=":
#             fc['number'] -= 1
#         else:
#             raise Exception("provenance_expressions should not have < or <=")
#         provenance_expression = fc['provenance_expression']
#         to_remove = pd.Series()
#         for pe in provenance_expression:
#             pe_ = pe.drop(sensitive_attributes, axis=1)
#             if pe_.equals(row):
#                 to_remove = pe
#                 break
#         provenance_expression.remove(to_remove)


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


def row_is_protected_group(row, fairness_constraints_provenance):
    for fc in fairness_constraints_provenance:
        provenance_expression = fc['provenance_expression']


def build_sorted_table(data, selected_attributes, numeric_attributes,
                       categorical_attributes, selection_numeric, selection_categorical,
                       sensitive_attributes, fairness_constraints,
                       fairness_constraints_provenance_greater_than, fairness_constraints_provenance_smaller_than,
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

    only_greater_than = False
    only_smaller_than = False
    if len(fairness_constraints_provenance_greater_than) != 0 and len(
            fairness_constraints_provenance_smaller_than) == 0:
        only_greater_than = True
    elif len(fairness_constraints_provenance_greater_than) == 0 and len(
            fairness_constraints_provenance_smaller_than) != 0:
        only_smaller_than = True

    columns_delta_table = numeric_attributes.copy()
    for att, domain in categorical_attributes.items():
        for value in domain:
            if only_greater_than and value in selection_categorical[att]:
                continue
            elif only_smaller_than and value not in selection_categorical[att]:
                continue
            else:
                columns_delta_table.append(att + "_" + value)


    # if there is only one direction in constraints, we remvoe some rows that already satisfy/dissatisfy the constraints,
    # and then need to change the size in constraint inequality
    change_constraint = 0
    list_of_rows_delta_table = []
    row_idx = 0

    # build delta table
    def iterrow(row, greater_than=True):  # greater than is symbol in fairness constraint is greater than (relaxation term)
        delta_values = []
        nonlocal change_constraint
        for att in numeric_attributes:
            if greater_than:
                if selection_numeric[att][0] == ">":
                    if row[att] > selection_numeric[att][1]:
                        delta_v = 0
                    else:
                        if isinstance(row[att], int):
                            delta_v = selection_numeric[att][1] - row[att] + 1
                        else:
                            delta_v = selection_numeric[att][1] - row[att] + 0.05
                elif selection_numeric[att][0] == ">=":
                    if row[att] >= selection_numeric[att][1]:
                        delta_v = 0
                    else:
                        delta_v = selection_numeric[att][1] - row[att]
                elif selection_numeric[att][0] == "<":
                    if row[att] < selection_numeric[att][1]:
                        delta_v = 0
                    else:
                        if isinstance(row[att], int):
                            delta_v = row[att] - selection_numeric[att][1] + 1
                        else:
                            delta_v = row[att] - selection_numeric[att][1] + 0.05
                else:
                    if row[att] <= selection_numeric[att][1]:
                        delta_v = 0
                    else:
                        delta_v = row[att] - selection_numeric[att][1]
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
                        if isinstance(row[att], int):
                            delta_v = - (row[att] - selection_numeric[att][1] + 1)
                        else:
                            delta_v = - (row[att] - selection_numeric[att][1] + 0.05)
                elif selection_numeric[att][0] == "<":
                    if row[att] >= selection_numeric[att][1]:
                        delta_v = 0
                    else:
                        delta_v = - (selection_numeric[att][1] - row[att])
                else:  # selection_numeric[att][0] == "<=":
                    if row[att] > selection_numeric[att][1]:
                        delta_v = 0
                    else:
                        if isinstance(row[att], int):
                            delta_v = - (selection_numeric[att][1] - row[att] + 1)
                        else:
                            delta_v = - (selection_numeric[att][1] - row[att] + 0.05)
            delta_values.append(delta_v)
        for att, domain in categorical_attributes.items():
            for value in domain:
                if only_greater_than and value in selection_categorical[att]:
                    continue
                elif only_smaller_than and value not in selection_categorical[att]:
                    continue
                if greater_than:
                    if row[att] in selection_categorical[att]:
                        delta_values.append(0)
                    elif row[att] == value:
                        delta_values.append(1)
                    else:  # make it 100 because it should be ranked the last
                        delta_values.append(100)
                else:
                    if row[att] not in selection_categorical[att]:
                        delta_values.append(0)
                    elif row[att] == value:
                        delta_values.append(-1)
                    else:
                        delta_values.append(-100)
        # FIXME: should I remove all-zero rows here?
        # remove all-zeros or non-all-zeros columns when constraints are single-direction
        delta_values.append(greater_than)
        nonlocal row_idx
        if only_greater_than:
            if not all(v == 0 for v in delta_values):
                list_of_rows_delta_table.append(delta_values)
            else:
                change_constraint -= 1

        elif only_smaller_than:
            if all(v == 0 for v in delta_values):
                list_of_rows_delta_table.append(delta_values)
            else:
                change_constraint += 1
        else:
            list_of_rows_delta_table.append(delta_values)
        row_idx += 1

    # print("data_rows_greater_than:\n", data_rows_greater_than)
    data_rows_greater_than.apply(iterrow, args=(True,), axis=1)
    print("delta_table with data_rows_greater_than:")
    print(pd.DataFrame(list_of_rows_delta_table, columns=columns_delta_table+['relaxation_term']))
    data_rows_smaller_than.apply(iterrow, args=(False,), axis=1)
    delta_table = pd.DataFrame(list_of_rows_delta_table, columns=columns_delta_table+['relaxation_term'])
    print("delta_table:\n", delta_table)

    sorted_table_by_column = dict()
    for att in columns_delta_table:
        s = np.array(list(zip(delta_table[att].abs(),
                              delta_table[columns_delta_table[0]].abs())),
                     dtype=[('value', delta_table.dtypes[att]),
                            ('tiebreaker', delta_table.dtypes[columns_delta_table[0]])])
        sorted_att_idx = np.argsort(s, order=['value', 'tiebreaker'])
        sorted_table_by_column[att] = sorted_att_idx

    sorted_table = pd.DataFrame(data=sorted_table_by_column)
    # print("sorted_table:\n", sorted_table)
    categorical_att_columns = [item for item in columns_delta_table if item not in numeric_attributes]
    # FIXME: should it be 0.5 or 0?
    # make it 100 because it should be ranked the last
    # but for value assignment, 100 means it doesn't change the selection conditions
    delta_table[categorical_att_columns] = delta_table[categorical_att_columns].replace(100, 0)
    delta_table[categorical_att_columns] = delta_table[categorical_att_columns].replace(-100, 0)
    # print("delta_table:\n", delta_table)
    return sorted_table, delta_table, columns_delta_table, only_greater_than, only_smaller_than, change_constraint


def assign_to_provenance(value_assignment, numeric_attributes, categorical_attributes, selection_numeric,
                         selection_categorical, columns_delta_table, num_columns,
                         fairness_constraints_provenance_greater_than,
                         fairness_constraints_provenance_smaller_than, change_constraint):
    va_dict = dict(zip(columns_delta_table, value_assignment))
    # va_dict = value_assignment.to_dict()
    # greater than
    for fc in fairness_constraints_provenance_greater_than:
        sum = 0
        satisfy_this_fairness_constraint = False
        for pe in fc['provenance_expression']:
            fail = False
            for att in pe:
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
                sum += 1
                if eval(str(sum) + fc['symbol'] + str(fc['number'] + change_constraint)):
                    satisfy_this_fairness_constraint = True
                    break
        if not satisfy_this_fairness_constraint:
            return False

    for fc in fairness_constraints_provenance_smaller_than:
        sum = 0
        satisfy_this_fairness_constraint = True
        for pe in fc['provenance_expression']:
            fail = False
            for att in pe:
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
                    # TODO: fix the following
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
                sum += 1
                if not eval(str(sum) + fc['symbol'] + str(fc['number'])):
                    satisfy_this_fairness_constraint = False
                    break
        if not satisfy_this_fairness_constraint:
            return False
    return True


def get_relaxation(terms, delta_table, only_greater_than, only_smaller_than):
    """
To get skyline of all terms in list terms
    :param terms: list of indices of terms. [1,3,5]
    :param delta_table: delta table returned by func build_sorted_table()
    :return: return Ture or false whether terms can have a legitimate value assignment .
    """
    if only_greater_than:
        value_assignments = [delta_table.iloc[terms].max().tolist()]
        return True, value_assignments
    column_names = delta_table.columns.tolist()
    num_col = len(column_names)

    value_assignments = [[]]
    mark_satisfied_terms = [0]
    rows_to_compare = delta_table.iloc[terms]
    rows_to_compare_indices = rows_to_compare.index.tolist()

    all_included = 0
    for i in rows_to_compare_indices:
        all_included = (1 << i) | all_included
    for i in range(num_col + 1):
        if i == num_col:
            num_result = len(value_assignments)
            j = 0
            while j < num_result:
                if mark_satisfied_terms[j] != all_included:
                    del value_assignments[j]
                    del mark_satisfied_terms[j]
                    num_result -= 1
                else:
                    j += 1
            return len(value_assignments) != 0, value_assignments
        col = column_names[i]
        max_in_col = rows_to_compare[col].max()
        if max_in_col > 0:
            for va in value_assignments:
                va.append(max_in_col)
            positive_terms_int = 0
            indices = rows_to_compare[rows_to_compare[col] > 0].index.tolist()
            for d in indices:
                positive_terms_int = (1 << d) | positive_terms_int
            for k in range(len(mark_satisfied_terms)):
                mark_satisfied_terms[k] |= positive_terms_int
            continue
        non_zeros = rows_to_compare[rows_to_compare[col] < 0].index.tolist()
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
                    covered = [x for x in non_zeros if rows_to_compare.loc[x][col] >= rows_to_compare.loc[n][col]]
                    to_append = mark_satisfied_terms[k]
                    for c in covered:
                        to_append |= 1 << c
                    mark_satisfied_terms_new.append(to_append)
                    v_ = value_assignments[k].copy()
                    v_.append(rows_to_compare.loc[n][col])
                    value_assignments_new.append(v_)
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


# TODO: now: only update stop line for the first time
def update_stop_line(combo_w_t, stop_line, minimal_added_relaxations, sorted_table):
    def itercol(column):
        total_num = len(combo_w_t)
        num = 0
        col_idx = 0
        for k, v in column.items():
            if v in combo_w_t:
                num += 1
                col_idx = k
                if num == total_num:
                    break
        return col_idx

    new_stop_line = sorted_table.apply(itercol, axis=0)
    return new_stop_line


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


def search(sorted_table, delta_table, columns_delta_table, numeric_attributes,
           categorical_attributes, selection_numeric, selection_categorical,
           fairness_constraints_provenance_greater_than, fairness_constraints_provenance_smaller_than,
           only_greater_than, only_smaller_than, change_constraint):
    minimal_added_relaxations = []  # relaxation to add to original selection conditions
    checked = []  # set of bit arrays
    num_columns = len(sorted_table.columns)
    stop_line = pd.Series([len(sorted_table)] * num_columns, sorted_table.columns)
    set_stop_line = False
    row_num = 0

    def iterrow(row):
        nonlocal row_num
        nonlocal set_stop_line
        nonlocal stop_line
        nonlocal minimal_added_relaxations
        for i, t in row.items():
            if stop_line[i] <= row_num:
                continue
            if only_smaller_than or only_greater_than:
                t_str = '0' * t + '1' + '0' * (num_columns - 1 - t)
                if t_str not in checked:
                    checked.append(t_str)
                    have_legitimate_value_assignment, value_assignments = get_relaxation([t],
                                                                                         delta_table,
                                                                                         only_greater_than,
                                                                                         only_smaller_than)
                    if have_legitimate_value_assignment:
                        for value_assignment in value_assignments:
                            if assign_to_provenance(value_assignment, numeric_attributes, categorical_attributes,
                                                    selection_numeric, selection_categorical, columns_delta_table,
                                                    num_columns, fairness_constraints_provenance_greater_than,
                                                    fairness_constraints_provenance_smaller_than, change_constraint):
                                this_is_minimal, minimal_added_relaxations = \
                                    update_minimal_relaxation(minimal_added_relaxations, value_assignment)
                                if this_is_minimal:
                                    # print("value_assignment: {}".format(value_assignment))
                                    if not set_stop_line:
                                        stop_line = update_stop_line([t], stop_line, minimal_added_relaxations,
                                                                     sorted_table)
                                        set_stop_line = True
                                checked.append(t_str)

            terms_above = list(sorted_table.loc[:row_num - 1, i])
            combo_list = [[y] for y in terms_above]
            while len(combo_list) > 0:
                combo: List[Any] = combo_list.pop(0)
                combo_w_t = [t] + combo
                combo_str = intbitset(combo_w_t).strbits()
                # combo_str += '0' * (num_columns - len(combo) - 1)
                if combo_str in checked:
                    continue
                if combo_w_t == [1, 10]:
                    print("combo_w_t: {}".format(combo_w_t))
                # optimization 1: if all terms in the set are relaxation/contraction terms, skip
                terms_set = delta_table.iloc[combo_w_t]
                if terms_set['relaxation_term'].eq(True).all() or terms_set['relaxation_term'].eq(False).all():
                    continue
                have_legitimate_value_assignment, value_assignments = get_relaxation(combo_w_t,
                                                                                     delta_table,
                                                                                     only_greater_than,
                                                                                     only_smaller_than)
                if have_legitimate_value_assignment:
                    # print("value_assignments: {}".format(value_assignments))
                    for value_assignment in value_assignments:
                        # check larger term set if this one doesn't satisfy the fairness constraints
                        # if it does, whether minimal or not, don't check larger ones
                        if assign_to_provenance(value_assignment, numeric_attributes, categorical_attributes,
                                                selection_numeric, selection_categorical,
                                                columns_delta_table, num_columns,
                                                fairness_constraints_provenance_greater_than,
                                                fairness_constraints_provenance_smaller_than,
                                                change_constraint):
                            this_is_minimal, minimal_added_relaxations = \
                                update_minimal_relaxation(minimal_added_relaxations, value_assignment)
                            if this_is_minimal:
                                print("terms: {}, value_assignment: {}".format(combo_w_t, value_assignment))
                                if not set_stop_line:
                                    stop_line = update_stop_line(combo_w_t, stop_line, minimal_added_relaxations,
                                                                 sorted_table)
                                    set_stop_line = True
                                    print("stop line {}".format(stop_line))
                        else:
                            # generate children
                            idx = terms_above.index(combo[-1])
                            for x in terms_above[idx + 1:]:
                                combo_list.append(combo + [x])
                checked.append(combo_str)
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
                at, va = columns_delta_table[att_idx].split('_')
                select_categorical[at].append(va)
            elif ar[att_idx] == -1:
                at, va = columns_delta_table[att_idx].split('_')
                select_categorical[at].remove(va)
        minimal_refinements.append({'numeric': select_numeric, 'categorical': select_categorical})
    return minimal_refinements



########################################################################################################################



def FindMinimalRefinement(data_file, selection_file):
    data = pd.read_csv(data_file)
    print(data)
    with open(selection_file) as f:
        info = json.load(f)


    all_attributes = data.columns.tolist()

    sensitive_attributes = info['all_sensitive_attributes']
    fairness_constraints = info['fairness_constraints']
    selection_numeric_attributes = info['selection_numeric_attributes']
    selection_categorical_attributes = info['selection_categorical_attributes']
    numeric_attributes = list(selection_numeric_attributes.keys())
    categorical_attributes = info['categorical_attributes']
    selected_attributes = numeric_attributes + [x for x in categorical_attributes]
    print("selected_attributes", selected_attributes)

    pd.set_option('display.float_format', '{:.2f}'.format)

    time1 = time.time()
    fairness_constraints_provenance_greater_than, fairness_constraints_provenance_smaller_than, \
    data_rows_greater_than, data_rows_smaller_than \
        = subtract_provenance(data, selected_attributes, sensitive_attributes, fairness_constraints,
                              numeric_attributes, categorical_attributes, selection_numeric_attributes,
                              selection_categorical_attributes)
    print("provenance_expressions")
    print(*fairness_constraints_provenance_greater_than, sep="\n")
    print(*fairness_constraints_provenance_smaller_than, sep="\n")
    #
    # print("data_rows_greater_than: \n{}".format(data_rows_greater_than))

    sorted_table, delta_table, columns_delta_table, \
    only_greater_than, only_smaller_than, change_constraint = build_sorted_table(data, selected_attributes,
                                                                                 numeric_attributes,
                                                                                 categorical_attributes,
                                                                                 selection_numeric_attributes,
                                                                                 selection_categorical_attributes,
                                                                                 sensitive_attributes, fairness_constraints,
                                                                                 fairness_constraints_provenance_greater_than,
                                                                                 fairness_constraints_provenance_smaller_than,
                                                                                 data_rows_greater_than,
                                                                                 data_rows_smaller_than
                                                                                 )
    print("delta table:\n{}".format(delta_table))
    print("sorted table:\n{}".format(sorted_table))

    minimal_added_refinements = search(sorted_table, delta_table, columns_delta_table, numeric_attributes,
                                       categorical_attributes, selection_numeric_attributes,
                                       selection_categorical_attributes,
                                       fairness_constraints_provenance_greater_than,
                                       fairness_constraints_provenance_smaller_than, only_greater_than, only_smaller_than,
                                       change_constraint)

    print("minimal_added_relaxations:{}".format(minimal_added_refinements))

    minimal_refinements = transform_to_refinement_format(minimal_added_refinements, numeric_attributes,
                                                         selection_numeric_attributes, selection_categorical_attributes,
                                                         columns_delta_table)
    time2 = time.time()

    print(*minimal_refinements, sep="\n")
    print("running time = {}".format(time2 - time1))
    return minimal_refinements, time2 - time1


data_file = r"toy_examples/example2.csv"
selection_file = r"toy_examples/selection2.json"
minimal_refinements, running_time = FindMinimalRefinement(data_file, selection_file)


