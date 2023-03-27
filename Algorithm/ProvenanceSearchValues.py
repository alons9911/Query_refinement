"""
For refinement:
add support for complex constraints like women >= 50% men

"""

import copy
from typing import List, Any
import numpy as np
import pandas as pd
import time
from intbitset import intbitset
import json
from Algorithm import LatticeTraversal as lt

assign_to_provenance_num = 0


def num2string(pattern):
    st = ''
    for i in pattern:
        if i != -1:
            st += str(i)
        st += '|'
    st = st[:-1]
    return st


def subtract_provenance_relaxation_contraction(data, selected_attributes, sensitive_attributes, fairness_constraints,
                                               numeric_attributes, categorical_attributes, selection_numeric_attributes,
                                               selection_categorical_attributes, only_greater_than, only_smaller_than):
    """
Get provenance expressions
    :param all_sensitive_attributes: list of all att involved in fairness constraints
    :param fairness_constraints: [{'Gender': 'F', 'symbol': '>=', 'number': 3}]
    :param data: dataframe
    :param selected_attributes: attributes in selection conditions
    :return: a list of dictionaries
    """
    time0 = time.time()
    fairness_constraints_provenance_greater_than = []
    fairness_constraints_provenance_smaller_than = []
    data['protected_greater_than'] = 0
    data['protected_smaller_than'] = 0
    data['satisfy'] = 0

    # threshold for contraction
    contraction_threshold = {att: data[att].max() if selection_numeric_attributes[att][0] == '>=' else data[att].min()
                             for att in selection_numeric_attributes}
    print("prepare time = {}".format(time.time() - time0))

    if only_greater_than:
        data['satisfy'] = 0
        all_relevant_attributes = sensitive_attributes + selected_attributes + ['protected_greater_than',
                                                                                'protected_smaller_than', 'satisfy']
        data = data[all_relevant_attributes]
        data = data.groupby(all_relevant_attributes, dropna=False, sort=False).size().reset_index(name='occurrence')
        for fc in fairness_constraints:
            fc_dic = dict()
            fc_dic['symbol'] = fc['symbol']
            fc_dic['number'] = fc['number']
            fc_data = copy.deepcopy(data)
            for k in fc["sensitive_attributes"]:
                fc_data = fc_data[fc_data[k] == fc["sensitive_attributes"][k]]
            fc_dic['provenance_expression'] = fc_data[selected_attributes + ['occurrence']]
            fairness_constraints_provenance_greater_than.append(fc_dic)
        return fairness_constraints_provenance_greater_than, fairness_constraints_provenance_smaller_than, \
            contraction_threshold

    else:  # contraction
        data['satisfy'] = 0
        all_relevant_attributes = sensitive_attributes + selected_attributes + ['protected_greater_than',
                                                                                'protected_smaller_than', 'satisfy']
        data = data[all_relevant_attributes]
        data = data.groupby(all_relevant_attributes, dropna=False, sort=False).size().reset_index(name='occurrence')
        for fc in fairness_constraints:
            fc_dic = dict()
            fc_dic['symbol'] = fc['symbol']
            fc_dic['number'] = fc['number']
            fc_data = copy.deepcopy(data)
            for k in fc["sensitive_attributes"]:
                fc_data = fc_data[fc_data[k] == fc["sensitive_attributes"][k]]
            fc_dic['provenance_expression'] = fc_data[selected_attributes + ['occurrence']]
            fairness_constraints_provenance_smaller_than.append(fc_dic)
        # print("time of get_provenance_contract_only = {}".format(time.time() - time2))
        return fairness_constraints_provenance_greater_than, fairness_constraints_provenance_smaller_than, \
            contraction_threshold


def subtract_provenance_refinement(data, selected_attributes, sensitive_attributes, fairness_constraints,
                                   numeric_attributes, categorical_attributes, selection_numeric_attributes,
                                   selection_categorical_attributes, only_greater_than, only_smaller_than):
    """
Get provenance expressions
    :param all_sensitive_attributes: list of all att involved in fairness constraints
    :param fairness_constraints: [{'Gender': 'F', 'symbol': '>=', 'number': 3}]
    :param data: dataframe
    :param selected_attributes: attributes in selection conditions
    :return: a list of dictionaries
    """
    time0 = time.time()
    fairness_constraints_provenance_greater_than = []
    fairness_constraints_provenance_smaller_than = []
    fairness_constraints_provenance_complex = []
    data['protected_greater_than'] = 0
    data['protected_smaller_than'] = 0
    data['satisfy'] = 0

    # threshold for contraction
    contraction_threshold = {}
    contraction_threshold = {att: data[att].max() if selection_numeric_attributes[att][0] == '>=' else data[att].min()
                             for att in selection_numeric_attributes}
    print("prepare time = {}".format(time.time() - time0))
    time1 = time.time()

    all_relevant_attributes = sensitive_attributes + selected_attributes + ['protected_greater_than',
                                                                            'protected_smaller_than', 'satisfy']
    data = data[all_relevant_attributes]
    data = data.groupby(all_relevant_attributes, dropna=False, sort=False).size().reset_index(name='occurrence')

    for att in selection_categorical_attributes:
        contraction_threshold[att] = set()

    def get_contraction_threshold(df, fc_number, att):
        nonlocal contraction_threshold
        if att in selection_numeric_attributes:
            if selection_numeric_attributes[att][0] == '>' or selection_numeric_attributes[att][0] == '>=':
                df = df.sort_values(by=[att], ascending=False)
                a = df['occurrence'].cumsum().searchsorted(fc_number)
                b = df.columns.get_loc(att)
                c = df.iloc[a, b]
                if a < len(df):
                    if att in contraction_threshold:
                        contraction_threshold[att] = min(c, contraction_threshold[att])
                    else:
                        contraction_threshold[att] = c
                else:
                    if att not in contraction_threshold:
                        contraction_threshold[att] = df[att].max()
            else:
                df = df.sort_values(by=[att], ascending=True)
                a = df['occurrence'].cumsum().searchsorted(fc_number)
                if a < len(df):
                    b = df.columns.get_loc(att)
                    c = df.iloc[a, b]
                    if att in contraction_threshold:
                        contraction_threshold[att] = max(c, contraction_threshold[att])
                    else:
                        contraction_threshold[att] = c
                else:
                    if att not in contraction_threshold:
                        contraction_threshold[att] = df[att].min()
        else:
            to_remove = selection_categorical_attributes[att]
            for value in to_remove:
                if value not in df[att].values.tolist():
                    contraction_threshold[att].add(value)
                else:
                    satisfying = df[df[att] == value]
                    if df['occurrence'].sum() - satisfying[
                        'occurrence'].sum() < fc_number:  # assume fairness constraints has >= but no >
                        contraction_threshold[att].add(value)

    for fc in fairness_constraints:
        fc_dic = dict()
        fc_dic['symbol'] = fc['symbol']
        if "first_sensitive_attributes" in fc:
            if "plus" in fc:
                fc_dic["plus"] = fc["plus"]
            elif "times" in fc:
                fc_dic["times"] = fc["times"]
            fc_data = copy.deepcopy(data)
            for k in fc["first_sensitive_attributes"]:
                fc_data = fc_data[fc_data[k] == fc["first_sensitive_attributes"][k]]
            fc_dic['first_provenance_expression'] = fc_data[selected_attributes + ['occurrence']]
            fc_data = copy.deepcopy(data)
            for k in fc["second_sensitive_attributes"]:
                fc_data = fc_data[fc_data[k] == fc["second_sensitive_attributes"][k]]
            fc_dic['second_provenance_expression'] = fc_data[selected_attributes + ['occurrence']]
            if fc_dic['symbol'] == "<" or fc_dic['symbol'] == "<=":
                fairness_constraints_provenance_complex.append(fc_dic)
        else:
            fc_dic['number'] = fc['number']
            fc_data = copy.deepcopy(data)
            for k in fc["sensitive_attributes"]:
                fc_data = fc_data[fc_data[k] == fc["sensitive_attributes"][k]]
            fc_dic['provenance_expression'] = fc_data[selected_attributes + ['occurrence']]
            if fc_dic['symbol'] == "<" or fc_dic['symbol'] == "<=":
                fairness_constraints_provenance_smaller_than.append(fc_dic)
            else:
                fairness_constraints_provenance_greater_than.append(fc_dic)
                for att in selected_attributes:
                    get_contraction_threshold(fc_data[[att, 'occurrence']], fc_dic['number'], att)

    return fairness_constraints_provenance_greater_than, fairness_constraints_provenance_smaller_than, \
        fairness_constraints_provenance_complex, contraction_threshold


# put categorical columns before numerical ones
def build_PVT_refinement(data, selected_attributes, numeric_attributes,
                         categorical_attributes, selection_numeric, selection_categorical,
                         sensitive_attributes, fairness_constraints,
                         fairness_constraints_provenance_greater_than,
                         fairness_constraints_provenance_smaller_than,
                         fairness_constraints_provenance_complex,
                         contraction_threshold):
    """
    to build the sorted table
    :param fairness_constraints_provenance_greater_than:
    :param fairness_constraints_provenance_smaller_than:
    :param selected_attributes:
    :param data: dataframe
    :param numeric_attributes: list of names of numeric attributes [city, major, state]
    :param categorical_attributes: dictionary: {city: [domain of city], major: [domain of major]}
    :param selection_numeric: dictionary: {grade:[80, >], age:[30, <], hours: [100, <=]}
    :param selection_categorical: dictionary: {city: [accepted cities], major: [accepted majors]}
    :return: return the whole sorted table, including rows that already satisfy the selection conditions;
            also return delta table
    """
    PVT_head = []
    for att in categorical_attributes:
        domain = data[att].dropna().unique().tolist()
        for value in domain:
            if value not in contraction_threshold[att]:
                col = att + "__" + value
                PVT_head.append(col)
    PVT_head = PVT_head + numeric_attributes
    print("PVT_head: {}".format(PVT_head))

    def itercol(col):
        nonlocal possible_values_sets
        att = col.name
        unique_values = col.unique()
        if att in selection_numeric:
            if selection_numeric[att][0] == '>' or selection_numeric[att][0] == '>=':
                unique_values.sort()
                try:
                    idx = next(i for i, v in enumerate(unique_values) if v >= selection_numeric[att][1])
                except StopIteration:
                    idx = len(unique_values) - 1
                # idx = next(i for i, v in enumerate(unique_values) if v >= selection_numeric[att][1])
                possible_values_sets[att].update(unique_values[:idx + 1])
                others = unique_values[idx:]
                try:
                    idx = next(i for i, v in enumerate(others) if v >= contraction_threshold[att])
                except StopIteration:
                    idx = len(others) - 1
                s = [x + selection_numeric[att][2] for x in others[:idx + 1]]
                possible_values_sets[att].update(s)
                # possible_values_sets[att].update([s + selection_numeric[att][2] for s in others[:idx + 1]])
            else:  # selection_numeric[att][0] == '<' or selection_numeric[att][0] == '<=':
                unique_values.sort()
                unique_values = unique_values
                try:
                    idx = next(i for i, v in enumerate(unique_values) if v >= selection_numeric[att][1])
                except StopIteration:
                    idx = len(unique_values) - 1
                # idx = next(i for i, v in enumerate(unique_values) if v <= selection_numeric[att][1])
                possible_values_sets[att].update(unique_values[idx:])
                others = unique_values[1:idx]
                try:
                    idx = next(i for i, v in enumerate(others) if v >= contraction_threshold[att])
                except StopIteration:
                    idx = len(others) - 1
                s = [x - selection_numeric[att][2] for x in others[idx:]]
                possible_values_sets[att].update(s)
                # possible_values_sets[att].update([s - selection_numeric[att][2] for s in others[:idx + 1]])

    data = data.drop_duplicates(
        subset=selected_attributes,
        keep='first').reset_index(drop=True)
    possible_values_sets = {x: set() for x in PVT_head}
    for att in selection_numeric:
        possible_values_sets[att].add(selection_numeric[att][1])

    data[selection_numeric.keys()].apply(itercol, axis=0)

    possible_values_lists = {x: list(possible_values_sets[x]) for x in possible_values_sets}
    for att in PVT_head:
        if att in selection_numeric:
            possible_values_lists[att].sort(key=lambda p: abs(p - selection_numeric[att][1]))
        else:
            pre, v = att.split("__")
            if v not in selection_categorical[pre]:
                possible_values_lists[att] = [0, 1]
            else:
                possible_values_lists[att] = [1, 0]
    possible_value_table = pd.DataFrame({key: pd.Series(value) for key, value in possible_values_lists.items()})
    # print("possible_value_table:\n", possible_value_table)
    possible_value_table = possible_value_table.drop_duplicates().reset_index(drop=True)
    categorical_att_columns = [item for item in PVT_head if item not in numeric_attributes]
    max_index_PVT = [len(value) - 1 for value in possible_values_lists.values()]
    return possible_value_table, PVT_head, categorical_att_columns, max_index_PVT, possible_values_lists


# define a function to check if a value is an integer
def is_int(val):
    if isinstance(val, int):
        return True
    elif isinstance(val, float):
        return val.is_integer()
    else:
        return False


def build_PVT_relax_only(data, selected_attributes, numeric_attributes,
                         categorical_attributes, selection_numeric, selection_categorical,
                         sensitive_attributes, fairness_constraints,
                         fairness_constraints_provenance_greater_than,
                         fairness_constraints_provenance_smaller_than
                         ):
    """
    to build the sorted table
    :param fairness_constraints_provenance_greater_than:
    :param fairness_constraints_provenance_smaller_than:
    :param selected_attributes:
    :param data: dataframe
    :param numeric_attributes: list of names of numeric attributes [city, major, state]
    :param categorical_attributes: dictionary: {city: [domain of city], major: [domain of major]}
    :param selection_numeric: dictionary: {grade:[80, >], age:[30, <], hours: [100, <=]}
    :param selection_categorical: dictionary: {city: [accepted cities], major: [accepted majors]}
    :return: return the whole sorted table, including rows that already satisfy the selection conditions;
            also return delta table
    """
    PVT_head = numeric_attributes.copy()
    for att in categorical_attributes:
        domain = data[att].dropna().unique().tolist()
        for value in domain:
            if value not in selection_categorical[att]:
                col = att + "__" + value
                PVT_head.append(col)
    print("PVT_head: {}".format(PVT_head))

    # build delta table
    def itercol(col):
        nonlocal possible_values_sets
        nonlocal possible_values_lists
        att = col.name
        unique_values = col.unique()
        if att in selection_numeric:
            unique_values.sort()
            if selection_numeric[att][0] == '>' or selection_numeric[att][0] == '>=':
                try:
                    idx = next(i for i, v in enumerate(unique_values) if v >= selection_numeric[att][1])
                except StopIteration:
                    idx = len(unique_values) - 1
                # idx = next(i for i, v in enumerate(unique_values) if v >= selection_numeric[att][1])
                others = unique_values[:idx]
                possible_values_sets[att].update(others)
                lst = list(possible_values_sets[att])
                if len(lst) > 1:
                    lst.sort()
                    lst = lst[::-1]
                    possible_values_lists[att] = lst
                else:
                    del possible_values_sets[att]
                    PVT_head.remove(att)
            else:
                try:
                    idx = next(i for i, v in enumerate(unique_values) if v >= selection_numeric[att][1])
                except StopIteration:
                    idx = 0
                others = unique_values[idx:]
                possible_values_sets[att].update(others)
                lst = list(possible_values_sets[att])
                if len(lst) > 1:
                    lst.sort()
                    possible_values_lists[att] = lst
                else:
                    del possible_values_sets[att]
                    PVT_head.remove(att)

    # data_rows_greater_than = data_rows_greater_than.drop_duplicates(
    #     subset=selected_attributes,
    #     keep='first').reset_index(drop=True)
    possible_values_sets = {x: set() for x in PVT_head}
    for att in selection_numeric:
        possible_values_sets[att].add(selection_numeric[att][1])
    # data_rows_greater_than.apply(iterrow, args=(True,), axis=1)
    possible_values_lists = dict()
    data[selection_numeric.keys()].apply(itercol, axis=0)

    for att in PVT_head:
        if att not in selection_numeric:
            pre, v = att.split('__')
            if v not in selection_categorical[pre]:
                possible_values_lists[att] = [0, 1]

    # print("possible_values_lists:\n", possible_values_lists)
    possible_value_table = pd.DataFrame({key: pd.Series(value) for key, value in possible_values_lists.items()})
    # print("possible_value_table:\n", possible_value_table)
    possible_value_table = possible_value_table.drop_duplicates().reset_index(drop=True)
    categorical_att_columns = [item for item in PVT_head if item not in numeric_attributes]
    max_index_PVT = [len(value) - 1 for value in possible_values_lists.values()]
    return possible_value_table, PVT_head, categorical_att_columns, max_index_PVT


def build_PVT_contract_only(data, selected_attributes, numeric_attributes,
                            categorical_attributes, selection_numeric, selection_categorical,
                            sensitive_attributes, fairness_constraints,
                            fairness_constraints_provenance_greater_than,
                            fairness_constraints_provenance_smaller_than):
    """
    to build the sorted table
    :param fairness_constraints_provenance_greater_than:
    :param fairness_constraints_provenance_smaller_than:
    :param selected_attributes:
    :param data: dataframe
    :param numeric_attributes: list of names of numeric attributes [city, major, state]
    :param categorical_attributes: dictionary: {city: [domain of city], major: [domain of major]}
    :param selection_numeric: dictionary: {grade:[80, >], age:[30, <], hours: [100, <=]}
    :param selection_categorical: dictionary: {city: [accepted cities], major: [accepted majors]}
    :return: return the whole sorted table, including rows that already satisfy the selection conditions;
            also return delta table
    """

    PVT_head = numeric_attributes.copy()

    for att in categorical_attributes:
        for value in selection_categorical[att]:
            col = att + "__" + value
            PVT_head.append(col)

    # build delta table
    def itercol(col):
        nonlocal possible_values_sets
        nonlocal possible_values_lists
        att = col.name
        unique_values = col.unique()
        unique_values.sort()
        if att in selection_numeric:
            if selection_numeric[att][0] == '>' or selection_numeric[att][0] == '>=':
                try:
                    idx = next(i for i, v in enumerate(unique_values) if v >= selection_numeric[att][1])
                except StopIteration:
                    idx = 0
                # idx = next(i for i, v in enumerate(unique_values) if v >= selection_numeric[att][1])
                others = unique_values[idx:]
                possible_values_sets[att].update([s + selection_numeric[att][2] for s in others])
                lst = list(possible_values_sets[att])
                if len(lst) > 1:
                    lst.sort()
                    possible_values_lists[att] = lst
                else:
                    del possible_values_sets[att]
                    PVT_head.remove(att)
            else:
                try:
                    idx = next(i for i, v in enumerate(unique_values) if v >= selection_numeric[att][1])
                except StopIteration:
                    idx = len(unique_values) - 1
                # idx = next(i for i, v in enumerate(unique_values) if v >= selection_numeric[att][1])
                others = unique_values[:idx + 1]
                possible_values_sets[att].update([s - selection_numeric[att][2] for s in others])
                lst = list(possible_values_sets[att])
                if len(lst) > 1:
                    lst.sort()
                    lst = lst[::-1]
                    possible_values_lists[att] = lst
                else:
                    del possible_values_sets[att]
                    PVT_head.remove(att)

    # data_rows_smaller_than = data_rows_smaller_than.drop_duplicates(
    #     subset=selected_attributes,
    #     keep='first').reset_index(drop=True)
    possible_values_sets = {x: set() for x in PVT_head}
    for att in selection_numeric:
        possible_values_sets[att].add(selection_numeric[att][1])
    # data_rows_smaller_than.apply(iterrow, args=(True,), axis=1)
    possible_values_lists = dict()
    data[selection_numeric.keys()].apply(itercol, axis=0)

    for att in PVT_head:
        if att not in selection_numeric:
            pre, v = att.split("__")
            if v in selection_categorical[pre]:
                if len(selection_categorical[
                           pre]) == 1:  # if there is only one value in original query, can't remote it
                    possible_values_lists[att] = [1]
                else:
                    possible_values_lists[att] = [1, 0]
    possible_value_table = pd.DataFrame({key: pd.Series(value) for key, value in possible_values_lists.items()})
    # print("possible_value_table:\n", possible_value_table)
    possible_value_table = possible_value_table.drop_duplicates().reset_index(drop=True)
    categorical_att_columns = [item for item in PVT_head if item not in numeric_attributes]
    max_index_PVT = [len(value) - 1 for value in possible_values_lists.values()]
    return possible_value_table, PVT_head, categorical_att_columns, max_index_PVT


def assign_to_provenance_relax_only_partial_query(value_assignment, numeric_attributes, categorical_attributes,
                                                  selection_numeric, selection_categorical, columns_delta_table,
                                                  fairness_constraints_provenance_greater_than):
    global assign_to_provenance_num
    assign_to_provenance_num += 1
    # greater than
    for fc in fairness_constraints_provenance_greater_than:
        pe_dataframe = fc["provenance_expression"]
        for va in numeric_attributes:
            if va in value_assignment:
                if selection_numeric[va][0] == '>':
                    pe_dataframe = pe_dataframe[pe_dataframe[va] > value_assignment[va]]
                elif selection_numeric[va][0] == ">=":
                    pe_dataframe = pe_dataframe[pe_dataframe[va] >= value_assignment[va]]
                elif selection_numeric[va][0] == "<":
                    pe_dataframe = pe_dataframe[pe_dataframe[va] < value_assignment[va]]
                else:
                    pe_dataframe = pe_dataframe[pe_dataframe[va] <= value_assignment[va]]
        if not eval(str(pe_dataframe["occurrence"].sum()) + fc['symbol'] + str(fc['number'])):
            return False
        not_included = [x for x in value_assignment if (x not in numeric_attributes and value_assignment[x] == 0)]
        not_included_dic = dict()
        for cate in not_included:
            at, va = cate.rsplit("__", 1)
            if at not in not_included_dic:
                not_included_dic[at] = [va]
            else:
                not_included_dic[at].append(va)
        for at in not_included_dic:
            pe_dataframe = pe_dataframe[~pe_dataframe[at].isin(not_included_dic[at])]
        if not eval(str(pe_dataframe["occurrence"].sum()) + fc['symbol'] + str(fc['number'])):
            return False
    return True


def assign_to_provenance_relax_only(value_assignment, numeric_attributes, categorical_attributes, selection_numeric,
                                    selection_categorical, columns_delta_table,
                                    fairness_constraints_provenance_greater_than):
    global assign_to_provenance_num
    assign_to_provenance_num += 1
    # greater than
    for fc in fairness_constraints_provenance_greater_than:
        pe_dataframe = fc["provenance_expression"]
        for va in numeric_attributes:
            if va in value_assignment:
                if selection_numeric[va][0] == '>':
                    pe_dataframe = pe_dataframe[pe_dataframe[va] > value_assignment[va]]
                elif selection_numeric[va][0] == ">=":
                    pe_dataframe = pe_dataframe[pe_dataframe[va] >= value_assignment[va]]
                elif selection_numeric[va][0] == "<":
                    pe_dataframe = pe_dataframe[pe_dataframe[va] < value_assignment[va]]
                else:
                    pe_dataframe = pe_dataframe[pe_dataframe[va] <= value_assignment[va]]
        if not eval(str(pe_dataframe["occurrence"].sum()) + fc['symbol'] + str(fc['number'])):
            return False
        # not_included = [x for x in value_assignment if (x not in numeric_attributes and value_assignment[x] == 0)]
        included = [x for x in value_assignment if (x not in numeric_attributes and value_assignment[x] == 1)]
        new_select = copy.deepcopy(selection_categorical)
        for cate in included:
            at, va = cate.rsplit("__", 1)
            new_select[at].append(va)
        for att in new_select:
            pe_dataframe = pe_dataframe[pe_dataframe[att].isin(new_select[att])]
        if not eval(str(pe_dataframe["occurrence"].sum()) + fc['symbol'] + str(fc['number'])):
            return False
    return True


def assign_to_provenance_contract_only(value_assignment, numeric_attributes, categorical_attributes, selection_numeric,
                                       selection_categorical, columns_delta_table,
                                       fairness_constraints_provenance_smaller_than):
    global assign_to_provenance_num
    assign_to_provenance_num += 1
    # smaller than
    for fc in fairness_constraints_provenance_smaller_than:
        pe_dataframe = fc["provenance_expression"]
        for va in numeric_attributes:
            if va in value_assignment:
                if selection_numeric[va][0] == '>':
                    pe_dataframe = pe_dataframe[pe_dataframe[va] > value_assignment[va]]
                elif selection_numeric[va][0] == ">=":
                    pe_dataframe = pe_dataframe[pe_dataframe[va] >= value_assignment[va]]
                elif selection_numeric[va][0] == "<":
                    pe_dataframe = pe_dataframe[pe_dataframe[va] < value_assignment[va]]
                else:
                    pe_dataframe = pe_dataframe[pe_dataframe[va] <= value_assignment[va]]
        if eval(str(pe_dataframe["occurrence"].sum()) + fc['symbol'] + str(fc['number'])):
            continue
        not_included = [x for x in value_assignment if (x not in numeric_attributes and value_assignment[x] == 0)]
        new_select = copy.deepcopy(selection_categorical)
        for cate in not_included:
            at, va = cate.rsplit("__", 1)
            if va in new_select[at]:
                new_select[at].remove(va)
        for att in new_select:
            pe_dataframe = pe_dataframe[pe_dataframe[att].isin(new_select[att])]
        if not eval(str(pe_dataframe["occurrence"].sum()) + fc['symbol'] + str(fc['number'])):
            return False
    return True


def assign_to_provenance_relax_in_refinement(value_assignment, numeric_attributes, categorical_attributes,
                                             selection_numeric,
                                             selection_categorical, columns_delta_table,
                                             fairness_constraints_provenance_greater_than):
    global assign_to_provenance_num
    assign_to_provenance_num += 1
    # greater than
    for fc in fairness_constraints_provenance_greater_than:
        pe_dataframe = fc["provenance_expression"]
        for va in numeric_attributes:
            if va in value_assignment:
                if selection_numeric[va][0] == '>':
                    pe_dataframe = pe_dataframe[pe_dataframe[va] > value_assignment[va]]
                elif selection_numeric[va][0] == ">=":
                    pe_dataframe = pe_dataframe[pe_dataframe[va] >= value_assignment[va]]
                elif selection_numeric[va][0] == "<":
                    pe_dataframe = pe_dataframe[pe_dataframe[va] < value_assignment[va]]
                else:
                    pe_dataframe = pe_dataframe[pe_dataframe[va] <= value_assignment[va]]
        if not eval(str(pe_dataframe["occurrence"].sum()) + fc['symbol'] + str(fc['number'])):
            return False
        not_included = [x for x in value_assignment if (x not in numeric_attributes and value_assignment[x] == 0)]
        included = [x for x in value_assignment if (x not in numeric_attributes and value_assignment[x] == 1)]
        new_select = copy.deepcopy(selection_categorical)
        for cate in included:
            at, va = cate.rsplit("__", 1)
            if va not in new_select[at]:
                new_select[at].append(va)
        for cate in not_included:
            at, va = cate.rsplit("__", 1)
            if va in new_select[at]:
                new_select[at].remove(va)
        for att in new_select:
            pe_dataframe = pe_dataframe[pe_dataframe[att].isin(new_select[att])]
        if not eval(str(pe_dataframe["occurrence"].sum()) + fc['symbol'] + str(fc['number'])):
            return False
    return True


def assign_to_provenance_contract_in_refinement(value_assignment, numeric_attributes, categorical_attributes,
                                                selection_numeric,
                                                selection_categorical, columns_delta_table,
                                                fairness_constraints_provenance_smaller_than):
    global assign_to_provenance_num
    assign_to_provenance_num += 1
    # smaller than
    for fc in fairness_constraints_provenance_smaller_than:
        pe_dataframe = fc["provenance_expression"]
        for va in numeric_attributes:
            if va in value_assignment:
                if selection_numeric[va][0] == '>':
                    pe_dataframe = pe_dataframe[pe_dataframe[va] > value_assignment[va]]
                elif selection_numeric[va][0] == ">=":
                    pe_dataframe = pe_dataframe[pe_dataframe[va] >= value_assignment[va]]
                elif selection_numeric[va][0] == "<":
                    pe_dataframe = pe_dataframe[pe_dataframe[va] < value_assignment[va]]
                else:
                    pe_dataframe = pe_dataframe[pe_dataframe[va] <= value_assignment[va]]
        if eval(str(pe_dataframe["occurrence"].sum()) + fc['symbol'] + str(fc['number'])):
            continue
        not_included = [x for x in value_assignment if (x not in numeric_attributes and value_assignment[x] == 0)]
        included = [x for x in value_assignment if (x not in numeric_attributes and value_assignment[x] == 1)]
        new_select = copy.deepcopy(selection_categorical)
        for cate in not_included:
            at, va = cate.rsplit("__", 1)
            if va in new_select[at]:
                new_select[at].remove(va)
        for cate in included:
            at, va = cate.rsplit("__", 1)
            if va not in new_select[at]:
                new_select[at].append(va)
        for att in new_select:
            pe_dataframe = pe_dataframe[pe_dataframe[att].isin(new_select[att])]
        if len(pe_dataframe) == 0:
            return False
        if not eval(str(pe_dataframe["occurrence"].sum()) + fc['symbol'] + str(fc['number'])):
            return False
    return True


# reason: 0: not relaxed enough, 1: not contracted enough, 3: complex
def assign_to_provenance(value_assignment, numeric_attributes, categorical_attributes, selection_numeric,
                         selection_categorical, columns_delta_table, fairness_constraints_provenance_greater_than,
                         fairness_constraints_provenance_smaller_than, fairness_constraints_provenance_complex):
    global assign_to_provenance_num
    assign_to_provenance_num += 1

    def filter(pe_dataframe):
        for va in numeric_attributes:
            if va in value_assignment:
                if selection_numeric[va][0] == '>':
                    pe_dataframe = pe_dataframe[pe_dataframe[va] > value_assignment[va]]
                elif selection_numeric[va][0] == ">=":
                    pe_dataframe = pe_dataframe[pe_dataframe[va] >= value_assignment[va]]
                elif selection_numeric[va][0] == "<":
                    pe_dataframe = pe_dataframe[pe_dataframe[va] < value_assignment[va]]
                else:
                    pe_dataframe = pe_dataframe[pe_dataframe[va] <= value_assignment[va]]
        included = [x for x in value_assignment if (x not in numeric_attributes and value_assignment[x] == 1)]
        new_select = copy.deepcopy(selection_categorical)
        for cate in included:
            at, va = cate.rsplit("__", 1)
            new_select[at].append(va)
        for att in new_select:
            pe_dataframe = pe_dataframe[pe_dataframe[att].isin(new_select[att])]
        return pe_dataframe

    # check fairness_constraints_provenance_complex
    for fc in fairness_constraints_provenance_complex:
        pe_dataframe1 = filter(fc["first_provenance_expression"])
        pe_dataframe2 = filter(fc["second_provenance_expression"])
        if "plus" in fc:
            if not eval(str(pe_dataframe1["occurrence"].sum()) + fc['symbol'] +
                        str(pe_dataframe2["occurrence"].sum() + fc['plus'])):
                return False, 3
        elif "times" in fc:
            if not eval(str(pe_dataframe1["occurrence"].sum()) + fc['symbol'] +
                        str(pe_dataframe2["occurrence"].sum() * fc['times'])):
                return False, 3

    if len(fairness_constraints_provenance_greater_than) > 0:
        survive = assign_to_provenance_relax_in_refinement(value_assignment, numeric_attributes, categorical_attributes,
                                                           selection_numeric, selection_categorical,
                                                           columns_delta_table,
                                                           fairness_constraints_provenance_greater_than)
        if not survive:
            # print("not relaxed enough")
            return False, 0
    if len(fairness_constraints_provenance_smaller_than) > 0:
        survive = assign_to_provenance_contract_in_refinement(value_assignment, numeric_attributes,
                                                              categorical_attributes,
                                                              selection_numeric, selection_categorical,
                                                              columns_delta_table,
                                                              fairness_constraints_provenance_smaller_than)
        # if not survive:
        #     print("relaxed enough but not contracted enough")
        return survive, 1
    return True, 1


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


def position_dominate(p1, p2):
    p1_higher = False
    p2_higher = False
    length = len(p1)
    for i in range(length):
        if p1[i] < p2[i]:
            p1_higher = True
        elif p2[i] < p1[i]:
            p2_higher = True
    if p1_higher and not p2_higher:
        return 1
    elif p2_higher and not p1_higher:
        return 2
    elif p2_higher and p1_higher:
        return 3
    else:  # p1 == p2
        return 4


def update_minimal_relaxation_and_position(minimal_refinements, minimal_refinements_positions,
                                           full_value_assignment, full_value_assignment_positions, shifted_length):
    num = len(minimal_refinements_positions)
    dominated = []
    dominated_refinements = []
    full_value_assignment_positions = [full_value_assignment_positions[i] + shifted_length[i] for i in
                                       range(len(shifted_length))]
    for i in range(num):
        p = minimal_refinements_positions[i]
        pd = position_dominate(p, full_value_assignment_positions)
        if pd == 1 or pd == 4:
            return minimal_refinements, minimal_refinements_positions, False
        elif pd == 2:
            dominated.append(p)
            to_remove_refinement = minimal_refinements[i]
            dominated_refinements.append(to_remove_refinement)
    minimal_refinements_positions = [p for p in minimal_refinements_positions if p not in dominated]
    minimal_refinements_positions.append(full_value_assignment_positions)
    minimal_refinements = [p for p in minimal_refinements if p not in dominated_refinements]
    minimal_refinements.append(full_value_assignment)
    return minimal_refinements, minimal_refinements_positions, True


def update_minimal_relaxation_and_position_refinement(minimal_refinements, minimal_refinements_positions,
                                                      full_value_assignment, full_value_assignment_positions,
                                                      shifted_length,
                                                      initial_PVT, selection_numeric, full_PVT_head):
    num = len(minimal_refinements_positions)
    dominated = []
    dominated_refinements = []
    full_value_assignment_positions = [full_value_assignment_positions[i] + shifted_length[i] for i in
                                       range(len(shifted_length))]
    # deal with the situation that absolute difference are the same, for numeric attributes
    for i in range(len(full_value_assignment_positions)):
        pos = full_value_assignment_positions[i]
        att = full_PVT_head[i]
        if att in selection_numeric and pos >= 2:
            if abs(initial_PVT.loc[pos, att] - initial_PVT.loc[0, att]) == \
                    abs(initial_PVT.loc[pos - 1, att] - initial_PVT.loc[0, att]):
                full_value_assignment_positions[i] = pos - 1
    for i in range(num):
        p = minimal_refinements_positions[i]
        pd = position_dominate(p, full_value_assignment_positions)
        if pd == 1 or pd == 4:
            return minimal_refinements, minimal_refinements_positions, False
        elif pd == 2:
            dominated.append(p)
            to_remove_refinement = minimal_refinements[i]
            dominated_refinements.append(to_remove_refinement)
    minimal_refinements_positions = [p for p in minimal_refinements_positions if p not in dominated]
    minimal_refinements_positions.append(full_value_assignment_positions)
    minimal_refinements = [p for p in minimal_refinements if p not in dominated_refinements]
    minimal_refinements.append(full_value_assignment)
    return minimal_refinements, minimal_refinements_positions, True


def searchPVT_relaxation(PVT, PVT_head, numeric_attributes, categorical_attributes,
                         selection_numeric, selection_categorical, num_columns,
                         fairness_constraints_provenance_greater_than,
                         full_PVT, full_PVT_head, max_index_PVT,
                         checked_assignments_satisfying, checked_assignments_not_satisfying, time_limit=5 * 60):
    time1 = time.time()
    global assign_to_provenance_num
    assign_to_provenance_num = 0
    PVT_stack = [PVT]
    PVT_head_stack = [PVT_head]
    max_index_PVT_stack = [max_index_PVT]
    parent_PVT_stack = [pd.DataFrame()]
    parent_PVT_head_stack = [[]]
    parent_max_index_PVT_stack = [pd.DataFrame()]
    col_idx_in_parent_PVT_stack = [0]
    idx_in_this_col_in_parent_PVT_stack = [0]
    find_relaxation = {x: [] for x in range(1, len(full_PVT_head) + 1)}
    fixed_value_assignments_stack = [{}]
    fixed_value_assignments_positions_stack = [{}]
    fixed_value_assignments_to_tighten_stack = [[]]
    left_side_binary_search_stack = [0]
    shifted_length_stack = [[0] * num_columns]
    to_put_to_stack = []
    minimal_refinements = []  # result set
    minimal_refinements_positions = []  # positions of result set
    fixed_value_assignments = {}
    fixed_value_assignments_positions = {}
    num_iterations = 0
    while PVT_stack:
        if time.time() - time1 > time_limit:
            print("provenance search alg time out")
            return minimal_refinements
        num_iterations += 1
        PVT = PVT_stack.pop()
        PVT_head = PVT_head_stack.pop()
        max_index_PVT = max_index_PVT_stack.pop()
        parent_PVT = parent_PVT_stack.pop()
        parent_PVT_head = parent_PVT_head_stack.pop()
        parent_max_index_PVT = parent_max_index_PVT_stack.pop()
        col_idx_in_parent_PVT = col_idx_in_parent_PVT_stack.pop()
        idx_in_this_col_in_parent_PVT = idx_in_this_col_in_parent_PVT_stack.pop()
        if idx_in_this_col_in_parent_PVT > 0:
            values_above = fixed_value_assignments_to_tighten_stack.pop()
        else:
            values_above = []
        fixed_value_assignments = fixed_value_assignments_stack.pop()
        fixed_value_assignments_positions = fixed_value_assignments_positions_stack.pop()
        shifted_length = shifted_length_stack.pop()
        find_bounding_relaxation = False
        num_columns = len(PVT_head)
        # print("==========================  searchPVT  ========================== ")
        # print("PVT_head: {}".format(PVT_head))
        # print("PVT:\n{}".format(PVT))
        # print("fixed_value_assignments: {}".format(fixed_value_assignments))
        # print("fixed_value_assignments_positions: {}".format(fixed_value_assignments_positions))
        # print("shifted_length: {}".format(shifted_length))
        # print("idx_in_this_col_in_parent_PVT:{}".format(idx_in_this_col_in_parent_PVT))

        satisfying_row_id = 0
        new_value_assignment = []
        last_satisfying_new_value_assignment = []
        full_value_assignment = {}
        last_satisfying_full_value_assignment = {}
        last_satisfying_bounding_relaxation_location = []
        left = left_side_binary_search_stack.pop()
        left = max(left, 0)
        right = max(max_index_PVT)
        # binary search can't use apply
        while left <= right:
            if time.time() - time1 > time_limit:
                print("provenance search alg time out")
                return minimal_refinements
            cur_row_id = int((right + left) / 2)
            new_bounding_relaxation_location = [cur_row_id if cur_row_id < x else x for x in max_index_PVT]
            new_value_assignment = [PVT.iloc[new_bounding_relaxation_location[x], x] for x in range(len(PVT_head))]
            full_value_assignment = dict(zip(PVT_head, new_value_assignment))
            full_value_assignment = {**full_value_assignment, **fixed_value_assignments}
            # print("value_assignment: ", full_value_assignment)
            full_value_assignment_str = num2string([full_value_assignment[k] for k in full_PVT_head])
            if full_value_assignment_str in checked_assignments_satisfying:
                # print("{} satisfies constraints".format(full_value_assignment))
                satisfying_row_id = cur_row_id
                right = cur_row_id - 1
                last_satisfying_full_value_assignment = full_value_assignment
                last_satisfying_new_value_assignment = new_value_assignment
                last_satisfying_bounding_relaxation_location = new_bounding_relaxation_location
                find_bounding_relaxation = True
            elif full_value_assignment_str in checked_assignments_not_satisfying:
                # print("{} doesn't satisfy constraints".format(full_value_assignment))
                left = cur_row_id + 1
            elif assign_to_provenance_relax_only(full_value_assignment, numeric_attributes, categorical_attributes,
                                                 selection_numeric, selection_categorical, full_PVT_head,
                                                 fairness_constraints_provenance_greater_than):
                checked_assignments_satisfying.append(full_value_assignment_str)
                # print("{} satisfies constraints".format(full_value_assignment))
                satisfying_row_id = cur_row_id
                right = cur_row_id - 1
                last_satisfying_full_value_assignment = full_value_assignment
                last_satisfying_new_value_assignment = new_value_assignment
                last_satisfying_bounding_relaxation_location = new_bounding_relaxation_location
                find_bounding_relaxation = True
            else:
                # print("{} doesn't satisfy constraints".format(full_value_assignment))
                checked_assignments_not_satisfying.append(full_value_assignment_str)
                left = cur_row_id + 1

        col_idx = 0
        find_relaxation[num_columns].append(find_bounding_relaxation)  # TODO: is this find_relaxation necessary?
        if not find_bounding_relaxation:
            # print("no base refinement here, size of PVT: {}*{}".format(len(PVT), len(PVT_head)))
            if len(PVT_head_stack) > 0:
                next_col_num_in_stack = len(PVT_head_stack[-1])
            else:
                next_col_num_in_stack = len(full_PVT_head)
            check_to_put_to_stack(to_put_to_stack, next_col_num_in_stack, num_columns, find_relaxation,
                                  PVT_stack, PVT_head_stack, max_index_PVT_stack, parent_PVT_stack,
                                  parent_PVT_head_stack, parent_max_index_PVT_stack, col_idx_in_parent_PVT_stack,
                                  idx_in_this_col_in_parent_PVT_stack,
                                  fixed_value_assignments_stack, fixed_value_assignments_positions_stack,
                                  fixed_value_assignments_to_tighten_stack, left_side_binary_search_stack,
                                  shifted_length_stack, idx_in_this_col_in_parent_PVT,
                                  PVT, PVT_head, max_index_PVT, parent_PVT, parent_PVT_head, parent_max_index_PVT,
                                  col_idx_in_parent_PVT, fixed_value_assignments, fixed_value_assignments_positions)
            continue

        full_value_assignment = last_satisfying_full_value_assignment
        new_value_assignment = last_satisfying_new_value_assignment
        if num_columns > 1:  # if there are more than one column, we can tighten the result
            nan_row = PVT.iloc[satisfying_row_id].isnull()
            col_non_tightenable = -1
            if sum(k is False for k in nan_row) == 1:
                true_lst = np.where(nan_row)[0]
                range_lst = range(0, num_columns)
                col_non_tightenable = [x for x in range_lst if x not in true_lst][0]
                # print("col {} doesn't need to be tightened".format(col_non_tightenable))

            tmp_max_idx_of_ol = 0

            def tighten_result(column):
                nonlocal col_idx
                nonlocal last_satisfying_full_value_assignment
                nonlocal tmp_max_idx_of_ol
                idx_in_this_col = last_satisfying_bounding_relaxation_location[col_idx]
                if col_idx == col_non_tightenable:
                    col_idx += 1
                    tmp_max_idx_of_ol = max(tmp_max_idx_of_ol, idx_in_this_col)
                    return
                while idx_in_this_col > 0:
                    idx_in_this_col -= 1
                    new_value_assignment[col_idx] = column[idx_in_this_col]
                    full_value_assignment[PVT_head[col_idx]] = column[idx_in_this_col]
                    # print("value_assignment: ", full_value_assignment)
                    full_value_assignment_str = num2string([full_value_assignment[k] for k in full_PVT_head])
                    if full_value_assignment_str in checked_assignments_satisfying:
                        # print("{} satisfies constraints".format(full_value_assignment))
                        last_satisfying_full_value_assignment = full_value_assignment
                        last_satisfying_bounding_relaxation_location[col_idx] = idx_in_this_col
                    elif full_value_assignment_str in checked_assignments_not_satisfying:
                        # print("{} doesn't satisfy constraints".format(full_value_assignment))
                        smallest_row = idx_in_this_col + 1
                        # last_satisfying_bounding_relaxation_location[col_idx] = smallest_row
                        new_value_assignment[col_idx] = column[smallest_row]
                        full_value_assignment[PVT_head[col_idx]] = column[smallest_row]
                        break
                    elif assign_to_provenance_relax_only(full_value_assignment, numeric_attributes,
                                                         categorical_attributes, selection_numeric,
                                                         selection_categorical, full_PVT_head,
                                                         fairness_constraints_provenance_greater_than):
                        checked_assignments_satisfying.append(full_value_assignment_str)
                        # print("{} satisfies constraints".format(full_value_assignment))
                        last_satisfying_full_value_assignment = full_value_assignment
                        last_satisfying_bounding_relaxation_location[col_idx] = idx_in_this_col
                    else:
                        # print("{} doesn't satisfy constraints".format(full_value_assignment))
                        checked_assignments_not_satisfying.append(full_value_assignment_str)
                        smallest_row = idx_in_this_col + 1
                        # last_satisfying_bounding_relaxation_location[col_idx] = smallest_row
                        new_value_assignment[col_idx] = column[smallest_row]
                        full_value_assignment[PVT_head[col_idx]] = column[smallest_row]
                        break

                col_idx += 1
                return

            PVT.apply(tighten_result, axis=0)
            # print("tight relaxation: {}".format(new_value_assignment))

        # optimization: tighten the last fixed column
        tight_value_idx = -1
        fixed_att = str()
        if idx_in_this_col_in_parent_PVT > 0:
            tight_success = False
            # binary search to tighten this column
            left = 0
            right = len(values_above) - 1
            fixed_att = list(fixed_value_assignments.keys())[-1]
            # print("tighten the last fixed column {}:\n {}".format(fixed_att, values_above))
            fixed_value_assignments_for_tighten = copy.deepcopy(fixed_value_assignments)
            while left <= right:
                cur_value_id = int((right + left) / 2)
                cur_fixed_value = values_above[cur_value_id]
                fixed_value_assignments_for_tighten[fixed_att] = cur_fixed_value
                full_value_assignment = {**dict(zip(PVT_head, new_value_assignment)),
                                         **fixed_value_assignments_for_tighten}
                # print("value_assignment: ", full_value_assignment)
                full_value_assignment_str = num2string([full_value_assignment[k] for k in full_PVT_head])
                if full_value_assignment_str in checked_assignments_satisfying:
                    # print("{} satisfies constraints".format(full_value_assignment))
                    right = cur_value_id - 1
                    tight_value_idx = cur_value_id
                elif full_value_assignment_str in checked_assignments_not_satisfying:
                    # print("{} doesn't satisfy constraints".format(full_value_assignment))
                    left = cur_value_id + 1
                elif assign_to_provenance_relax_only(full_value_assignment, numeric_attributes, categorical_attributes,
                                                     selection_numeric, selection_categorical, full_PVT_head,
                                                     fairness_constraints_provenance_greater_than):
                    checked_assignments_satisfying.append(full_value_assignment_str)
                    # print("{} satisfies constraints".format(full_value_assignment))
                    right = cur_value_id - 1
                    tight_value_idx = cur_value_id
                else:
                    # print("{} doesn't satisfy constraints".format(full_value_assignment))
                    checked_assignments_not_satisfying.append(full_value_assignment_str)
                    left = cur_value_id + 1
            if tight_value_idx >= 0:  # can be tightened
                if tight_value_idx < idx_in_this_col_in_parent_PVT:
                    # tight this fixed column successfully
                    # last_satisfying_bounding_relaxation_location[PVT_head.index(fixed_att)] = tight_value_idx
                    fixed_value_assignments_for_tighten[fixed_att] = values_above[tight_value_idx]
                    full_value_assignment[fixed_att] = values_above[tight_value_idx]
                    tight_success = True
                fva = [full_value_assignment[k] for k in full_PVT_head]
                full_value_assignment_positions = dict(zip(PVT_head, last_satisfying_bounding_relaxation_location))
                full_value_assignment_positions = {**full_value_assignment_positions,
                                                   **fixed_value_assignments_positions, fixed_att: tight_value_idx}
            else:
                full_value_assignment = {**dict(zip(PVT_head, new_value_assignment)),
                                         **fixed_value_assignments}
                fva = [full_value_assignment[k] for k in full_PVT_head]
                full_value_assignment_positions = dict(zip(PVT_head, last_satisfying_bounding_relaxation_location))
                full_value_assignment_positions = {**full_value_assignment_positions,
                                                   **fixed_value_assignments_positions}
            minimal_refinements, minimal_refinements_positions, added = \
                update_minimal_relaxation_and_position(minimal_refinements, minimal_refinements_positions,
                                                       fva, [full_value_assignment_positions[x] for x in full_PVT_head],
                                                       shifted_length)
            # print("find base refinement {}".format(new_value_assignment))
            # print("position: {}".format(full_value_assignment_positions))
            # print("minimal_refinements: {}".format(minimal_refinements))
            # print("num of minimal_refinements: {}".format(len(minimal_refinements)))

            # if tight success, when doing recursion, fixed attribute should use the tightened value
            # and if tight success, there is no need to traverse between original position fixed and the tightened one
            if tight_success and added:
                # in the following situation, we need to traverse the values between tightened and original value
                if not all(v == 0 for v in last_satisfying_bounding_relaxation_location):
                    for j in range(idx_in_this_col_in_parent_PVT - 1, tight_value_idx, -1):
                        to_put = copy.deepcopy(to_put_to_stack[-1])
                        to_put['idx_in_this_col_in_parent_PVT'] = j
                        to_put['fixed_value_assignments'][fixed_att] = values_above[j]
                        to_put['fixed_value_assignments_to_tighten'] = values_above[:j]
                        to_put['fixed_value_assignments_positions'][fixed_att] = j
                        PVT_stack.append(to_put['PVT'])
                        PVT_head_stack.append(to_put['PVT_head'])
                        max_index_PVT_stack.append(to_put['max_index_PVT'])
                        parent_PVT_stack.append(to_put['parent_PVT'])
                        parent_PVT_head_stack.append(to_put['parent_PVT_head'])
                        parent_max_index_PVT_stack.append(to_put['parent_max_index_PVT'])
                        col_idx_in_parent_PVT_stack.append(to_put['col_idx_in_parent_PVT'])
                        idx_in_this_col_in_parent_PVT_stack.append(to_put['idx_in_this_col_in_parent_PVT'])
                        fixed_value_assignments_stack.append(to_put['fixed_value_assignments'])
                        fixed_value_assignments_positions_stack.append(to_put['fixed_value_assignments_positions'])
                        left_side_binary_search_stack.append(to_put['for_left_binary'])
                        shifted_length_stack.append(to_put['shifted_length'])
                        fixed_value_assignments_to_tighten_stack.append(to_put['fixed_value_assignments_to_tighten'])
                fixed_value_assignments = fixed_value_assignments_for_tighten
                fixed_value_assignments_positions[fixed_att] = tight_value_idx
                if tight_value_idx == 0 and len(to_put_to_stack) > 0:
                    to_put_to_stack.pop()
                else:
                    #  no need to test values between now and last fixed value
                    if len(to_put_to_stack) > 0:
                        to_put_to_stack[-1]['idx_in_this_col_in_parent_PVT'] = tight_value_idx - 1
                        to_put_to_stack[-1]['fixed_value_assignments'][fixed_att] = values_above[tight_value_idx - 1]
                        to_put_to_stack[-1]['fixed_value_assignments_to_tighten'] = values_above[: tight_value_idx - 1]
                        to_put_to_stack[-1]['fixed_value_assignments_positions'][fixed_att] = tight_value_idx - 1
        else:
            fva = [full_value_assignment[k] for k in full_PVT_head]
            full_value_assignment_positions = dict(zip(PVT_head, last_satisfying_bounding_relaxation_location))
            full_value_assignment_positions = {**full_value_assignment_positions, **fixed_value_assignments_positions}

            minimal_refinements, minimal_refinements_positions, added = \
                update_minimal_relaxation_and_position(minimal_refinements, minimal_refinements_positions,
                                                       fva, [full_value_assignment_positions[x] for x in full_PVT_head],
                                                       shifted_length)
            # print("find base refinement {}".format(new_value_assignment))
            # print("position: {}".format(full_value_assignment_positions))
            # print("minimal_refinements: {}".format(minimal_refinements))
            # print("num of minimal_refinements: {}".format(len(minimal_refinements)))

        if num_columns == 1:
            if len(PVT_head_stack) > 0:
                next_col_num_in_stack = len(PVT_head_stack[-1])
            else:
                next_col_num_in_stack = len(full_PVT_head)
            check_to_put_to_stack(to_put_to_stack, next_col_num_in_stack, num_columns, find_relaxation,
                                  PVT_stack, PVT_head_stack, max_index_PVT_stack, parent_PVT_stack,
                                  parent_PVT_head_stack,
                                  parent_max_index_PVT_stack, col_idx_in_parent_PVT_stack,
                                  idx_in_this_col_in_parent_PVT_stack,
                                  fixed_value_assignments_stack, fixed_value_assignments_positions_stack,
                                  fixed_value_assignments_to_tighten_stack, left_side_binary_search_stack,
                                  shifted_length_stack,
                                  idx_in_this_col_in_parent_PVT,
                                  PVT, PVT_head, max_index_PVT, parent_PVT, parent_PVT_head, parent_max_index_PVT,
                                  col_idx_in_parent_PVT, fixed_value_assignments, fixed_value_assignments_positions)
            continue
        # recursion
        col_idx = 0
        if len(PVT_head_stack) > 0:
            next_col_num_in_stack = len(PVT_head_stack[-1])
        else:
            next_col_num_in_stack = len(full_PVT_head)
        check_to_put_to_stack(to_put_to_stack, next_col_num_in_stack, num_columns, find_relaxation,
                              PVT_stack, PVT_head_stack, max_index_PVT_stack, parent_PVT_stack, parent_PVT_head_stack,
                              parent_max_index_PVT_stack, col_idx_in_parent_PVT_stack,
                              idx_in_this_col_in_parent_PVT_stack, fixed_value_assignments_stack,
                              fixed_value_assignments_positions_stack, fixed_value_assignments_to_tighten_stack,
                              left_side_binary_search_stack, shifted_length_stack, idx_in_this_col_in_parent_PVT,
                              PVT, PVT_head, max_index_PVT, parent_PVT, parent_PVT_head, parent_max_index_PVT,
                              col_idx_in_parent_PVT, fixed_value_assignments, fixed_value_assignments_positions)
        index_to_insert_to_stack = len(PVT_stack)
        insert_idx_fixed_value_assignments_to_tighten_stack = len(fixed_value_assignments_to_tighten_stack)
        index_to_insert_to_put = len(to_put_to_stack)
        original_max_index_PVT = max_index_PVT.copy()
        original_shifted_length = copy.deepcopy(shifted_length)

        no_recursion_to_do = True

        def recursion(column):
            nonlocal col_idx
            nonlocal new_value_assignment
            nonlocal last_satisfying_bounding_relaxation_location
            nonlocal shifted_length
            nonlocal no_recursion_to_do
            idx_in_this_col = last_satisfying_bounding_relaxation_location[col_idx]
            # optimization: if there are no other columns to be moved down, return
            if idx_in_this_col == 0:
                col_idx += 1
                return
            if sum(last_satisfying_bounding_relaxation_location[i] < original_max_index_PVT[i] for i in
                   range(len(PVT_head)) if
                   i != col_idx) == 0:
                col_idx += 1
                return
            idx_in_this_col -= 1
            found = False
            # optimization: fixing this value doesn't dissatisfy inequalities
            one_more_fix = copy.deepcopy(fixed_value_assignments)
            while idx_in_this_col >= 0:
                one_more_fix[PVT_head[col_idx]] = column[idx_in_this_col]
                if not assign_to_provenance_relax_only_partial_query(one_more_fix, numeric_attributes,
                                                                     categorical_attributes,
                                                                     selection_numeric, selection_categorical,
                                                                     full_PVT_head,
                                                                     fairness_constraints_provenance_greater_than):
                    # print("fixing {} = {} dissatisfies constraints".format(PVT_head[col_idx], column[idx_in_this_col]))
                    idx_in_this_col -= 1
                else:
                    found = True
                    break
            if not found:
                col_idx += 1
                return
            fixed_value_assignments_for_stack = copy.deepcopy(fixed_value_assignments)
            fixed_value_assignments_for_stack[PVT_head[col_idx]] = column[idx_in_this_col]
            fixed_value_assignments_positions_for_stack = copy.deepcopy(fixed_value_assignments_positions)
            fixed_value_assignments_positions_for_stack[PVT_head[col_idx]] = idx_in_this_col
            new_PVT_head = [PVT_head[x] for x in range(len(PVT_head)) if x != col_idx]
            new_max_index_PVT = max_index_PVT[:col_idx] + max_index_PVT[col_idx + 1:]
            # optimization: if there is only one column left to be moved down,
            #  this column in the new recursion should start from where it stopped before
            if len(new_PVT_head) == 1:
                if col_idx == 0:
                    PVT_for_recursion = PVT[new_PVT_head].iloc[
                                        last_satisfying_bounding_relaxation_location[1] + 1:
                                        max(new_max_index_PVT) + 1].reset_index(drop=True)
                    shifted_length[full_PVT_head.index(PVT_head[1])] += \
                        last_satisfying_bounding_relaxation_location[1] + 1
                    new_max_index_PVT = [len(PVT_for_recursion) - 1]
                else:
                    PVT_for_recursion = PVT[new_PVT_head].iloc[: new_max_index_PVT[0] + 1].reset_index(drop=True)
                    # shifted_length[full_PVT_head.index(PVT_head[1])] -= \
                    #     last_satisfying_bounding_relaxation_location[1] + 1
                    shifted_length = original_shifted_length
            else:
                PVT_for_recursion = PVT[new_PVT_head].head(max(new_max_index_PVT) + 1)
                shifted_length = original_shifted_length
            PVT_stack.insert(index_to_insert_to_stack, PVT_for_recursion)
            PVT_head_stack.insert(index_to_insert_to_stack, new_PVT_head)
            max_index_PVT_stack.insert(index_to_insert_to_stack, new_max_index_PVT)
            parent_PVT_stack.insert(index_to_insert_to_stack, PVT.copy())
            parent_PVT_head_stack.insert(index_to_insert_to_stack, PVT_head)
            parent_max_index_PVT_stack.insert(index_to_insert_to_stack, max_index_PVT)
            col_idx_in_parent_PVT_stack.insert(index_to_insert_to_stack, col_idx)
            idx_in_this_col_in_parent_PVT_stack.insert(index_to_insert_to_stack, idx_in_this_col)
            fixed_value_assignments_stack.insert(index_to_insert_to_stack, fixed_value_assignments_for_stack)
            fixed_value_assignments_positions_stack.insert(index_to_insert_to_stack,
                                                           fixed_value_assignments_positions_for_stack)
            before_shift = last_satisfying_bounding_relaxation_location[:col_idx] + \
                           last_satisfying_bounding_relaxation_location[col_idx + 1:]
            shift_for_col = [shifted_length[PVT_head.index(att)] for att in PVT_head]
            shift_len = shift_for_col[:col_idx] + shift_for_col[col_idx + 1:]
            after_shift = [before_shift[i] - shift_len[i] for i in range(num_columns - 1)]
            for_left_binary = max(after_shift)
            left_side_binary_search_stack.insert(index_to_insert_to_stack, for_left_binary)
            shifted_length_stack.insert(index_to_insert_to_stack, copy.deepcopy(shifted_length))
            if idx_in_this_col > 0:
                fixed_value_assignments_to_tighten_stack.insert(insert_idx_fixed_value_assignments_to_tighten_stack,
                                                                column[:idx_in_this_col].copy())
                # to_put_to_stack
                to_put = dict()
                to_put['PVT'] = PVT_for_recursion.copy()
                to_put['PVT_head'] = new_PVT_head.copy()
                to_put['max_index_PVT'] = new_max_index_PVT.copy()
                to_put['parent_PVT'] = PVT.copy()
                to_put['parent_PVT_head'] = PVT_head.copy()
                to_put['parent_max_index_PVT'] = max_index_PVT.copy()
                to_put['col_idx_in_parent_PVT'] = col_idx
                to_put['idx_in_this_col_in_parent_PVT'] = idx_in_this_col - 1
                fixed_value_assignments_to_put = copy.deepcopy(fixed_value_assignments_for_stack)
                fixed_value_assignments_to_put[PVT_head[col_idx]] = column[idx_in_this_col - 1]
                to_put['fixed_value_assignments'] = fixed_value_assignments_to_put
                fixed_value_assignments_positions_to_put = copy.deepcopy(fixed_value_assignments_positions_for_stack)
                fixed_value_assignments_positions_to_put[PVT_head[col_idx]] = idx_in_this_col - 1
                to_put['fixed_value_assignments_positions'] = fixed_value_assignments_positions_to_put
                if to_put['idx_in_this_col_in_parent_PVT'] > 0:
                    to_put['fixed_value_assignments_to_tighten'] = column[:idx_in_this_col - 1].copy()
                to_put['for_left_binary'] = for_left_binary
                to_put['shifted_length'] = copy.deepcopy(shifted_length)
                to_put_to_stack.insert(index_to_insert_to_put, to_put)

            # avoid repeated checking: for columns that are done with moving up,
            #  we need to remove values above the 'stop line'
            seri = PVT[PVT_head[col_idx]]
            PVT[PVT_head[col_idx]] = seri.shift(periods=-last_satisfying_bounding_relaxation_location[col_idx])
            max_index_PVT[col_idx] -= last_satisfying_bounding_relaxation_location[col_idx]
            original_shifted_length[full_PVT_head.index(PVT_head[col_idx])] += \
                last_satisfying_bounding_relaxation_location[col_idx]
            col_idx += 1
            no_recursion_to_do = False
            return

        PVT.apply(recursion, axis=0)
        if no_recursion_to_do and tight_value_idx > 0:
            # print("no_recursion_to_do")
            if len(PVT_head_stack) > 0:
                if PVT_head == PVT_head_stack[-1]:
                    fixed_value_assignments_positions_stack[-1][fixed_att] = tight_value_idx - 1
                    fixed_value_assignments_to_tighten_stack[-1] = values_above[: tight_value_idx - 1]
                    fixed_value_assignments_stack[-1][fixed_att] = values_above[tight_value_idx - 1]
                    idx_in_this_col_in_parent_PVT_stack[-1] = tight_value_idx - 1
            if len(to_put_to_stack) > 0:
                if to_put_to_stack[-1]['PVT_head'] == PVT_head:
                    to_put_to_stack.pop()

    # print("num of iterations = {}, search space = {}, assign_to_provenance_num = {}".format(
    #     num_iterations, search_space, assign_to_provenance_num))
    return minimal_refinements


def searchPVT_contraction(PVT, PVT_head, numeric_attributes, categorical_attributes,
                          selection_numeric, selection_categorical, num_columns,
                          fairness_constraints_provenance_smaller_than,
                          full_PVT, full_PVT_head, max_index_PVT,
                          checked_assignments_satisfying, checked_assignments_not_satisfying, time_limit=5 * 60):
    time1 = time.time()
    global assign_to_provenance_num
    assign_to_provenance_num = 0
    PVT_stack = [PVT]
    PVT_head_stack = [PVT_head]
    max_index_PVT_stack = [max_index_PVT]
    parent_PVT_stack = [pd.DataFrame()]
    parent_PVT_head_stack = [[]]
    parent_max_index_PVT_stack = [pd.DataFrame()]
    col_idx_in_parent_PVT_stack = [0]
    idx_in_this_col_in_parent_PVT_stack = [0]
    find_relaxation = {x: [] for x in range(1, len(full_PVT_head) + 1)}
    fixed_value_assignments_stack = [{}]
    fixed_value_assignments_positions_stack = [{}]
    fixed_value_assignments_to_tighten_stack = [[]]
    left_side_binary_search_stack = [0]
    shifted_length_stack = [[0] * num_columns]
    to_put_to_stack = []
    minimal_refinements = []  # result set
    minimal_refinements_positions = []  # positions of result set
    fixed_value_assignments = {}
    fixed_value_assignments_positions = {}
    num_iterations = 0
    search_space = 0
    while PVT_stack:
        if time.time() - time1 > time_limit:
            print("provenance search alg time out")
            return minimal_refinements
        PVT = PVT_stack.pop()
        PVT_head = PVT_head_stack.pop()
        max_index_PVT = max_index_PVT_stack.pop()
        parent_PVT = parent_PVT_stack.pop()
        parent_PVT_head = parent_PVT_head_stack.pop()
        parent_max_index_PVT = parent_max_index_PVT_stack.pop()
        col_idx_in_parent_PVT = col_idx_in_parent_PVT_stack.pop()
        idx_in_this_col_in_parent_PVT = idx_in_this_col_in_parent_PVT_stack.pop()
        if idx_in_this_col_in_parent_PVT > 0:
            values_above = fixed_value_assignments_to_tighten_stack.pop()
        else:
            values_above = []
        fixed_value_assignments = fixed_value_assignments_stack.pop()
        fixed_value_assignments_positions = fixed_value_assignments_positions_stack.pop()
        shifted_length = shifted_length_stack.pop()
        find_bounding_relaxation = False
        num_columns = len(PVT_head)
        # print("==========================  searchPVT  ========================== ")
        # print("PVT_head: {}".format(PVT_head))
        # print("PVT:\n{}".format(PVT))
        # print("fixed_value_assignments: {}".format(fixed_value_assignments))
        # print("fixed_value_assignments_positions: {}".format(fixed_value_assignments_positions))
        # print("shifted_length: {}".format(shifted_length))
        # print("idx_in_this_col_in_parent_PVT:{}".format(idx_in_this_col_in_parent_PVT))

        satisfying_row_id = 0
        new_value_assignment = []
        last_satisfying_new_value_assignment = []
        full_value_assignment = {}
        last_satisfying_full_value_assignment = {}
        last_satisfying_bounding_relaxation_location = []
        left = left_side_binary_search_stack.pop()
        left = max(left, 0)
        # print("left = {}".format(left))
        right = max(max_index_PVT)
        # binary search can't use apply
        while left <= right:
            if time.time() - time1 > time_limit:
                print("provenance search alg time out")
                return minimal_refinements
            # print("left = {}, right={}".format(left, right))
            cur_row_id = int((right + left) / 2)
            new_bounding_relaxation_location = [cur_row_id if cur_row_id < x else x for x in max_index_PVT]
            new_value_assignment = [PVT.iloc[new_bounding_relaxation_location[x], x] for x in range(len(PVT_head))]
            full_value_assignment = dict(zip(PVT_head, new_value_assignment))
            full_value_assignment = {**full_value_assignment, **fixed_value_assignments}
            # print("value_assignment: ", full_value_assignment)
            full_value_assignment_str = num2string([full_value_assignment[k] for k in full_PVT_head])
            if full_value_assignment_str in checked_assignments_satisfying:
                # print("{} satisfies constraints".format(full_value_assignment))
                satisfying_row_id = cur_row_id
                right = cur_row_id - 1
                last_satisfying_full_value_assignment = full_value_assignment
                last_satisfying_new_value_assignment = new_value_assignment
                last_satisfying_bounding_relaxation_location = new_bounding_relaxation_location
                find_bounding_relaxation = True
            elif full_value_assignment_str in checked_assignments_not_satisfying:
                # print("{} doesn't satisfy constraints".format(full_value_assignment))
                left = cur_row_id + 1
            elif assign_to_provenance_contract_only(full_value_assignment, numeric_attributes, categorical_attributes,
                                                    selection_numeric, selection_categorical, full_PVT_head,
                                                    fairness_constraints_provenance_smaller_than):
                checked_assignments_satisfying.append(full_value_assignment_str)
                # print("{} satisfies constraints".format(full_value_assignment))
                satisfying_row_id = cur_row_id
                right = cur_row_id - 1
                last_satisfying_full_value_assignment = full_value_assignment
                last_satisfying_new_value_assignment = new_value_assignment
                last_satisfying_bounding_relaxation_location = new_bounding_relaxation_location
                find_bounding_relaxation = True
            else:
                # print("{} doesn't satisfy constraints".format(full_value_assignment))
                checked_assignments_not_satisfying.append(full_value_assignment_str)
                left = cur_row_id + 1

        col_idx = 0
        find_relaxation[num_columns].append(find_bounding_relaxation)
        if not find_bounding_relaxation:
            if len(PVT_head_stack) > 0:
                next_col_num_in_stack = len(PVT_head_stack[-1])
            else:
                next_col_num_in_stack = len(full_PVT_head)
            check_to_put_to_stack(to_put_to_stack, next_col_num_in_stack, num_columns, find_relaxation,
                                  PVT_stack, PVT_head_stack, max_index_PVT_stack, parent_PVT_stack,
                                  parent_PVT_head_stack, parent_max_index_PVT_stack, col_idx_in_parent_PVT_stack,
                                  idx_in_this_col_in_parent_PVT_stack,
                                  fixed_value_assignments_stack, fixed_value_assignments_positions_stack,
                                  fixed_value_assignments_to_tighten_stack, left_side_binary_search_stack,
                                  shifted_length_stack, idx_in_this_col_in_parent_PVT,
                                  PVT, PVT_head, max_index_PVT, parent_PVT, parent_PVT_head, parent_max_index_PVT,
                                  col_idx_in_parent_PVT, fixed_value_assignments, fixed_value_assignments_positions)
            continue

        full_value_assignment = last_satisfying_full_value_assignment
        new_value_assignment = last_satisfying_new_value_assignment
        if num_columns > 1:
            nan_row = PVT.iloc[satisfying_row_id].isnull()
            col_non_tightenable = -1
            if sum(k is False for k in nan_row) == 1:
                true_lst = np.where(nan_row)[0]
                range_lst = range(0, num_columns)
                col_non_tightenable = [x for x in range_lst if x not in true_lst][0]
                # print("col {} doesn't need to be tightened".format(col_non_tightenable))

            tmp_max_idx_of_ol = 0

            def tighten_result(column):
                nonlocal col_idx
                nonlocal last_satisfying_full_value_assignment
                nonlocal tmp_max_idx_of_ol
                idx_in_this_col = last_satisfying_bounding_relaxation_location[col_idx]
                if col_idx == col_non_tightenable:
                    col_idx += 1
                    tmp_max_idx_of_ol = max(tmp_max_idx_of_ol, idx_in_this_col)
                    return
                while idx_in_this_col > 0:
                    idx_in_this_col -= 1
                    new_value_assignment[col_idx] = column[idx_in_this_col]
                    full_value_assignment[PVT_head[col_idx]] = column[idx_in_this_col]
                    # print("value_assignment: ", full_value_assignment)
                    full_value_assignment_str = num2string([full_value_assignment[k] for k in full_PVT_head])
                    if full_value_assignment_str in checked_assignments_satisfying:
                        # print("{} satisfies constraints".format(full_value_assignment))
                        last_satisfying_full_value_assignment = full_value_assignment
                        last_satisfying_bounding_relaxation_location[col_idx] = idx_in_this_col
                        smallest_row = idx_in_this_col
                    elif full_value_assignment_str in checked_assignments_not_satisfying:
                        # print("{} doesn't satisfy constraints".format(full_value_assignment))
                        smallest_row = idx_in_this_col + 1
                        # last_satisfying_bounding_relaxation_location[col_idx] = smallest_row
                        new_value_assignment[col_idx] = column[smallest_row]
                        full_value_assignment[PVT_head[col_idx]] = column[smallest_row]
                        break
                    elif assign_to_provenance_contract_only(full_value_assignment, numeric_attributes,
                                                            categorical_attributes, selection_numeric,
                                                            selection_categorical, full_PVT_head,
                                                            fairness_constraints_provenance_smaller_than):
                        checked_assignments_satisfying.append(full_value_assignment_str)
                        # print("{} satisfies constraints".format(full_value_assignment))
                        last_satisfying_full_value_assignment = full_value_assignment
                        last_satisfying_bounding_relaxation_location[col_idx] = idx_in_this_col
                        smallest_row = idx_in_this_col
                    else:
                        # print("{} doesn't satisfy constraints".format(full_value_assignment))
                        checked_assignments_not_satisfying.append(full_value_assignment_str)
                        smallest_row = idx_in_this_col + 1
                        # last_satisfying_bounding_relaxation_location[col_idx] = smallest_row
                        new_value_assignment[col_idx] = column[smallest_row]
                        full_value_assignment[PVT_head[col_idx]] = column[smallest_row]
                        break

                col_idx += 1
                return

            PVT.apply(tighten_result, axis=0)
            # print("tight relaxation, new_value_assignment: {}".format(new_value_assignment))

        # optimization: tighten the last fixed column
        tight_value_idx = -1
        fixed_att = str()
        if idx_in_this_col_in_parent_PVT > 0:
            tight_success = False
            # binary search to tighten this column
            left = 0
            right = len(values_above) - 1
            fixed_att = list(fixed_value_assignments.keys())[-1]
            # print("tighten the last fixed column {}:\n {}".format(fixed_att, values_above))
            fixed_value_assignments_for_tighten = copy.deepcopy(fixed_value_assignments)
            while left <= right:
                cur_value_id = int((right + left) / 2)
                cur_fixed_value = values_above[cur_value_id]
                fixed_value_assignments_for_tighten[fixed_att] = cur_fixed_value
                full_value_assignment = {**dict(zip(PVT_head, new_value_assignment)),
                                         **fixed_value_assignments_for_tighten}
                # print("value_assignment: ", full_value_assignment)
                full_value_assignment_str = num2string([full_value_assignment[k] for k in full_PVT_head])
                if full_value_assignment_str in checked_assignments_satisfying:
                    # print("{} satisfies constraints".format(full_value_assignment))
                    right = cur_value_id - 1
                    tight_value_idx = cur_value_id
                elif full_value_assignment_str in checked_assignments_not_satisfying:
                    # print("{} doesn't satisfy constraints".format(full_value_assignment))
                    left = cur_value_id + 1
                elif assign_to_provenance_contract_only(full_value_assignment, numeric_attributes,
                                                        categorical_attributes, selection_numeric,
                                                        selection_categorical, full_PVT_head,
                                                        fairness_constraints_provenance_smaller_than):
                    checked_assignments_satisfying.append(full_value_assignment_str)
                    # print("{} satisfies constraints".format(full_value_assignment))
                    right = cur_value_id - 1
                    tight_value_idx = cur_value_id
                else:
                    # print("{} doesn't satisfy constraints".format(full_value_assignment))
                    checked_assignments_not_satisfying.append(full_value_assignment_str)
                    left = cur_value_id + 1
            if tight_value_idx >= 0:  # can be tightened
                if tight_value_idx < idx_in_this_col_in_parent_PVT:
                    # tight this fixed column successfully
                    # last_satisfying_bounding_relaxation_location[PVT_head.index(fixed_att)] = tight_value_idx
                    fixed_value_assignments_for_tighten[fixed_att] = values_above[tight_value_idx]
                    full_value_assignment[fixed_att] = values_above[tight_value_idx]
                    tight_success = True
                fva = [full_value_assignment[k] for k in full_PVT_head]
                full_value_assignment_positions = dict(zip(PVT_head, last_satisfying_bounding_relaxation_location))
                full_value_assignment_positions = {**full_value_assignment_positions,
                                                   **fixed_value_assignments_positions, fixed_att: tight_value_idx}
            else:
                full_value_assignment = {**dict(zip(PVT_head, new_value_assignment)),
                                         **fixed_value_assignments}
                fva = [full_value_assignment[k] for k in full_PVT_head]
                full_value_assignment_positions = dict(zip(PVT_head, last_satisfying_bounding_relaxation_location))
                full_value_assignment_positions = {**full_value_assignment_positions,
                                                   **fixed_value_assignments_positions}
                # if tight success, when doing recursion, fixed attribute should use the tightened value
                # and if tight success, there is no need to traverse between original position fixed and the tightened one
            minimal_refinements, minimal_refinements_positions, added = \
                update_minimal_relaxation_and_position(minimal_refinements, minimal_refinements_positions,
                                                       fva, [full_value_assignment_positions[x] for x in full_PVT_head],
                                                       shifted_length)
            # print("find base refinement {}".format(new_value_assignment))
            # print("position: {}".format(full_value_assignment_positions))
            # print("minimal_refinements: {}".format(minimal_refinements))
            # print("num of minimal_refinements: {}".format(len(minimal_refinements)))

            if tight_success and added:
                # in the following situation, we need to traverse the values between tightened and original value
                if not all(v == 0 for v in last_satisfying_bounding_relaxation_location):
                    for j in range(idx_in_this_col_in_parent_PVT - 1, tight_value_idx, -1):
                        to_put = copy.deepcopy(to_put_to_stack[-1])
                        to_put['idx_in_this_col_in_parent_PVT'] = j
                        to_put['fixed_value_assignments'][fixed_att] = values_above[j]
                        to_put['fixed_value_assignments_to_tighten'] = values_above[:j]
                        to_put['fixed_value_assignments_positions'][fixed_att] = j
                        PVT_stack.append(to_put['PVT'])
                        PVT_head_stack.append(to_put['PVT_head'])
                        max_index_PVT_stack.append(to_put['max_index_PVT'])
                        parent_PVT_stack.append(to_put['parent_PVT'])
                        parent_PVT_head_stack.append(to_put['parent_PVT_head'])
                        parent_max_index_PVT_stack.append(to_put['parent_max_index_PVT'])
                        col_idx_in_parent_PVT_stack.append(to_put['col_idx_in_parent_PVT'])
                        idx_in_this_col_in_parent_PVT_stack.append(to_put['idx_in_this_col_in_parent_PVT'])
                        fixed_value_assignments_stack.append(to_put['fixed_value_assignments'])
                        fixed_value_assignments_positions_stack.append(to_put['fixed_value_assignments_positions'])
                        left_side_binary_search_stack.append(to_put['for_left_binary'])
                        shifted_length_stack.append(to_put['shifted_length'])
                        fixed_value_assignments_to_tighten_stack.append(to_put['fixed_value_assignments_to_tighten'])
                fixed_value_assignments = fixed_value_assignments_for_tighten
                fixed_value_assignments_positions[fixed_att] = tight_value_idx
                if tight_value_idx == 0 and len(to_put_to_stack) > 0:
                    to_put_to_stack.pop()
                else:
                    #  no need to test values between now and last fixed value
                    if len(to_put_to_stack) > 0:
                        to_put_to_stack[-1]['idx_in_this_col_in_parent_PVT'] = tight_value_idx - 1
                        to_put_to_stack[-1]['fixed_value_assignments'][fixed_att] = values_above[tight_value_idx - 1]
                        to_put_to_stack[-1]['fixed_value_assignments_to_tighten'] = values_above[: tight_value_idx - 1]
                        to_put_to_stack[-1]['fixed_value_assignments_positions'][fixed_att] = tight_value_idx - 1
        else:
            fva = [full_value_assignment[k] for k in full_PVT_head]
            full_value_assignment_positions = dict(zip(PVT_head, last_satisfying_bounding_relaxation_location))
            full_value_assignment_positions = {**full_value_assignment_positions, **fixed_value_assignments_positions}

            minimal_refinements, minimal_refinements_positions, _ = \
                update_minimal_relaxation_and_position(minimal_refinements, minimal_refinements_positions,
                                                       fva, [full_value_assignment_positions[x] for x in full_PVT_head],
                                                       shifted_length)
            # print("find base refinement {}".format(new_value_assignment))
            # print("position: {}".format(full_value_assignment_positions))
            # print("minimal_refinements: {}".format(minimal_refinements))
            # print("num of minimal_refinements: {}".format(len(minimal_refinements)))

        if num_columns == 1:
            if len(PVT_head_stack) > 0:
                next_col_num_in_stack = len(PVT_head_stack[-1])
            else:
                next_col_num_in_stack = len(full_PVT_head)
            check_to_put_to_stack(to_put_to_stack, next_col_num_in_stack, num_columns, find_relaxation,
                                  PVT_stack, PVT_head_stack, max_index_PVT_stack, parent_PVT_stack,
                                  parent_PVT_head_stack, parent_max_index_PVT_stack, col_idx_in_parent_PVT_stack,
                                  idx_in_this_col_in_parent_PVT_stack,
                                  fixed_value_assignments_stack, fixed_value_assignments_positions_stack,
                                  fixed_value_assignments_to_tighten_stack, left_side_binary_search_stack,
                                  shifted_length_stack, idx_in_this_col_in_parent_PVT,
                                  PVT, PVT_head, max_index_PVT, parent_PVT, parent_PVT_head, parent_max_index_PVT,
                                  col_idx_in_parent_PVT, fixed_value_assignments, fixed_value_assignments_positions)
            continue
        # recursion
        col_idx = 0
        if len(PVT_head_stack) > 0:
            next_col_num_in_stack = len(PVT_head_stack[-1])
        else:
            next_col_num_in_stack = len(full_PVT_head)
        check_to_put_to_stack(to_put_to_stack, next_col_num_in_stack, num_columns, find_relaxation,
                              PVT_stack, PVT_head_stack, max_index_PVT_stack, parent_PVT_stack, parent_PVT_head_stack,
                              parent_max_index_PVT_stack, col_idx_in_parent_PVT_stack,
                              idx_in_this_col_in_parent_PVT_stack, fixed_value_assignments_stack,
                              fixed_value_assignments_positions_stack, fixed_value_assignments_to_tighten_stack,
                              left_side_binary_search_stack, shifted_length_stack, idx_in_this_col_in_parent_PVT,
                              PVT, PVT_head, max_index_PVT, parent_PVT, parent_PVT_head, parent_max_index_PVT,
                              col_idx_in_parent_PVT, fixed_value_assignments, fixed_value_assignments_positions)
        index_to_insert_to_stack = len(PVT_stack)
        insert_idx_fixed_value_assignments_to_tighten_stack = len(fixed_value_assignments_to_tighten_stack)
        index_to_insert_to_put = len(to_put_to_stack)
        original_max_index_PVT = max_index_PVT.copy()
        original_shifted_length = copy.deepcopy(shifted_length)

        no_recursion_to_do = True

        def recursion(column):
            nonlocal col_idx
            nonlocal new_value_assignment
            nonlocal last_satisfying_bounding_relaxation_location
            nonlocal shifted_length
            nonlocal no_recursion_to_do
            idx_in_this_col = last_satisfying_bounding_relaxation_location[col_idx]
            # optimization: if there are no other columns to be moved down, return
            if idx_in_this_col == 0:
                col_idx += 1
                return
            if sum(last_satisfying_bounding_relaxation_location[i] < original_max_index_PVT[i] for i in
                   range(len(PVT_head)) if
                   i != col_idx) == 0:
                col_idx += 1
                return
            idx_in_this_col -= 1
            # optimization: fixing this value doesn't dissatisfy inequalities
            one_more_fix = copy.deepcopy(fixed_value_assignments)
            one_more_fix[PVT_head[col_idx]] = column[idx_in_this_col]
            fixed_value_assignments_for_stack = copy.deepcopy(fixed_value_assignments)
            fixed_value_assignments_for_stack[PVT_head[col_idx]] = column[idx_in_this_col]
            fixed_value_assignments_positions_for_stack = copy.deepcopy(fixed_value_assignments_positions)
            fixed_value_assignments_positions_for_stack[PVT_head[col_idx]] = idx_in_this_col

            new_PVT_head = [PVT_head[x] for x in range(len(PVT_head)) if x != col_idx]
            new_max_index_PVT = max_index_PVT[:col_idx] + max_index_PVT[col_idx + 1:]
            # optimization: if there is only one column left to be moved down,
            #  this column in the new recursion should start from where it stopped before
            if len(new_PVT_head) == 1:
                if col_idx == 0:
                    PVT_for_recursion = PVT[new_PVT_head].iloc[
                                        last_satisfying_bounding_relaxation_location[1] + 1:
                                        max(new_max_index_PVT) + 1].reset_index(drop=True)
                    shifted_length[full_PVT_head.index(PVT_head[1])] += \
                        last_satisfying_bounding_relaxation_location[1] + 1
                    new_max_index_PVT = [len(PVT_for_recursion) - 1]
                else:
                    PVT_for_recursion = PVT[new_PVT_head].iloc[: new_max_index_PVT[0] + 1].reset_index(drop=True)
                    # shifted_length[full_PVT_head.index(PVT_head[1])] -= \
                    #     last_satisfying_bounding_relaxation_location[1] + 1
                    shifted_length = original_shifted_length
            else:
                PVT_for_recursion = PVT[new_PVT_head].head(max(new_max_index_PVT) + 1)
                shifted_length = original_shifted_length
            PVT_stack.insert(index_to_insert_to_stack, PVT_for_recursion)
            PVT_head_stack.insert(index_to_insert_to_stack, new_PVT_head)
            max_index_PVT_stack.insert(index_to_insert_to_stack, new_max_index_PVT)
            parent_PVT_stack.insert(index_to_insert_to_stack, PVT.copy())
            parent_PVT_head_stack.insert(index_to_insert_to_stack, PVT_head)
            parent_max_index_PVT_stack.insert(index_to_insert_to_stack, max_index_PVT)
            col_idx_in_parent_PVT_stack.insert(index_to_insert_to_stack, col_idx)
            idx_in_this_col_in_parent_PVT_stack.insert(index_to_insert_to_stack, idx_in_this_col)
            fixed_value_assignments_stack.insert(index_to_insert_to_stack, fixed_value_assignments_for_stack)
            fixed_value_assignments_positions_stack.insert(index_to_insert_to_stack,
                                                           fixed_value_assignments_positions_for_stack)
            before_shift = last_satisfying_bounding_relaxation_location[:col_idx] + \
                           last_satisfying_bounding_relaxation_location[col_idx + 1:]
            shift_for_col = [shifted_length[PVT_head.index(att)] for att in PVT_head]
            shift_len = shift_for_col[:col_idx] + shift_for_col[col_idx + 1:]
            after_shift = [before_shift[i] - shift_len[i] for i in range(num_columns - 1)]
            for_left_binary = max(after_shift)
            left_side_binary_search_stack.insert(index_to_insert_to_stack, for_left_binary)
            shifted_length_stack.insert(index_to_insert_to_stack, copy.deepcopy(shifted_length))
            if idx_in_this_col > 0:
                fixed_value_assignments_to_tighten_stack.insert(insert_idx_fixed_value_assignments_to_tighten_stack,
                                                                column[:idx_in_this_col].copy())
                # to_put_to_stack
                to_put = dict()
                to_put['PVT'] = PVT_for_recursion.copy()
                to_put['PVT_head'] = new_PVT_head.copy()
                to_put['max_index_PVT'] = new_max_index_PVT.copy()
                to_put['parent_PVT'] = PVT.copy()
                to_put['parent_PVT_head'] = PVT_head.copy()
                to_put['parent_max_index_PVT'] = max_index_PVT.copy()
                to_put['col_idx_in_parent_PVT'] = col_idx
                to_put['idx_in_this_col_in_parent_PVT'] = idx_in_this_col - 1
                fixed_value_assignments_to_put = copy.deepcopy(fixed_value_assignments_for_stack)
                fixed_value_assignments_to_put[PVT_head[col_idx]] = column[idx_in_this_col - 1]
                to_put['fixed_value_assignments'] = fixed_value_assignments_to_put
                fixed_value_assignments_positions_to_put = copy.deepcopy(fixed_value_assignments_positions_for_stack)
                fixed_value_assignments_positions_to_put[PVT_head[col_idx]] = idx_in_this_col - 1
                to_put['fixed_value_assignments_positions'] = fixed_value_assignments_positions_to_put
                if to_put['idx_in_this_col_in_parent_PVT'] > 0:
                    to_put['fixed_value_assignments_to_tighten'] = column[:idx_in_this_col - 1].copy()
                to_put['for_left_binary'] = for_left_binary
                to_put['shifted_length'] = copy.deepcopy(shifted_length)
                to_put_to_stack.insert(index_to_insert_to_put, to_put)

            # avoid repeated checking: for columns that are done with moving up,
            #  we need to remove values above the 'stop line'
            seri = PVT[PVT_head[col_idx]]
            PVT[PVT_head[col_idx]] = seri.shift(periods=-last_satisfying_bounding_relaxation_location[col_idx])
            max_index_PVT[col_idx] -= last_satisfying_bounding_relaxation_location[col_idx]
            original_shifted_length[full_PVT_head.index(PVT_head[col_idx])] += \
                last_satisfying_bounding_relaxation_location[col_idx]
            col_idx += 1
            no_recursion_to_do = False
            return

        PVT.apply(recursion, axis=0)
        if no_recursion_to_do and tight_value_idx > 0:
            # print("no_recursion_to_do")
            if len(PVT_head_stack) > 0:
                if PVT_head == PVT_head_stack[-1]:
                    fixed_value_assignments_positions_stack[-1][fixed_att] = tight_value_idx - 1
                    fixed_value_assignments_to_tighten_stack[-1] = values_above[: tight_value_idx - 1]
                    fixed_value_assignments_stack[-1][fixed_att] = values_above[tight_value_idx - 1]
                    idx_in_this_col_in_parent_PVT_stack[-1] = tight_value_idx - 1
            if len(to_put_to_stack) > 0:
                if to_put_to_stack[-1]['PVT_head'] == PVT_head:
                    to_put_to_stack.pop()

    return minimal_refinements


def searchPVT_refinement(PVT, PVT_head, possible_values_lists, numeric_attributes, categorical_attributes,
                         selection_numeric, selection_categorical, num_columns,
                         fairness_constraints_provenance_greater_than,
                         fairness_constraints_provenance_smaller_than, fairness_constraints_provenance_complex,
                         full_PVT, full_PVT_head, max_index_PVT,
                         checked_assignments_satisfying, checked_assignments_not_satisfying, time_limit=5 * 60):
    time1 = time.time()
    global assign_to_provenance_num
    assign_to_provenance_num = 0
    PVT_stack = [PVT]
    PVT_head_stack = [PVT_head]
    max_index_PVT_stack = [max_index_PVT]
    parent_PVT_stack = [pd.DataFrame()]
    parent_PVT_head_stack = [[]]
    parent_max_index_PVT_stack = [pd.DataFrame()]
    col_idx_in_parent_PVT_stack = [0]
    idx_in_this_col_in_parent_PVT_stack = [0]
    find_relaxation = {x: [] for x in range(1, len(full_PVT_head) + 1)}
    fixed_value_assignments_stack = [{}]
    fixed_value_assignments_positions_stack = [{}]
    fixed_value_assignments_to_tighten_stack = [[]]
    left_side_binary_search_stack = [0]
    shifted_length_stack = [[0] * num_columns]
    to_put_to_stack = []
    minimal_refinements = []  # result set
    minimal_refinements_positions = []  # positions of result set
    fixed_value_assignments = {}
    fixed_value_assignments_positions = {}
    num_iterations = 0
    initial_PVT = PVT.copy()
    while PVT_stack:
        if time.time() - time1 > time_limit:
            print("provenance search alg time out")
            return minimal_refinements
        PVT = PVT_stack.pop()
        num_iterations += 1
        PVT_head = PVT_head_stack.pop()
        max_index_PVT = max_index_PVT_stack.pop()
        parent_PVT = parent_PVT_stack.pop()
        parent_PVT_head = parent_PVT_head_stack.pop()
        parent_max_index_PVT = parent_max_index_PVT_stack.pop()
        col_idx_in_parent_PVT = col_idx_in_parent_PVT_stack.pop()
        idx_in_this_col_in_parent_PVT = idx_in_this_col_in_parent_PVT_stack.pop()
        if idx_in_this_col_in_parent_PVT > 0:
            values_above = fixed_value_assignments_to_tighten_stack.pop()
        else:
            values_above = []
        fixed_value_assignments = fixed_value_assignments_stack.pop()
        fixed_value_assignments_positions = fixed_value_assignments_positions_stack.pop()
        shifted_length = shifted_length_stack.pop()
        num_columns = len(PVT_head)
        fixed_attributes = list(fixed_value_assignments.keys())
        # print("==========================  searchPVT  ========================== ")
        # print("PVT_head: {}".format(PVT_head))
        # print("PVT:\n{}".format(PVT))
        # print("fixed_value_assignments: {}".format(fixed_value_assignments))
        # print("fixed_value_assignments_positions: {}".format(fixed_value_assignments_positions))
        # print("shifted_length: {}".format(shifted_length))
        # print("idx_in_this_col_in_parent_PVT:{}".format(idx_in_this_col_in_parent_PVT))
        # if fixed_value_assignments == {'COW__2': 0.0, 'COW__6': 0.0, 'COW__5': 0.0, 'COW__7': 0.0}:
        #     print("debug")
        new_value_assignment = {}
        full_value_assignment = {}
        last_satisfying_bounding_relaxation_location = []
        new_value_assignment_position = [-1] * num_columns
        att_idx = 0
        find_base_refinement = False
        while att_idx < num_columns and att_idx >= 0:
            if time.time() - time1 > time_limit:
                print("provenance search alg time out")
                return minimal_refinements
            col = PVT_head[att_idx]
            find_value_this_col = False
            idx_in_col = 0
            full_att = fixed_attributes + PVT_head[:att_idx + 1]
            new_value_assignment_position[att_idx] += 1
            for idx_in_col in range(new_value_assignment_position[att_idx], max_index_PVT[att_idx] + 1):
                new_value_assignment[col] = PVT.loc[idx_in_col, col]
                full_value_assignment = {**new_value_assignment, **fixed_value_assignments}
                if att_idx + 1 == num_columns:
                    # print(full_value_assignment)
                    # if full_value_assignment['COW__1'] == 0.0 and full_value_assignment['COW__3'] == 1 and \
                    #         full_value_assignment['COW__4'] == 1 and full_value_assignment['COW__8'] == 0:
                    #     print("debug")
                    assign, reason = assign_to_provenance(full_value_assignment, numeric_attributes,
                                                          categorical_attributes, selection_numeric,
                                                          selection_categorical,
                                                          full_PVT_head, fairness_constraints_provenance_greater_than,
                                                          fairness_constraints_provenance_smaller_than,
                                                          fairness_constraints_provenance_complex)
                    if assign:
                        # print("{} satisfies constraints".format(new_value_assignment))
                        last_satisfying_bounding_relaxation_location = new_value_assignment_position
                        find_base_refinement = True
                        find_value_this_col = True
                        break
                else:
                    if assign_to_provenance_relax_only_partial_query(full_value_assignment, numeric_attributes,
                                                                     categorical_attributes,
                                                                     selection_numeric, selection_categorical,
                                                                     full_PVT_head,
                                                                     fairness_constraints_provenance_greater_than):
                        # print("{} satisfies constraints".format(new_value_assignment))
                        last_satisfying_bounding_relaxation_location = new_value_assignment_position
                        find_value_this_col = True
                        break
                idx_in_col += 1
            if find_value_this_col:
                new_value_assignment_position[att_idx] = idx_in_col
                att_idx += 1
            else:
                new_value_assignment_position[att_idx] = -1
                del new_value_assignment[col]
                att_idx -= 1
        if not find_base_refinement:
            if len(PVT_head_stack) > 0:
                next_col_num_in_stack = len(PVT_head_stack[-1])
            else:
                next_col_num_in_stack = len(full_PVT_head)
            check_to_put_to_stack_refinement(to_put_to_stack, next_col_num_in_stack, num_columns, find_relaxation,
                                             PVT_stack, PVT_head_stack, max_index_PVT_stack, parent_PVT_stack,
                                             parent_PVT_head_stack,
                                             parent_max_index_PVT_stack, col_idx_in_parent_PVT_stack,
                                             idx_in_this_col_in_parent_PVT_stack,
                                             fixed_value_assignments_stack, fixed_value_assignments_positions_stack,
                                             fixed_value_assignments_to_tighten_stack, left_side_binary_search_stack,
                                             shifted_length_stack,
                                             idx_in_this_col_in_parent_PVT,
                                             PVT, PVT_head, max_index_PVT, parent_PVT, parent_PVT_head,
                                             parent_max_index_PVT,
                                             col_idx_in_parent_PVT, fixed_value_assignments,
                                             fixed_value_assignments_positions)
            # print("no base refinement here, size of PVT: {}*{}".format(len(PVT), len(PVT_head)))
            continue

        # print("find base refinement {}".format(new_value_assignment))
        # print("position: {}".format(new_value_assignment_position))
        tight_success = False
        tight_value_idx = -1
        fixed_att = str()

        def recursion(column):
            nonlocal col_idx
            nonlocal new_value_assignment
            nonlocal last_satisfying_bounding_relaxation_location
            nonlocal shifted_length
            nonlocal col_name_to_drop
            nonlocal recursion_end
            idx_in_this_col = last_satisfying_bounding_relaxation_location[col_idx]
            # optimization: if there are no other columns to be moved down, return
            if idx_in_this_col == 0:
                col_idx += 1
                return
            if sum(last_satisfying_bounding_relaxation_location[i] < original_max_index_PVT[i] for i in
                   range(original_PVT_col_num) if
                   i != col_idx) == 0:
                col_idx += 1
                return
            idx_in_this_col -= 1
            found = False
            col_name = original_PVT_head[col_idx]
            one_more_fix = copy.deepcopy(fixed_value_assignments)
            while idx_in_this_col >= 0:
                # optimization: fixing this value doesn't dissatisfy inequalities
                one_more_fix[col_name] = column[idx_in_this_col]

                if not assign_to_provenance_relax_only_partial_query(one_more_fix, numeric_attributes,
                                                                     categorical_attributes,
                                                                     selection_numeric, selection_categorical,
                                                                     full_PVT_head,
                                                                     fairness_constraints_provenance_greater_than):
                    # print("fixing {} = {} dissatisfies constraints".format(PVT_head[col_idx], column[idx_in_this_col]))
                    idx_in_this_col -= 1
                else:
                    found = True
                    break
            if not found:
                col_idx += 1
                return
            fixed_value_assignments_for_stack = copy.deepcopy(fixed_value_assignments)
            fixed_value_assignments_for_stack[col_name] = column[idx_in_this_col]
            fixed_value_assignments_positions_for_stack = copy.deepcopy(fixed_value_assignments_positions)
            fixed_value_assignments_positions_for_stack[col_name] = idx_in_this_col

            new_PVT_head = [original_PVT_head[x] for x in range(original_PVT_col_num) if x != col_idx and
                            original_PVT_head[x] in PVT_head]
            # new_max_index_PVT = max_index_PVT[:col_idx] + max_index_PVT[col_idx + 1:]
            new_max_index_PVT = [max_index_PVT[x] for x in range(len(max_index_PVT)) if x != col_idx and
                                 x not in col_idx_to_drop]
            # optimization: if there is only one column left to be moved down,
            #  this column in the new recursion should start from where it stopped before
            if len(new_PVT_head) == 1:  # FIXME
                if col_idx == 0:
                    PVT_for_recursion = PVT[new_PVT_head].iloc[
                                        last_satisfying_bounding_relaxation_location[1] + 1:
                                        max(new_max_index_PVT) + 1].reset_index(drop=True)
                    shifted_length[full_PVT_head.index(PVT_head[1])] += \
                        last_satisfying_bounding_relaxation_location[1] + 1
                    new_max_index_PVT = [len(PVT_for_recursion) - 1]
                else:
                    PVT_for_recursion = PVT[new_PVT_head].iloc[: new_max_index_PVT[0] + 1].reset_index(drop=True)
                    # shifted_length[full_PVT_head.index(PVT_head[1])] -= \
                    #     last_satisfying_bounding_relaxation_location[1] + 1
                    shifted_length = original_shifted_length
            else:
                PVT_for_recursion = PVT[new_PVT_head].head(max(new_max_index_PVT) + 1)
                shifted_length = original_shifted_length
            recursion_end = False
            PVT_stack.insert(index_to_insert_to_stack, PVT_for_recursion)
            PVT_head_stack.insert(index_to_insert_to_stack, new_PVT_head)
            max_index_PVT_stack.insert(index_to_insert_to_stack, new_max_index_PVT)
            parent_PVT_stack.insert(index_to_insert_to_stack, PVT.copy())
            parent_PVT_head_stack.insert(index_to_insert_to_stack, original_PVT_head)
            parent_max_index_PVT_stack.insert(index_to_insert_to_stack, max_index_PVT)
            col_idx_in_parent_PVT_stack.insert(index_to_insert_to_stack, col_idx)
            idx_in_this_col_in_parent_PVT_stack.insert(index_to_insert_to_stack, idx_in_this_col)
            fixed_value_assignments_stack.insert(index_to_insert_to_stack, fixed_value_assignments_for_stack)
            fixed_value_assignments_positions_stack.insert(index_to_insert_to_stack,
                                                           fixed_value_assignments_positions_for_stack)
            before_shift = last_satisfying_bounding_relaxation_location[:col_idx] + \
                           last_satisfying_bounding_relaxation_location[col_idx + 1:]
            shift_for_col = [shifted_length[original_PVT_head.index(att)] for att in original_PVT_head]
            shift_len = shift_for_col[:col_idx] + shift_for_col[col_idx + 1:]
            after_shift = [before_shift[i] - shift_len[i] for i in range(num_columns - 1)]
            for_left_binary = max(after_shift)
            left_side_binary_search_stack.insert(index_to_insert_to_stack, for_left_binary)
            shifted_length_stack.insert(index_to_insert_to_stack, copy.deepcopy(shifted_length))
            if idx_in_this_col > 0:
                fixed_value_assignments_to_tighten_stack.insert(insert_idx_fixed_value_assignments_to_tighten_stack,
                                                                column[:idx_in_this_col].copy())
                # to_put_to_stack
                to_put = dict()
                to_put['PVT'] = PVT_for_recursion.copy()
                to_put['PVT_head'] = new_PVT_head.copy()
                to_put['max_index_PVT'] = new_max_index_PVT.copy()
                to_put['parent_PVT'] = PVT.copy()
                to_put['parent_PVT_head'] = original_PVT_head.copy()
                to_put['parent_max_index_PVT'] = max_index_PVT.copy()
                to_put['col_idx_in_parent_PVT'] = col_idx
                to_put['idx_in_this_col_in_parent_PVT'] = idx_in_this_col - 1
                fixed_value_assignments_to_put = copy.deepcopy(fixed_value_assignments_for_stack)
                fixed_value_assignments_to_put[col_name] = column[idx_in_this_col - 1]
                to_put['fixed_value_assignments'] = fixed_value_assignments_to_put
                fixed_value_assignments_positions_to_put = copy.deepcopy(fixed_value_assignments_positions_for_stack)
                fixed_value_assignments_positions_to_put[col_name] = idx_in_this_col - 1
                to_put['fixed_value_assignments_positions'] = fixed_value_assignments_positions_to_put
                if to_put['idx_in_this_col_in_parent_PVT'] > 0:
                    to_put['fixed_value_assignments_to_tighten'] = column[:idx_in_this_col - 1].copy()
                to_put['for_left_binary'] = for_left_binary
                to_put['shifted_length'] = copy.deepcopy(shifted_length)
                to_put_to_stack.insert(index_to_insert_to_put, to_put)

            seri = PVT[col_name]
            newcolumn = seri.shift(periods=-last_satisfying_bounding_relaxation_location[col_idx])
            PVT[col_name] = newcolumn
            max_index_PVT[col_idx] -= last_satisfying_bounding_relaxation_location[col_idx]
            original_shifted_length[full_PVT_head.index(col_name)] += \
                last_satisfying_bounding_relaxation_location[col_idx]
            if newcolumn.notna().sum() == 1:  # if a column only has one value left, add it to fixed values
                fixed_value_assignments[col_name] = newcolumn[0]
                fixed_value_assignments_positions[col_name] = idx_in_this_col + 1
                col_idx_to_drop.append(col_idx)
                col_name_to_drop.append(col_name)
                original_shifted_length[full_PVT_head.index(col_name)] = 0  # FIXME
                PVT_head.remove(col_name)
            col_idx += 1
            return

        # optimization: tighten the last fixed column
        if idx_in_this_col_in_parent_PVT > 0:
            left = 0
            fixed_att = list(fixed_value_assignments.keys())[-1]
            right = len(values_above) - 1
            fixed_value_assignments_for_tighten = copy.deepcopy(fixed_value_assignments)
            # if len(fairness_constraints_provenance_complex) > 0:  #  can't use binary
            while left <= right:
                cur_value_id = left
                cur_fixed_value = values_above[cur_value_id]
                fixed_value_assignments_for_tighten[fixed_att] = cur_fixed_value
                full_value_assignment = {**new_value_assignment, **fixed_value_assignments_for_tighten}
                # print("value_assignment: ", full_value_assignment)
                assign, reason = assign_to_provenance(full_value_assignment, numeric_attributes,
                                                      categorical_attributes, selection_numeric,
                                                      selection_categorical, full_PVT_head,
                                                      fairness_constraints_provenance_greater_than,
                                                      fairness_constraints_provenance_smaller_than,
                                                      fairness_constraints_provenance_complex)
                if assign:
                    # print("{} satisfies constraints".format(full_value_assignment))
                    tight_value_idx = cur_value_id
                    break
                else:
                    # print("{} doesn't satisfy constraints".format(full_value_assignment))
                    left += 1
            if tight_value_idx >= 0:  # can be tightened
                if tight_value_idx < idx_in_this_col_in_parent_PVT:
                    # tight this fixed column successfully
                    fixed_value_assignments_for_tighten[fixed_att] = values_above[tight_value_idx]
                    full_value_assignment[fixed_att] = values_above[tight_value_idx]
                    tight_success = True
                fva = [full_value_assignment[k] for k in full_PVT_head]
                full_value_assignment_positions = dict(zip(PVT_head, last_satisfying_bounding_relaxation_location))
                full_value_assignment_positions = {**full_value_assignment_positions,
                                                   **fixed_value_assignments_positions, fixed_att: tight_value_idx}
            else:
                full_value_assignment = {**new_value_assignment, **fixed_value_assignments}
                fva = [full_value_assignment[k] for k in full_PVT_head]
                full_value_assignment_positions = dict(zip(PVT_head, last_satisfying_bounding_relaxation_location))
                full_value_assignment_positions = {**full_value_assignment_positions,
                                                   **fixed_value_assignments_positions}
            find_relaxation[num_columns].append(find_base_refinement)

            minimal_refinements, minimal_refinements_positions, added = \
                update_minimal_relaxation_and_position_refinement(minimal_refinements, minimal_refinements_positions,
                                                                  fva, [full_value_assignment_positions[x] for x in
                                                                        full_PVT_head],
                                                                  shifted_length, initial_PVT, selection_numeric,
                                                                  full_PVT_head)
            # if tight success, when doing recursion, fixed attribute should use the tightened value
            if tight_success and added:
                # in the following situation, we need to traverse the values between tightened and original value
                if not all(v == 0 for v in last_satisfying_bounding_relaxation_location):
                    for j in range(idx_in_this_col_in_parent_PVT - 1, tight_value_idx, -1):
                        to_put = copy.deepcopy(to_put_to_stack[-1])
                        to_put['idx_in_this_col_in_parent_PVT'] = j
                        to_put['fixed_value_assignments'][fixed_att] = values_above[j]
                        to_put['fixed_value_assignments_to_tighten'] = values_above[:j]
                        to_put['fixed_value_assignments_positions'][fixed_att] = j
                        PVT_stack.append(to_put['PVT'])
                        PVT_head_stack.append(to_put['PVT_head'])
                        max_index_PVT_stack.append(to_put['max_index_PVT'])
                        parent_PVT_stack.append(to_put['parent_PVT'])
                        parent_PVT_head_stack.append(to_put['parent_PVT_head'])
                        parent_max_index_PVT_stack.append(to_put['parent_max_index_PVT'])
                        col_idx_in_parent_PVT_stack.append(to_put['col_idx_in_parent_PVT'])
                        idx_in_this_col_in_parent_PVT_stack.append(to_put['idx_in_this_col_in_parent_PVT'])
                        fixed_value_assignments_stack.append(to_put['fixed_value_assignments'])
                        fixed_value_assignments_positions_stack.append(to_put['fixed_value_assignments_positions'])
                        left_side_binary_search_stack.append(to_put['for_left_binary'])
                        shifted_length_stack.append(to_put['shifted_length'])
                        fixed_value_assignments_to_tighten_stack.append(to_put['fixed_value_assignments_to_tighten'])
                fixed_value_assignments = fixed_value_assignments_for_tighten
                fixed_value_assignments_positions[fixed_att] = tight_value_idx
                if tight_value_idx == 0 and len(to_put_to_stack) > 0:
                    to_put_to_stack.pop()
                else:
                    #  no need to test values between now and last fixed value
                    if len(to_put_to_stack) > 0:
                        to_put_to_stack[-1]['idx_in_this_col_in_parent_PVT'] = tight_value_idx - 1
                        to_put_to_stack[-1]['fixed_value_assignments'][fixed_att] = values_above[tight_value_idx - 1]
                        to_put_to_stack[-1]['fixed_value_assignments_to_tighten'] = values_above[: tight_value_idx - 1]
                        to_put_to_stack[-1]['fixed_value_assignments_positions'][fixed_att] = tight_value_idx - 1
        else:
            fva = [full_value_assignment[k] for k in full_PVT_head]
            full_value_assignment_positions = dict(zip(PVT_head, last_satisfying_bounding_relaxation_location))
            full_value_assignment_positions = {**full_value_assignment_positions,
                                               **fixed_value_assignments_positions}

            find_relaxation[num_columns].append(find_base_refinement)  # FIXME: is this find_relaxation necessary?

            minimal_refinements, minimal_refinements_positions, added = \
                update_minimal_relaxation_and_position_refinement(minimal_refinements, minimal_refinements_positions,
                                                                  fva, [full_value_assignment_positions[x] for x in
                                                                        full_PVT_head],
                                                                  shifted_length, initial_PVT, selection_numeric,
                                                                  full_PVT_head)
        # print("minimal_refinements: {}".format(minimal_refinements))

        if num_columns == 1:
            if len(PVT_head_stack) > 0:
                next_col_num_in_stack = len(PVT_head_stack[-1])
            else:
                next_col_num_in_stack = len(full_PVT_head)
            check_to_put_to_stack_refinement(to_put_to_stack, next_col_num_in_stack, num_columns, find_relaxation,
                                             PVT_stack, PVT_head_stack, max_index_PVT_stack, parent_PVT_stack,
                                             parent_PVT_head_stack,
                                             parent_max_index_PVT_stack, col_idx_in_parent_PVT_stack,
                                             idx_in_this_col_in_parent_PVT_stack,
                                             fixed_value_assignments_stack, fixed_value_assignments_positions_stack,
                                             fixed_value_assignments_to_tighten_stack, left_side_binary_search_stack,
                                             shifted_length_stack,
                                             idx_in_this_col_in_parent_PVT,
                                             PVT, PVT_head, max_index_PVT, parent_PVT, parent_PVT_head,
                                             parent_max_index_PVT,
                                             col_idx_in_parent_PVT, fixed_value_assignments,
                                             fixed_value_assignments_positions)
            continue
        # recursion
        col_idx = 0

        index_to_insert_to_stack = len(PVT_stack)
        insert_idx_fixed_value_assignments_to_tighten_stack = len(fixed_value_assignments_to_tighten_stack)
        index_to_insert_to_put = len(to_put_to_stack)
        original_max_index_PVT = max_index_PVT.copy()
        original_shifted_length = copy.deepcopy(shifted_length)
        original_PVT_col_num = len(PVT_head)
        original_PVT_head = copy.deepcopy(PVT_head)

        col_name_to_drop = []
        col_idx_to_drop = []

        recursion_end = True

        PVT.apply(recursion, axis=0)
        if recursion_end:
            if len(PVT_head_stack) > 0:
                next_col_num_in_stack = len(PVT_head_stack[-1])
            else:
                next_col_num_in_stack = len(full_PVT_head)
            check_to_put_to_stack_refinement(to_put_to_stack, next_col_num_in_stack, num_columns, find_relaxation,
                                             PVT_stack, PVT_head_stack, max_index_PVT_stack, parent_PVT_stack,
                                             parent_PVT_head_stack,
                                             parent_max_index_PVT_stack, col_idx_in_parent_PVT_stack,
                                             idx_in_this_col_in_parent_PVT_stack,
                                             fixed_value_assignments_stack, fixed_value_assignments_positions_stack,
                                             fixed_value_assignments_to_tighten_stack, left_side_binary_search_stack,
                                             shifted_length_stack,
                                             idx_in_this_col_in_parent_PVT,
                                             PVT, PVT_head, max_index_PVT, parent_PVT, parent_PVT_head,
                                             parent_max_index_PVT,
                                             col_idx_in_parent_PVT, fixed_value_assignments,
                                             fixed_value_assignments_positions)

    return minimal_refinements


# this procedure is called only when the current recursion ends here
def check_to_put_to_stack_refinement(to_put_to_stack, next_col_num_in_stack, this_num_columns, find_relaxation,
                                     PVT_stack, PVT_head_stack, max_index_PVT_stack, parent_PVT_stack,
                                     parent_PVT_head_stack,
                                     parent_max_index_PVT_stack, col_idx_in_parent_PVT_stack,
                                     idx_in_this_col_in_parent_PVT_stack,
                                     fixed_value_assignments_stack, fixed_value_assignments_positions_stack,
                                     fixed_value_assignments_to_tighten_stack, left_side_binary_search_stack,
                                     shifted_length_stack,
                                     idx_in_this_col_in_parent_PVT,
                                     PVT, PVT_head, max_index_PVT, parent_PVT, parent_PVT_head, parent_max_index_PVT,
                                     col_idx_in_parent_PVT, fixed_value_assignments, fixed_value_assignments_positions):
    # if idx_in_this_col_in_parent_PVT == 0:
    #     return False
    if len(to_put_to_stack) == 0:
        return False
    PVT_from_to_put = False
    # print("next_col_num_in_stack = {}, this_num_columns = {}".format(next_col_num_in_stack, this_num_columns))
    if True:  # len([k for k in find_relaxation[this_num_columns] if k is True]) > 0:
        to_put = to_put_to_stack.pop()
        if to_put != {}:
            PVT_stack.append(to_put['PVT'])
            PVT_head_stack.append(to_put['PVT_head'])
            max_index_PVT_stack.append(to_put['max_index_PVT'])
            parent_PVT_stack.append(to_put['parent_PVT'])
            parent_PVT_head_stack.append(to_put['parent_PVT_head'])
            parent_max_index_PVT_stack.append(to_put['parent_max_index_PVT'])
            col_idx_in_parent_PVT_stack.append(to_put['col_idx_in_parent_PVT'])
            idx_in_this_col_in_parent_PVT_stack.append(to_put['idx_in_this_col_in_parent_PVT'])
            fixed_value_assignments_stack.append(to_put['fixed_value_assignments'])
            fixed_value_assignments_positions_stack.append(to_put['fixed_value_assignments_positions'])
            left_side_binary_search_stack.append(to_put['for_left_binary'])
            shifted_length_stack.append(to_put['shifted_length'])
            if to_put['idx_in_this_col_in_parent_PVT'] > 0:
                fixed_value_assignments_to_tighten_stack.append(to_put['fixed_value_assignments_to_tighten'])
                to_put2 = dict()
                to_put2['PVT'] = to_put['PVT'].copy()
                to_put2['PVT_head'] = to_put['PVT_head'].copy()
                to_put2['max_index_PVT'] = to_put['max_index_PVT'].copy()
                to_put2['parent_PVT'] = to_put['parent_PVT'].copy()
                to_put2['parent_PVT_head'] = to_put['parent_PVT_head'].copy()
                to_put2['parent_max_index_PVT'] = to_put['parent_max_index_PVT'].copy()
                to_put2['col_idx_in_parent_PVT'] = to_put['col_idx_in_parent_PVT']
                to_put2['idx_in_this_col_in_parent_PVT'] = to_put['idx_in_this_col_in_parent_PVT'] - 1
                fixed_value_assignments_to_put = copy.deepcopy(to_put['fixed_value_assignments'])
                a = to_put['parent_PVT_head']
                b = to_put['col_idx_in_parent_PVT']
                c = a[b]
                fixed_value_assignments_to_put[c] \
                    = to_put['parent_PVT'].iloc[
                    to_put['idx_in_this_col_in_parent_PVT'] - 1, to_put['col_idx_in_parent_PVT']]
                to_put2['fixed_value_assignments'] = fixed_value_assignments_to_put
                fixed_value_assignments_positions_to_put = copy.deepcopy(to_put['fixed_value_assignments_positions'])
                fixed_value_assignments_positions_to_put[to_put['parent_PVT_head'][to_put['col_idx_in_parent_PVT']]] \
                    = to_put['idx_in_this_col_in_parent_PVT'] - 1
                to_put2['fixed_value_assignments_positions'] = fixed_value_assignments_positions_to_put
                to_put2['for_left_binary'] = to_put['for_left_binary']
                to_put2['shifted_length'] = to_put['shifted_length']
                if to_put['idx_in_this_col_in_parent_PVT'] > 1:
                    to_put2['fixed_value_assignments_to_tighten'] = to_put['fixed_value_assignments_to_tighten'][:-1]
                to_put_to_stack.append(to_put2)
        # find_relaxation[this_num_columns].pop()
    return PVT_from_to_put


# this procedure is called only when the current recursion ends here
def check_to_put_to_stack(to_put_to_stack, next_col_num_in_stack, this_num_columns, find_relaxation,
                          PVT_stack, PVT_head_stack, max_index_PVT_stack, parent_PVT_stack, parent_PVT_head_stack,
                          parent_max_index_PVT_stack, col_idx_in_parent_PVT_stack, idx_in_this_col_in_parent_PVT_stack,
                          fixed_value_assignments_stack, fixed_value_assignments_positions_stack,
                          fixed_value_assignments_to_tighten_stack, left_side_binary_search_stack, shifted_length_stack,
                          idx_in_this_col_in_parent_PVT,
                          PVT, PVT_head, max_index_PVT, parent_PVT, parent_PVT_head, parent_max_index_PVT,
                          col_idx_in_parent_PVT, fixed_value_assignments, fixed_value_assignments_positions):
    # if idx_in_this_col_in_parent_PVT == 0:
    #     return False
    if len(to_put_to_stack) == 0:
        return False
    PVT_from_to_put = False
    # print("next_col_num_in_stack = {}, this_num_columns = {}".format(next_col_num_in_stack, this_num_columns))
    if True:  # len([k for k in find_relaxation[this_num_columns] if k is True]) > 0:
        to_put = to_put_to_stack.pop()
        if to_put != {}:
            PVT_stack.append(to_put['PVT'])
            PVT_head_stack.append(to_put['PVT_head'])
            max_index_PVT_stack.append(to_put['max_index_PVT'])
            parent_PVT_stack.append(to_put['parent_PVT'])
            parent_PVT_head_stack.append(to_put['parent_PVT_head'])
            parent_max_index_PVT_stack.append(to_put['parent_max_index_PVT'])
            col_idx_in_parent_PVT_stack.append(to_put['col_idx_in_parent_PVT'])
            idx_in_this_col_in_parent_PVT_stack.append(to_put['idx_in_this_col_in_parent_PVT'])
            fixed_value_assignments_stack.append(to_put['fixed_value_assignments'])
            fixed_value_assignments_positions_stack.append(to_put['fixed_value_assignments_positions'])
            left_side_binary_search_stack.append(to_put['for_left_binary'])
            shifted_length_stack.append(to_put['shifted_length'])
            if to_put['idx_in_this_col_in_parent_PVT'] > 0:
                fixed_value_assignments_to_tighten_stack.append(to_put['fixed_value_assignments_to_tighten'])
                to_put2 = dict()
                to_put2['PVT'] = to_put['PVT'].copy()
                to_put2['PVT_head'] = to_put['PVT_head'].copy()
                to_put2['max_index_PVT'] = to_put['max_index_PVT'].copy()
                to_put2['parent_PVT'] = to_put['parent_PVT'].copy()
                to_put2['parent_PVT_head'] = to_put['parent_PVT_head'].copy()
                to_put2['parent_max_index_PVT'] = to_put['parent_max_index_PVT'].copy()
                to_put2['col_idx_in_parent_PVT'] = to_put['col_idx_in_parent_PVT']
                to_put2['idx_in_this_col_in_parent_PVT'] = to_put['idx_in_this_col_in_parent_PVT'] - 1
                fixed_value_assignments_to_put = copy.deepcopy(to_put['fixed_value_assignments'])
                a = to_put['parent_PVT_head']
                b = to_put['col_idx_in_parent_PVT']
                c = a[b]
                fixed_value_assignments_to_put[c] \
                    = to_put['parent_PVT'].iloc[
                    to_put['idx_in_this_col_in_parent_PVT'] - 1, to_put['col_idx_in_parent_PVT']]
                to_put2['fixed_value_assignments'] = fixed_value_assignments_to_put
                fixed_value_assignments_positions_to_put = copy.deepcopy(to_put['fixed_value_assignments_positions'])
                fixed_value_assignments_positions_to_put[to_put['parent_PVT_head'][to_put['col_idx_in_parent_PVT']]] \
                    = to_put['idx_in_this_col_in_parent_PVT'] - 1
                to_put2['fixed_value_assignments_positions'] = fixed_value_assignments_positions_to_put
                to_put2['for_left_binary'] = to_put['for_left_binary']
                to_put2['shifted_length'] = to_put['shifted_length']
                if to_put['idx_in_this_col_in_parent_PVT'] > 1:
                    to_put2['fixed_value_assignments_to_tighten'] = to_put['fixed_value_assignments_to_tighten'][:-1]
                to_put_to_stack.append(to_put2)
        # find_relaxation[this_num_columns].pop()
    return PVT_from_to_put


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
                at, va = columns_delta_table[att_idx].rsplit("__", 1)
                select_categorical[at].append(va)
            elif ar[att_idx] == -1:
                at, va = columns_delta_table[att_idx].rsplit("__", 1)
                select_categorical[at].remove(va)
        minimal_refinements.append({'numeric': select_numeric, 'categorical': select_categorical})
    return minimal_refinements


def whether_satisfy_fairness_constraints(data_file_prefix, separator, data_file_format, tables, joinkeys, comparekeys,
                                         selected_attributes,
                                         sensitive_attributes, fairness_constraints, numeric_attributes,
                                         categorical_attributes, selection_numeric_attributes,
                                         selection_categorical_attributes):
    global df1, df2
    if len(tables) == 1:  # no join
        data = pd.read_csv(data_file_prefix + tables[0] + data_file_format, sep=separator)
    else:
        data = pd.read_csv(data_file_prefix + tables[0] + data_file_format, sep=separator)
        for idx in range(1, len(tables)):
            righttable = pd.read_csv(data_file_prefix + tables[idx] + data_file_format, sep=separator)
            print(joinkeys[idx - 1][0], joinkeys[idx - 1][1], righttable.columns.tolist(), )
            data = pd.merge(left=data, right=righttable, how="inner", left_on=joinkeys[idx - 1][0],
                            right_on=joinkeys[idx - 1][1])
        # compare keys
        if len(comparekeys) > 0:
            for ck in comparekeys:
                data = data[data[ck[0]] < data[ck[1]]]
    # if all values of a categorical attribute are integer, change that column type to string
    for att in categorical_attributes:
        if data[att].apply(is_int).all():
            data[att] = data[att].apply(lambda x: str(int(x)) if isinstance(x, float) else str(x))
    pe_dataframe = copy.deepcopy(data)
    # get data selected
    for att in selection_numeric_attributes:
        if selection_numeric_attributes[att][0] == '>':
            pe_dataframe = pe_dataframe[pe_dataframe[att] > selection_numeric_attributes[att][1]]
        elif selection_numeric_attributes[att][0] == ">=":
            pe_dataframe = pe_dataframe[pe_dataframe[att] >= selection_numeric_attributes[att][1]]
        elif selection_numeric_attributes[att][0] == "<":
            pe_dataframe = pe_dataframe[pe_dataframe[att] < selection_numeric_attributes[att][1]]
        else:
            pe_dataframe = pe_dataframe[pe_dataframe[att] <= selection_numeric_attributes[att][1]]
    for att in selection_categorical_attributes:
        pe_dataframe = pe_dataframe[pe_dataframe[att].isin(selection_categorical_attributes[att])]

    for fc in fairness_constraints:
        if "first_sensitive_attributes" in fc:
            df = copy.deepcopy(pe_dataframe)
            first_sensitive_attributes = fc['first_sensitive_attributes']
            second_sensitive_attributes = fc['second_sensitive_attributes']
            for att in first_sensitive_attributes:
                df1 = df[df[att] == first_sensitive_attributes[att]]
            num1 = len(df1)
            for att in second_sensitive_attributes:
                df2 = df[df[att] == second_sensitive_attributes[att]]
            num2 = len(df2)
            if 'plus' in fc:
                if not eval(str(num1) + fc['symbol'] + str(num2 + fc['plus'])):
                    return False, data
            elif 'times' in fc:
                if not eval(str(num1) + fc['symbol'] + str(num2 * fc['times'])):
                    return False, data
        else:
            df = copy.deepcopy(pe_dataframe)
            sensitive_attributes = fc['sensitive_attributes']
            for att in sensitive_attributes:
                df = df[df[att] == sensitive_attributes[att]]
            num = len(df)
            if not eval(str(num) + fc['symbol'] + str(fc['number'])):
                return False, data
    return True, data


########################################################################################################################


def FindMinimalRefinement(data_file_prefix, separator, query_info, constraint_info, data_file_format,
                          time_limit=5 * 60):
    time1 = time.time()
    global assign_to_provenance_num
    assign_to_provenance_num = 0
    # data = pd.read_csv(data_file, index_col=False)

    tables = query_info['tables']
    joinkeys = []
    comparekeys = []
    if "joinkeys" in query_info:
        joinkeys = query_info["joinkeys"]
    if "comparekeys" in query_info:
        comparekeys = query_info["comparekeys"]
    numeric_attributes = []
    categorical_attributes = {}
    selection_numeric_attributes = {}
    selection_categorical_attributes = {}
    if 'selection_numeric_attributes' in query_info:
        selection_numeric_attributes = query_info['selection_numeric_attributes']
        numeric_attributes = list(selection_numeric_attributes.keys())
    if 'selection_categorical_attributes' in query_info:
        selection_categorical_attributes = query_info['selection_categorical_attributes']
        categorical_attributes = list(selection_categorical_attributes.keys())  # query_info['categorical_attributes']
    selected_attributes = numeric_attributes + [x for x in categorical_attributes]
    # print("selected_attributes", selected_attributes)

    sensitive_attributes = constraint_info['all_sensitive_attributes']
    fairness_constraints = constraint_info['fairness_constraints']

    pd.set_option('display.float_format', '{:.2f}'.format)

    # data:after join
    whether_satisfy, data = whether_satisfy_fairness_constraints(data_file_prefix, separator, data_file_format, tables,
                                                                 joinkeys, comparekeys, selected_attributes,
                                                                 sensitive_attributes, fairness_constraints,
                                                                 numeric_attributes, categorical_attributes,
                                                                 selection_numeric_attributes,
                                                                 selection_categorical_attributes)
    if whether_satisfy:
        print("original query satisfies constraints already")
        return {}, time.time() - time1, assign_to_provenance_num, time.time() - time1, 0

    # whether it's single-direction
    only_smaller_than = True
    only_greater_than = True
    for fc in fairness_constraints:
        if 'first_sensitive_attributes' in fc:
            only_greater_than = False
            only_smaller_than = False
            break
        elif fc['symbol'] == ">=" or fc['symbol'] == ">":
            only_smaller_than = False
        else:
            only_greater_than = False
        if (not only_greater_than) and (not only_smaller_than):
            break

    if only_greater_than:
        fairness_constraints_provenance_greater_than, fairness_constraints_provenance_smaller_than, \
            contraction_threshold \
            = subtract_provenance_relaxation_contraction(data, selected_attributes, sensitive_attributes,
                                                         fairness_constraints,
                                                         numeric_attributes, categorical_attributes,
                                                         selection_numeric_attributes,
                                                         selection_categorical_attributes, only_greater_than,
                                                         only_smaller_than)
        time_provenance2 = time.time()
        provenance_time = time_provenance2 - time1
        if time.time() - time1 > time_limit:
            print("time out")
            return [], time.time() - time1, assign_to_provenance_num, provenance_time, 0

        time_search1 = time.time()
        PVT, PVT_head, categorical_att_columns, \
            max_index_PVT = build_PVT_relax_only(data, selected_attributes,
                                                 numeric_attributes,
                                                 categorical_attributes,
                                                 selection_numeric_attributes,
                                                 selection_categorical_attributes,
                                                 sensitive_attributes,
                                                 fairness_constraints,
                                                 fairness_constraints_provenance_greater_than,
                                                 fairness_constraints_provenance_smaller_than)

        if time.time() - time1 > time_limit:
            print("time out")
            return [], time.time() - time1, assign_to_provenance_num, provenance_time, 0

        checked_assignments_satisfying = []
        checked_assignments_unsatisfying = []
        original_PVT_head = copy.deepcopy(PVT_head)
        minimal_refinements = searchPVT_relaxation(PVT, PVT_head, numeric_attributes,
                                                   categorical_attributes, selection_numeric_attributes,
                                                   selection_categorical_attributes, len(PVT_head),
                                                   fairness_constraints_provenance_greater_than, PVT,
                                                   PVT_head,
                                                   max_index_PVT,
                                                   checked_assignments_satisfying,
                                                   checked_assignments_unsatisfying, time_limit)
        time2 = time.time()
        print("provenance time = {}".format(provenance_time))
        print("searching time = {}".format(time2 - time_search1))
        order_in_results = original_PVT_head
        print("order_in_results = {}".format(order_in_results))
        return minimal_refinements, order_in_results, time2 - time1, assign_to_provenance_num, provenance_time, time2 - time_search1

    elif only_smaller_than:
        fairness_constraints_provenance_greater_than, fairness_constraints_provenance_smaller_than, \
            contraction_threshold \
            = subtract_provenance_relaxation_contraction(data, selected_attributes, sensitive_attributes,
                                                         fairness_constraints,
                                                         numeric_attributes, categorical_attributes,
                                                         selection_numeric_attributes,
                                                         selection_categorical_attributes, only_greater_than,
                                                         only_smaller_than)
        time_provenance2 = time.time()
        provenance_time = time_provenance2 - time1
        if time.time() - time1 > time_limit:
            print("time out")
            return [], time.time() - time1, assign_to_provenance_num, provenance_time, 0

        time_search1 = time.time()
        PVT, PVT_head, categorical_att_columns, \
            max_index_PVT = build_PVT_contract_only(data, selected_attributes, numeric_attributes,
                                                    categorical_attributes,
                                                    selection_numeric_attributes,
                                                    selection_categorical_attributes,
                                                    sensitive_attributes,
                                                    fairness_constraints,
                                                    fairness_constraints_provenance_greater_than,
                                                    fairness_constraints_provenance_smaller_than)
        if time.time() - time1 > time_limit:
            print("time out")
            return [], time.time() - time1, assign_to_provenance_num, provenance_time, 0

        checked_assignments_satisfying = []
        checked_assignments_unsatisfying = []
        original_PVT_head = copy.deepcopy(PVT_head)
        minimal_refinements = searchPVT_contraction(PVT, PVT_head, numeric_attributes,
                                                    categorical_attributes, selection_numeric_attributes,
                                                    selection_categorical_attributes, len(PVT_head),
                                                    fairness_constraints_provenance_smaller_than, PVT,
                                                    PVT_head,
                                                    max_index_PVT,
                                                    checked_assignments_satisfying,
                                                    checked_assignments_unsatisfying, time_limit)
        time2 = time.time()
        print("provenance time = {}".format(provenance_time))
        print("searching time = {}".format(time2 - time_search1))
        order_in_results = original_PVT_head
        print("order_in_results = {}".format(order_in_results))
        return minimal_refinements, order_in_results, time2 - time1, assign_to_provenance_num, provenance_time, time2 - time_search1

    fairness_constraints_provenance_greater_than, fairness_constraints_provenance_smaller_than, \
        fairness_constraints_provenance_complex, contraction_threshold \
        = subtract_provenance_refinement(data, selected_attributes, sensitive_attributes,
                                         fairness_constraints,
                                         numeric_attributes, categorical_attributes,
                                         selection_numeric_attributes,
                                         selection_categorical_attributes, only_greater_than,
                                         only_smaller_than)
    time_provenance2 = time.time()
    provenance_time = time_provenance2 - time1
    if time.time() - time1 > time_limit:
        print("time out")
        return [], time.time() - time1, assign_to_provenance_num, provenance_time, 0

    time_search1 = time.time()
    PVT, PVT_head, categorical_att_columns, \
        max_index_PVT, possible_values_lists = build_PVT_refinement(data, selected_attributes,
                                                                    numeric_attributes,
                                                                    categorical_attributes,
                                                                    selection_numeric_attributes,
                                                                    selection_categorical_attributes,
                                                                    sensitive_attributes,
                                                                    fairness_constraints,
                                                                    fairness_constraints_provenance_greater_than,
                                                                    fairness_constraints_provenance_smaller_than,
                                                                    fairness_constraints_provenance_complex,
                                                                    contraction_threshold)
    if time.time() - time1 > time_limit:
        print("time out")
        return [], time.time() - time1, assign_to_provenance_num, provenance_time, 0
    original_PVT_head = copy.deepcopy(PVT_head)
    checked_assignments_satisfying = []
    checked_assignments_unsatisfying = []
    minimal_refinements = searchPVT_refinement(PVT, PVT_head, possible_values_lists, numeric_attributes,
                                               categorical_attributes, selection_numeric_attributes,
                                               selection_categorical_attributes, len(PVT_head),
                                               fairness_constraints_provenance_greater_than,
                                               fairness_constraints_provenance_smaller_than,
                                               fairness_constraints_provenance_complex, PVT,
                                               copy.deepcopy(PVT_head), max_index_PVT,
                                               checked_assignments_satisfying,
                                               checked_assignments_unsatisfying, time_limit)
    time2 = time.time()
    print("provenance time = {}".format(provenance_time))
    print("searching time = {}".format(time2 - time_search1))
    print("assign_to_provenance_num = {}".format(assign_to_provenance_num))
    # reorder the minimal refinements, and add values of categorical attribtues in contraction threshold

    head_length = len(original_PVT_head)
    num_categorical = head_length - len(numeric_attributes)
    order_in_results = original_PVT_head[num_categorical:] + original_PVT_head[:num_categorical]
    # print(order_in_results)
    reordered_minimal_refinements = list()
    for mr in minimal_refinements:
        reordered_minimal_refinements.append(mr[num_categorical:] + mr[:num_categorical])
    for att in contraction_threshold:
        if att not in numeric_attributes:
            for v in contraction_threshold[att]:
                for i in range(len(order_in_results)):
                    col = order_in_results[i]
                    if col in numeric_attributes:
                        continue
                    attcol, vcol = col.split("__")
                    if attcol == att:
                        order_in_results.insert(i, att + '__' + v)
                        for mr in reordered_minimal_refinements:
                            mr.insert(i, 1)
                        break
    print("order in results: {}".format(order_in_results))
    return reordered_minimal_refinements, order_in_results, time2 - time1, assign_to_provenance_num, provenance_time, time2 - time_search1

# data_file = r"../InputData/Adult/adult.data"
# query_file = r"../Experiment/adult/exp_1_runningtime/query1.json"
# constraint_file = r"../Experiment/adult/exp_1_runningtime/constraint1.json"
# time_limit = 5 * 60

# data_file = r"../InputData/Healthcare/incomeK/before_selection_incomeK.csv"
# query_file = r"../Experiment/Healthcare/constraint_change/query9.json"
# constraint_file = r"../Experiment/Healthcare/constraint_change/constraint1.json"
#
#
# data_file = r"../InputData/Pipelines/healthcare/incomeK/before_selection_incomeK.csv"
# query_file = r"../InputData/Pipelines/healthcare/incomeK/relaxation/query4.json"
# constraint_file = r"../InputData/Pipelines/healthcare/incomeK/relaxation/constraint2.json"
#
#
# data_file = r"toy_examples/example5.csv"
# query_file = r"toy_examples/query.json"
# constraint_file = r"toy_examples/constraint.json"

# data_file = r"../InputData/Student/student-mat_cat_ranked.csv"
# query_file = r"../Experiment/student/demo/query1.json"
# constraint_file = r"../Experiment/student/demo/constraint1.json"
# time_limit = 5 * 60


#
# data_file_prefix = r"../InputData/TPC-H/1Mdata/"
# query_file = r"../Experiment/TPCH/1M/q3_contract3/q3_contract3.json"
# constraint_file = r"../Experiment/TPCH/1M/q3_contract3/constraint1.json"
# time_limit = 5 * 60
#
# print("\nour algorithm:\n")
#
# minimal_refinements, running_time, assign_num, _, _ = FindMinimalRefinement(data_file_prefix, query_file, constraint_file)
#
# minimal_refinements = [[float(y) for y in x] for x in minimal_refinements]
#
# print(*minimal_refinements, sep="\n")
# print("running time = {}".format(running_time))
#
#
# print("\nnaive algorithm:\n")
#
# minimal_refinements2, minimal_added_refinements2, running_time2 = lt.FindMinimalRefinement(data_file, query_file,
#                                                                                            constraint_file)
#
# # minimal_refinements2 = [[float(y) for y in x] for x in minimal_refinements2]
#
# print(*minimal_refinements2, sep="\n")
# print("running time = {}".format(running_time2))
#
#
# print("in naive_ans but not our:\n")
# for na in minimal_refinements2:
#     if na not in minimal_refinements:
#         print(na)
#
# print("in our but not naive_ans:\n")
# for na in minimal_refinements:
#     if na not in minimal_refinements2:
#         print(na)