"""
This script can't run!
This is an early skeleton of the code.
"""

import numpy as np
import pandas as pd
import time
from intbitset import intbitset
import json


def subtract_provenance(data, selected_attributes, all_sensitive_attributes, fairness_constraints):
    """
Get provenance expressions
    :param all_sensitive_attributes: list of all att involved in fairness constraints
    :param fairness_constraints: [{'Gender': 'F', 'symbol': '>=', 'number': 3}]
    :param data: dataframe
    :param selected_attributes: attributes in selection conditions
    :return: a list of dictionaries
    """
    total_size_of_protected_group = 0
    fairness_constraints_provenance_greater_than = []
    fairness_constraints_provenance_smaller_than = []
    data['protected_greater_than'] = 0
    data['protected_smaller_than'] = 0

    def get_provenance(row, fc_dic, fc, protected_greater_than):
        sensitive_att_of_fc = list(fc["sensitive_attributes"].keys())
        sensitive_values_of_fc = {k: fc["sensitive_attributes"][k] for k in sensitive_att_of_fc}
        fairness_value_of_row = row[sensitive_att_of_fc]
        nonlocal total_size_of_protected_group
        if sensitive_values_of_fc == fairness_value_of_row.to_dict():
            terms = row[selected_attributes]
            fc_dic['provenance_expression'].append(terms)
            total_size_of_protected_group += 1
            if protected_greater_than:
                row['protected_greater_than'] = 1
            else:
                row['protected_smaller_than'] = 1

    for fc in fairness_constraints:
        fc_dic = dict()
        fc_dic['symbol'] = fc['symbol']
        fc_dic['number'] = fc['number']
        fc_dic['provenance_expression'] = []
        data[selected_attributes + sensitive_attributes +
             ['protected_greater_than', 'protected_smaller_than']].apply(get_provenance,
                                                               args=(fc_dic,
                                                                     fc,
                                                                     (fc['symbol'] == ">" or fc['symbol'] == ">=")),
                                                               axis=1)
        """
        handle < and <=:
        1. change to > and >=
        2. change number to size - number
        """
        if fc_dic['symbol'] == "<" or fc_dic['symbol'] == "<=":
            if fc_dic['symbol'] == "<":
                fc_dic['symbol'] = ">"
            else:
                fc_dic['symbol'] = ">="
            fc_dic['number'] = total_size_of_protected_group - fc_dic['number']
            fairness_constraints_provenance_smaller_than.append(fc_dic)
        else:
            fairness_constraints_provenance_greater_than.append(fc_dic)

    return fairness_constraints_provenance_greater_than, fairness_constraints_provenance_smaller_than


#
# def subtract_provenance(data, selected_attributes, all_sensitive_attributes, fairness_constraints):
#     """
# Get provenance expressions
#     :param all_sensitive_attributes: list of all att involved in fairness constraints
#     :param fairness_constraints: [{'Gender': 'F', 'symbol': '>=', 'number': 3}]
#     :param data: dataframe
#     :param selected_attributes: attributes in selection conditions
#     :return: a list of dictionaries
#     """
#     fairness_constraints_provenance = []
#
#     for fc in fairness_constraints:
#         fc_dic = dict()
#         fc_dic['symbol'] = fc['symbol']
#         fc_dic['number'] = fc['number']
#         fc_dic['provenance_expression'] = []
#         fairness_constraints_provenance.append(fc_dic)
#
#     def get_provenance(row):
#         for fc in fairness_constraints:
#
#
#         sensitive_att_of_fc = list(fc["sensitive_attributes"].keys())
#         sensitive_values_of_fc = {k: fc["sensitive_attributes"][k] for k in sensitive_att_of_fc}
#         fairness_value_of_row = row[sensitive_att_of_fc]
#         if sensitive_values_of_fc == fairness_value_of_row.to_dict():
#             terms = row[selected_attributes]
#             fc_dic['provenance_expression'].append(terms)
#
#     data[selected_attributes + sensitive_attributes].apply(get_provenance, axis=1)
#
#     return fairness_constraints_provenance
#

def simplify(row, provenance_expressions, sensitive_attributes):
    """
    simplify each condition in provenance_expressions about row, since row already satisfies it
    :param row: a row in sorted table but with all values zero
    :param provenance_expressions: got from subtract_provenance()
    """
    for fc in provenance_expressions:
        if fc['symbol'] == ">" or fc['symbol'] == ">=":
            fc['number'] -= 1
        else:
            raise Exception("provenance_expressions should not have < or <=")
        provenance_expression = fc['provenance_expression']
        to_remove = pd.Series()
        for pe in provenance_expression:
            pe_ = pe.drop(sensitive_attributes, axis=1)
            if pe_.equals(row):
                to_remove = pe
                break
        provenance_expression.remove(to_remove)


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
                       fairness_constraints_provenance_greater_than, fairness_constraints_provenance_smaller_than):
    """
    to build the sorted table
    :param data: dataframe
    :param numeric_attributes: list of names of numeric attributes [city, major, state]
    :param categorical_attributes: dictionary: {city: [domain of city], major: [domain of major]}
    :param selection_numeric: dictionary: {grade:[80, >], age:[30, <], hours: [100, <=]}
    :param selection_categorical: dictionary: {city: [accepted cities], major: [accepted majors]}
    :return: return the whole sorted table, including rows that already satisfy the selection conditions;
            also return delta table
    """
    columns = numeric_attributes.copy()
    for att, domain in categorical_attributes.items():
        for value in domain:
            columns.append(att + "_" + value)
    list_of_rows = []

    # remove rows that have nothing to do with fairness constraints
    def check_fc(row, sa):
        for k, v in sa.items():
            if row[k] != v:
                return 0
        return 1

    data['fairness_c'] = 0
    for fc in fairness_constraints:
        sa = fc["sensitive_attributes"]
        data['fairness_c'] = data[sensitive_attributes].apply(check_fc, args=(sa,), axis=1)
    data = data[data['fairness_c'] == 1]
    data.drop('fairness_c', axis=1, inplace=True)
    print(data)

    # remove columns for fairness constraints
    data.drop(sensitive_attributes, axis=1, inplace=True)

    # build delta table
    def iterrow(row):
        if row['protected_greater_than'] == 1:
            delta_values = []
            for att in numeric_attributes:
                delta_v = compute_delta_numeric_att(row[att], selection_numeric[att][0], selection_numeric[att][1], True)
                delta_values.append(delta_v)
            for att, domain in categorical_attributes.items():
                for value in domain:
                    if value not in selection_categorical[att] and row[att] == value:
                        delta_values.append(1)
                    else:
                        delta_values.append(0)
            if all(v == 0 for v in delta_values):
                simplify(row, fairness_constraints_provenance_greater_than, sensitive_attributes)
            else:
                list_of_rows.append(delta_values)
        if row['protected_smaller_than'] == 1:
            delta_values = []
            for att in numeric_attributes:
                delta_v = compute_delta_numeric_att(row[att], selection_numeric[att][0], selection_numeric[att][1], False)
                delta_values.append(delta_v)
            for att, domain in categorical_attributes.items():
                for value in domain:
                    if value not in selection_categorical[att] and row[att] == value:
                        delta_values.append(0)
                    else:
                        delta_values.append(1)
            if all(v == 0 for v in delta_values):
                simplify(row, fairness_constraints_provenance_smaller_than, sensitive_attributes)
            else:
                list_of_rows.append(delta_values)

    data.apply(iterrow, axis=1)
    delta_table = pd.DataFrame(list_of_rows, columns=columns)

    print("delta_table:\n", delta_table)
    sorted_table_by_column = dict()
    sorted_att_idx0 = np.argsort(delta_table[columns[0]])
    sorted_table_by_column[columns[0]] = sorted_att_idx0

    for att in columns[1:]:
        s = np.array(list(zip(delta_table[att], sorted_att_idx0)), dtype=[('value', 'int'), ('tiebreaker', 'int')])
        sorted_att_idx = np.argsort(s, order=['value', 'tiebreaker'])
        sorted_table_by_column[att] = sorted_att_idx
    sorted_table = pd.DataFrame(data=sorted_table_by_column)
    print("sorted_table:\n", sorted_table)
    return sorted_table, delta_table


"""
t is 
"""


# TODO
def assign_to_provenance(t, provenance_expressions):
    return True


# TODO
def get_relaxation_by_skyline(terms, delta_table):
    """
To get skyline of all terms in list terms
    :param terms: list of indices of terms. [1,3,5]
    :param delta_table: delta table returned by func build_sorted_table()
    :return: return skyline.
    """
    skyline = delta_table.iloc[terms].max()
    return skyline


def dominate(a, b):
    """
    whether relaxation a dominates b. relaxation has all delta values. it is not a selection condition
    :param a: relaxation, format of sorted table
    :param b: relaxation
    :return: true if a dominates b (a is minimal compared to b)
    """
    length = len(a)
    for i in range(length):
        if a[i] > b[i]:
            return False
    return True


# TODO: update stop line??
def update_minimal_relaxation(minimal_added_relaxations, r):
    dominated = []
    for mr in minimal_added_relaxations:
        if dominate(mr, r):
            return False
        elif dominate(r, mr):
            dominated.append(mr)
    if len(dominated) > 0:
        minimal_added_relaxations = [x for x in minimal_added_relaxations if x not in dominated]
    minimal_added_relaxations.append(r)
    return True


def search(sorted_table, delta_table, provenance_expressions, numeric_attributes,
           categorical_attributes, selection_numeric,
           selection_categorical):
    # TODO: simplify provenance expressions
    # TODO: simplify sorted table

    minimal_added_relaxations = []  # relaxation to add to original selection conditions
    checked = []  # set of bit arrays
    num_columns = len(sorted_table.columns)
    stop_line = pd.Series([len(sorted_table)] * num_columns, sorted_table.columns)
    row_num = 0

    def iterrow(row):
        nonlocal row_num
        for i, t in row.items():
            if stop_line[i] <= row_num:
                continue
            t_str = '0' * t + '1' + '0' * (num_columns - 1 - t)
            if t_str not in checked:
                skyline = get_relaxation_by_skyline([t], delta_table)
                if assign_to_provenance(skyline, provenance_expressions):
                    update_minimal_relaxation(minimal_added_relaxations, list(skyline))
                    checked.append(t_str)
            terms_above = list(sorted_table.loc[:row_num - 1, i])
            combo_list = [[y] for y in terms_above]
            while len(combo_list) > 0:
                combo = combo_list.pop(0)
                combo_w_t = [t] + combo
                combo_str = intbitset(combo_w_t)
                combo_str += '0' * (num_columns - len(combo) - 1)
                if combo_str in checked:
                    continue
                skyline = get_relaxation_by_skyline(combo_w_t, delta_table)
                if assign_to_provenance(skyline, provenance_expressions):
                    if not update_minimal_relaxation(minimal_added_relaxations, list(skyline)):
                        # generate children
                        idx = terms_above.index(combo[-1])
                        for x in terms_above[idx:]:
                            combo_list.append(combo + [x])
                checked.append(combo_str)
        row_num += 1

    sorted_table[:1].apply(iterrow, axis=1)
    max_stop_line = stop_line.max()
    sorted_table[1:max_stop_line].apply(iterrow, axis=1)
    return minimal_added_relaxations


"""
numeric_attributes: list of names of numeric attributes [city, major, state]
categorical_attributes: dictionary: {city: [domain of city], major: [domain of major]}
selection_numeric: dictionary: {grade:[80, >], age:[30, <], hours: [100, <=]}
selection_categorical: dictionary: {city: [accepted cities], major: [accepted majors]}
sensitive_attributes: list of attributes involved in fairness constraints: ['Gender', 'race']
fairness_constraints: list. Each item is a constraint represented in dict.
    "sensitive_attributes": {"Gender": "F", "Race": "Black"}
    "symbol": ">="
    "number": 3
"""
data = pd.read_csv("toy_examples/example.csv")
print(data)
with open('toy_examples/selection.json') as f:
    info = json.load(f)
all_attributes = data.columns.tolist()

sensitive_attributes = info['all_sensitive_attributes']
fairness_constraints = info['fairness_constraints']
selected_attributes = [x for x in all_attributes if x not in sensitive_attributes]
print("selected_attributes", selected_attributes)
selection_numeric_attributes = info['selection_numeric_attributes']
selection_categorical_attributes = info['selection_categorical_attributes']
numeric_attributes = list(selection_numeric_attributes.keys())
categorical_attributes = info['categorical_attributes']

fairness_constraints_provenance_greater_than, fairness_constraints_provenance_smaller_than = subtract_provenance(data, selected_attributes, sensitive_attributes, fairness_constraints)
print("provenance_expressions", fairness_constraints_provenance_greater_than, fairness_constraints_provenance_smaller_than)

sorted_table, delta_table = build_sorted_table(data, selected_attributes, numeric_attributes,
                                               categorical_attributes,
                                               selection_numeric_attributes,
                                               selection_categorical_attributes,
                                               sensitive_attributes, fairness_constraints,
                                               fairness_constraints_provenance_greater_than,
                                               fairness_constraints_provenance_smaller_than)

search(sorted_table, delta_table, "", numeric_attributes,
       categorical_attributes, selection_numeric_attributes, selection_categorical_attributes)
