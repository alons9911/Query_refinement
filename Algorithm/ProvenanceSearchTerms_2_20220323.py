"""
This script can't run!
I tried to define function programmatically on the fly but it shows scope errors.

"""

import numpy as np
import pandas as pd
import time
from intbitset import intbitset
import json


def subtract_provenance(data, selected_attributes, all_sensitive_attributes, fairness_constraints,
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

    columns_delta_table = numeric_attributes.copy()
    for att, domain in categorical_attributes.items():
        for value in domain:
            columns_delta_table.append(att + "_" + value)
    num_numeric_att = len(numeric_attributes)
    # provenance_in_one = f"def satisfy_constraints("
    # provenance_in_one += f", ".join(map(str, columns_delta_table)) + "):\n return ("
    provenance_in_one = "def satisfy_constraints():\n return ("

    def get_provenance(row, fc_dic, fc, protected_greater_than):
        print("row:", row)
        sensitive_att_of_fc = list(fc["sensitive_attributes"].keys())
        sensitive_values_of_fc = {k: fc["sensitive_attributes"][k] for k in sensitive_att_of_fc}
        fairness_value_of_row = row[sensitive_att_of_fc]
        nonlocal provenance_in_one
        if sensitive_values_of_fc == fairness_value_of_row.to_dict():
            terms = row[selected_attributes]
            for t in selected_attributes:
                if t in numeric_attributes:
                    provenance_in_one += " (" + str(terms[t]) + " " + selection_numeric_attributes[t][
                        0] + " " + t + ")*"
                else:
                    provenance_in_one += " (" + t + "_" + terms[t] + ")*"
            provenance_in_one = provenance_in_one[:-1]
            provenance_in_one += "+"
            fc_dic['provenance_expression'].append(terms)
            if protected_greater_than:
                row['protected_greater_than'] = 1
            else:
                row['protected_smaller_than'] = 1
        return row

        # return row['protected_greater_than'], row['protected_smaller_than']

    for fc in fairness_constraints:
        fc_dic = dict()
        fc_dic['symbol'] = fc['symbol']
        fc_dic['number'] = fc['number']
        fc_dic['provenance_expression'] = []
        # data['protected_greater_than'], data['protected_smaller_than'] \
        print("right before apply:\n")
        print(data[selected_attributes + sensitive_attributes +
                   ['protected_greater_than', 'protected_smaller_than']])
        data = data[selected_attributes + sensitive_attributes +
                    ['protected_greater_than', 'protected_smaller_than']].apply(get_provenance,
                                                                                args=(fc_dic,
                                                                                      fc,
                                                                                      (fc['symbol'] == ">" or fc[
                                                                                          'symbol'] == ">=")),
                                                                                axis=1)
        provenance_in_one = provenance_in_one[:-1]
        provenance_in_one += fc['symbol']
        provenance_in_one += str(fc['number'])
        provenance_in_one += ") and ("

        if fc_dic['symbol'] == "<" or fc_dic['symbol'] == "<=":
            fairness_constraints_provenance_smaller_than.append(fc_dic)
        else:
            fairness_constraints_provenance_greater_than.append(fc_dic)
    size = len(provenance_in_one)
    provenance_in_one = provenance_in_one[:size - 5]
    print("provenance_in_one:\n", provenance_in_one)
    data_rows_greater_than = data[data['protected_greater_than'] == 1]
    data_rows_smaller_than = data[data['protected_smaller_than'] == 1]
    return provenance_in_one, fairness_constraints_provenance_greater_than, fairness_constraints_provenance_smaller_than, \
           data_rows_greater_than, data_rows_smaller_than, columns_delta_table


# TODO: We can't remove a term from inequalities even if it satisfies selection conditions
# since when we change selection conditions, it may not satisfy
# for such a term, we can only remove the corresponding row in delta table (and sorted table)
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
                       fairness_constraints_provenance_greater_than, fairness_constraints_provenance_smaller_than,
                       data_rows_greater_than, data_rows_smaller_than, columns_delta_table, provenance_in_one
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
    list_of_rows_delta_table = []

    # # remove rows that have nothing to do with fairness constraints
    # def check_fc(row, sa):
    #     for k, v in sa.items():
    #         if row[k] != v:
    #             return 0
    #     return 1
    #
    # data['fairness_c'] = 0
    # for fc in fairness_constraints:
    #     sa = fc["sensitive_attributes"]
    #     data['fairness_c'] = data[sensitive_attributes].apply(check_fc, args=(sa,), axis=1)
    # data = data[data['fairness_c'] == 1]
    # data.drop('fairness_c', axis=1, inplace=True)
    # print(data)
    #
    # # remove columns for fairness constraints
    # data.drop(sensitive_attributes, axis=1, inplace=True)

    # build delta table
    def iterrow(row, greater_than=True):  # greater than is symbol in fairness constraint is greater than
        delta_values = []
        for att in numeric_attributes:
            if greater_than:
                if selection_numeric[att][0] == ">":
                    if row[att] > selection_numeric[att][1]:
                        delta_v = 0
                    else:
                        delta_v = selection_numeric[att][1] - row[att] + 1
                elif selection_numeric[att][0] == ">=":
                    if row[att] >= selection_numeric[att][1]:
                        delta_v = 0
                    else:
                        delta_v = selection_numeric[att][1] - row[att]
                elif selection_numeric[att][0] == "<":
                    if row[att] < selection_numeric[att][1]:
                        delta_v = 0
                    else:
                        delta_v = row[att] - selection_numeric[att][1] + 1
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
                        delta_v = - (row[att] - selection_numeric[att][1] + 1)
                elif selection_numeric[att][0] == ">=":
                    if row[att] < selection_numeric[att][1]:
                        delta_v = 0
                    else:
                        delta_v = - (row[att] - selection_numeric[att][1])
                elif selection_numeric[att][0] == "<":
                    if row[att] >= selection_numeric[att][1]:
                        delta_v = 0
                    else:
                        delta_v = - (selection_numeric[att][1] - row[att] + 1)
                else:  # selection_numeric[att][0] == "<=":
                    if row[att] > selection_numeric[att][1]:
                        delta_v = 0
                    else:
                        delta_v = - (selection_numeric[att][1] - row[att])
            delta_values.append(delta_v)
        for att, domain in categorical_attributes.items():
            for value in domain:
                if greater_than:
                    if value not in selection_categorical[att] and row[att] == value:
                        delta_values.append(1)
                    else:
                        delta_values.append(0)
                else:
                    if value in selection_categorical[att] and row[att] == value:
                        delta_values.append(-1)
                    else:
                        delta_values.append(0)
        if not all(v == 0 for v in delta_values):
            list_of_rows_delta_table.append(delta_values)

    data_rows_greater_than.apply(iterrow, args=(True,), axis=1)
    data_rows_smaller_than.apply(iterrow, args=(False,), axis=1)
    delta_table = pd.DataFrame(list_of_rows_delta_table, columns=columns_delta_table)

    print("delta_table:\n", delta_table)
    sorted_table_by_column = dict()
    # sorted_att_idx0 = np.argsort(delta_table[columns[0]].abs())
    # sorted_table_by_column[columns[0]] = sorted_att_idx0
    #
    # print(sorted_att_idx0)
    #
    # for att in columns[1:]:
    #     s = np.array(list(zip(delta_table[att].abs(), delta_table[columns[0]].abs())), dtype=[('value', 'int'), ('tiebreaker', 'int')])
    #     sorted_att_idx = np.argsort(s, order=['value', 'tiebreaker'])
    #     sorted_table_by_column[att] = sorted_att_idx
    #
    for att in columns_delta_table:
        s = np.array(list(zip(delta_table[att].abs(),
                              delta_table[columns_delta_table[0]].abs())),
                     dtype=[('value', 'int'), ('tiebreaker', 'int')])
        sorted_att_idx = np.argsort(s, order=['value', 'tiebreaker'])
        sorted_table_by_column[att] = sorted_att_idx

    sorted_table = pd.DataFrame(data=sorted_table_by_column)
    print("sorted_table:\n", sorted_table)
    return sorted_table, delta_table


# TODO
def assign_to_provenance(value_assignment, numeric_attributes, categorical_attributes, selection_numeric,
                         selection_categorical, columns_delta_table, num_columns):
    to_format = f"provenance_in_one.format("
    for i in range(num_columns):
        variable = columns_delta_table[i]
        if variable in numeric_attributes:
            if selection_numeric[variable][0] == ">" or selection_numeric[variable][0] == ">=":
                to_format += variable + "=" + str(selection_numeric[variable][1] - value_assignment[i]) + ", "
                # exec("%s = %d" % (variable, selection_numeric[variable][1] - value_assignment[i]), globals())
            else:
                to_format += variable + "=" + str(selection_numeric[variable][1] + value_assignment[i]) + ", "
                # exec("%s = %d" % (variable, selection_numeric[variable][1] + value_assignment[i]), globals())
        else:
            att, v = variable.split('_')
            if v in selection_categorical[att]:
                to_format += variable + "=" + str(1 + value_assignment[i]) + ", "
                # exec("%s = %d" % (variable, 1 + value_assignment[i]), globals())
            else:
                to_format += variable + "=" + str(value_assignment[i]) + ", "
                # exec("%s = %d" % (variable, value_assignment[i]), globals())
    to_format = to_format[:-2]
    to_format += ")"
    # a = exec(to_format)
    print("2:\n", provenance_in_one)
    b = exec(provenance_in_one)
    # print(a, b)
    # return satisfy_constraints()


# TODO
def get_relaxation(terms, delta_table, only_greater_than, only_smaller_than):
    """
To get skyline of all terms in list terms
    :param terms: list of indices of terms. [1,3,5]
    :param delta_table: delta table returned by func build_sorted_table()
    :return: return Ture or false whether terms can have a legitimate value assignment .
    """
    if only_greater_than:
        value_assignments = [delta_table.iloc[terms].max()]
        return True, value_assignments
    column_names = delta_table.columns.tolist()
    num_row = len(terms)
    num_col = len(column_names)
    
    value_assignments = [[]]
    # mark_satisfied_terms = [[0]*num_row]
    mark_satisfied_terms = [0]
    rows_to_compare = delta_table.iloc[terms]
    rows_to_compare_indices = rows_to_compare.index.tolist()
    final_result = []
    col = 0
    all_included = 0
    for i in rows_to_compare_indices:
        all_included = (1 << i) | all_included
    for i in range(num_col + 1):
        if i == num_col:
            num_result = len(value_assignments)
            for j in range(num_result):
                if mark_satisfied_terms[j] != all_included:
                    del value_assignments[j]
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
            # positive_terms_int = int(intbitset(rows_to_compare[rows_to_compare[col] > 0].index.tolist()).strbits(), 2)
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
        positive_terms_int = 0
        for n in non_zeros:
            positive_terms_int = (1 << n) | positive_terms_int
            for k in range(len(mark_satisfied_terms)):
                if mark_satisfied_terms[k] & positive_terms_int == 0:
                    mark_satisfied_terms_new.append(mark_satisfied_terms[k] | positive_terms_int)
                    v_ = value_assignments[k].copy()
                    v_.append(rows_to_compare.loc[n][col])
                    value_assignments_new.append(v_)
        value_assignments = value_assignments_new
        mark_satisfied_terms = mark_satisfied_terms_new

    return True, value_assignments


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


def search(sorted_table, delta_table, columns_delta_table, numeric_attributes,
           categorical_attributes, selection_numeric, selection_categorical,
           fairness_constraints_provenance_greater_than, fairness_constraints_provenance_smaller_than):
    # TODO: simplify provenance expressions
    # TODO: simplify sorted table

    minimal_added_relaxations = []  # relaxation to add to original selection conditions
    checked = []  # set of bit arrays
    num_columns = len(sorted_table.columns)
    stop_line = pd.Series([len(sorted_table)] * num_columns, sorted_table.columns)
    row_num = 0
    only_greater_than = False
    only_smaller_than = False
    if len(fairness_constraints_provenance_greater_than) != 0 and len(
            fairness_constraints_provenance_smaller_than) == 0:
        only_greater_than = True
    elif len(fairness_constraints_provenance_greater_than) == 0 and len(
            fairness_constraints_provenance_smaller_than) != 0:
        only_smaller_than = True

    def iterrow(row):
        nonlocal row_num
        for i, t in row.items():
            if stop_line[i] <= row_num:
                continue
            if only_smaller_than or only_greater_than:
                t_str = '0' * t + '1' + '0' * (num_columns - 1 - t)
                if t_str not in checked:
                    have_legitimate_value_assignment, value_assignments = get_relaxation([t],
                                                                                         delta_table,
                                                                                         only_greater_than,
                                                                                         only_smaller_than)
                    if have_legitimate_value_assignment:
                        if assign_to_provenance(value_assignments, numeric_attributes, categorical_attributes, selection_numeric, selection_categorical,
                                                columns_delta_table, num_columns):
                            update_minimal_relaxation(minimal_added_relaxations, list(value_assignments))
                            checked.append(t_str)
            terms_above = list(sorted_table.loc[:row_num - 1, i])
            combo_list = [[y] for y in terms_above]
            while len(combo_list) > 0:
                combo = combo_list.pop(0)
                combo_w_t = [t] + combo
                combo_str = intbitset(combo_w_t).strbits()
                # combo_str += '0' * (num_columns - len(combo) - 1)
                if combo_str in checked:
                    continue
                if combo_str == "1001":
                    print("break point ")
                    print("\n")
                have_legitimate_value_assignment, value_assignments = get_relaxation(combo_w_t,
                                                                                     delta_table,
                                                                                     only_greater_than,
                                                                                     only_smaller_than)
                if have_legitimate_value_assignment:
                    for va in value_assignments:
                        for value_assignment in value_assignments:
                            if assign_to_provenance(value_assignment, numeric_attributes, categorical_attributes, selection_numeric, selection_categorical,
                                                    columns_delta_table, num_columns):
                                if not update_minimal_relaxation(minimal_added_relaxations, list(va)):
                                    # generate children
                                    idx = terms_above.index(combo[-1])
                                    for x in terms_above[idx:]:
                                        combo_list.append(combo + [x])
                checked.append(combo_str)
        row_num += 1

    # sorted_table[:1].apply(iterrow, axis=1)
    # max_stop_line = stop_line.max()
    sorted_table.apply(iterrow, axis=1)
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

provenance_in_one, fairness_constraints_provenance_greater_than, fairness_constraints_provenance_smaller_than, \
data_rows_greater_than, data_rows_smaller_than, columns_delta_table \
    = subtract_provenance(data, selected_attributes, sensitive_attributes, fairness_constraints,
                          numeric_attributes, categorical_attributes, selection_numeric_attributes,
                          selection_categorical_attributes)
print("provenance_expressions", provenance_in_one, fairness_constraints_provenance_greater_than,
      fairness_constraints_provenance_smaller_than)
print("1: ", provenance_in_one)

# exec(provenance_in_one)


sorted_table, delta_table = build_sorted_table(data, selected_attributes, numeric_attributes,
                                               categorical_attributes,
                                               selection_numeric_attributes,
                                               selection_categorical_attributes,
                                               sensitive_attributes, fairness_constraints,
                                               fairness_constraints_provenance_greater_than,
                                               fairness_constraints_provenance_smaller_than,
                                               data_rows_greater_than,
                                               data_rows_smaller_than,
                                               columns_delta_table,
                                               provenance_in_one)

search(sorted_table, delta_table, columns_delta_table, numeric_attributes,
       categorical_attributes, selection_numeric_attributes, selection_categorical_attributes,
       fairness_constraints_provenance_greater_than, fairness_constraints_provenance_smaller_than)
