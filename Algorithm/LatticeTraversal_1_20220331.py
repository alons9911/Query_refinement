"""
Executable
but only for relaxation (one-direction refinement)
"""

import copy
import json
import time

import numpy as np
import pandas as pd


def num2string(pattern):
    st = ''
    for i in pattern:
        if i != -1:
            st += str(i)
        st += '|'
    st = st[:-1]
    return st


def transform_to_refinement_format(minimal_added_refinements, numeric_attributes, selection_numeric_attributes,
                                   selection_categorical_attributes, numeric_att_domain_to_relax,
                                   categorical_att_domain_too_add):
    minimal_refinements = []
    for ar in minimal_added_refinements:
        select_numeric = copy.deepcopy(selection_numeric_attributes)
        select_categorical = copy.deepcopy(selection_categorical_attributes)
        i = 0
        for att in numeric_attributes:
            select_numeric[att][1] += ar[i]
            i += 1
        for k in categorical_att_domain_too_add:
            if ar[i] == 1:
                select_categorical[k[0]].append(k[1])
            i += 1
        minimal_refinements.append({'numeric': select_numeric, 'categorical': select_categorical})
    return minimal_refinements


def whether_satisfy_fairness_constraints(data, selected_attributes, sensitive_attributes, fairness_constraints,
                                         numeric_attributes, categorical_attributes, selection_numeric_attributes,
                                         selection_categorical_attributes):
    # get data selected
    def select(row):
        for att in selection_numeric_attributes:
            if pd.isnull(row[att]):
                return 0
            if not eval(
                    str(row[att]) + selection_numeric_attributes[att][0] + str(selection_numeric_attributes[att][1])):
                return 0
        for att in selection_categorical_attributes:
            if pd.isnull(row[att]):
                return 0
            if row[att] not in selection_categorical_attributes[att]:
                return 0
        return 1

    data['satisfy_selection'] = data[selected_attributes].apply(select, axis=1)
    data_selected = data[data['satisfy_selection'] == 1]
    # whether satisfy fairness constraint
    for fc in fairness_constraints:
        sensitive_attributes = fc['sensitive_attributes']
        df1 = data_selected[list(sensitive_attributes.keys())]
        df2 = pd.DataFrame([sensitive_attributes])
        data_selected_satisfying_fairness_constraint = df1.merge(df2)
        num = len(data_selected_satisfying_fairness_constraint)
        if not eval(str(num) + fc['symbol'] + str(fc['number'])):
            return False
    return True


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


def update_minimal_refinement(minimal_added_relaxations, r):
    dominated = []
    for mr in minimal_added_relaxations:
        if mr == r:
            return minimal_added_relaxations
        if dominate(mr, r):
            return minimal_added_relaxations
        elif dominate(r, mr):
            dominated.append(mr)
    if len(dominated) > 0:
        minimal_added_relaxations = [x for x in minimal_added_relaxations if x not in dominated]
    minimal_added_relaxations.append(r)
    return minimal_added_relaxations


def transform_refinement_format(this_refinement, numeric_att_domain_to_relax, num_numeric_att,
                                selection_numeric_attributes, numeric_attributes):
    minimal_refinement_values = copy.deepcopy(this_refinement)
    for i in range(num_numeric_att):
        if minimal_refinement_values[i] == -1:
            minimal_refinement_values[i] = 0
        else:
            minimal_refinement_values[i] = minimal_refinement_values[i] - \
                                           selection_numeric_attributes[numeric_attributes[i]][1]
    return minimal_refinement_values


"""
format:
numeric_att_domain: [age: {10, 20, 30}, income: {50, 60, 70}]
categorical_att_domain: [state: MI, state: IL, state: WI, city: Detroit, city: AA]
"""


# TODO: now it only considers relaxation
def LatticeTraversal(data, selected_attributes, sensitive_attributes, fairness_constraints,
                     numeric_attributes, categorical_attributes, selection_numeric_attributes,
                     selection_categorical_attributes, time_limit=5 * 60):

    categorical_att_domain_too_add = []
    for att in categorical_attributes:
        s = [(att, v) for v in categorical_attributes[att] if v not in selection_categorical_attributes[att]]
        categorical_att_domain_too_add += s
    numeric_att_domain_to_relax = dict()
    for att in numeric_attributes:
        if selection_numeric_attributes[att][0] == ">" or selection_numeric_attributes[att][0] == ">=":
            unique_values = data[att].unique().tolist()
            unique_values.sort(reverse=True)  # descending
            domain = unique_values
            for idx_domain in range(len(domain)):
                if domain[idx_domain] < selection_numeric_attributes[att][1]:
                    numeric_att_domain_to_relax[att] = [selection_numeric_attributes[att][1]] + domain[idx_domain:]
                    break
        else:
            unique_values = data[att].unique().tolist()
            unique_values.sort()  # ascending
            domain = unique_values
            for idx_domain in range(len(domain)):
                if domain[idx_domain] > selection_numeric_attributes[att][1]:
                    numeric_att_domain_to_relax[att] = [selection_numeric_attributes[att][1]] + domain[idx_domain:]
                    break
    num_numeric_att = len(selection_numeric_attributes)
    num_cate_variables = len(categorical_att_domain_too_add)
    print("categorical_att_domain_too_add:\n", categorical_att_domain_too_add)
    print("numeric_att_domain_to_relax:\n", numeric_att_domain_to_relax)
    minimal_refinements = []
    if whether_satisfy_fairness_constraints(data, selected_attributes, sensitive_attributes, fairness_constraints,
                                            numeric_attributes, categorical_attributes, selection_numeric_attributes,
                                            selection_categorical_attributes):
        return [{'numeric': selection_numeric_attributes, 'categorical': selection_categorical_attributes}]

    # I need to remember where I was last time
    att_idx = num_numeric_att + num_cate_variables
    last_time_selection = [0] * num_numeric_att + [0] * num_cate_variables
    this_refinement = [numeric_att_domain_to_relax[a][0] for a in numeric_attributes] + [0] * num_cate_variables
    new_selection_numeric_attributes = copy.deepcopy(selection_numeric_attributes)
    new_selection_categorical_attributes = copy.deepcopy(selection_categorical_attributes)
    while att_idx >= 0:
        # if time.time() - time1 > time_limit:
        #     break
        if att_idx == num_numeric_att + num_cate_variables:
            if whether_satisfy_fairness_constraints(data, selected_attributes, sensitive_attributes,
                                                    fairness_constraints, numeric_attributes, categorical_attributes,
                                                    new_selection_numeric_attributes,
                                                    new_selection_categorical_attributes):
                minimal_refinement_values = transform_refinement_format(this_refinement, numeric_att_domain_to_relax,
                                                                        num_numeric_att, selection_numeric_attributes,
                                                                        numeric_attributes)
                minimal_refinements = update_minimal_refinement(minimal_refinements, minimal_refinement_values)
            att_idx -= 1
        if att_idx < num_numeric_att:  # numeric
            if last_time_selection[att_idx] + 1 == len(numeric_att_domain_to_relax[numeric_attributes[att_idx]]):
                last_time_selection[att_idx] = -1
                this_refinement[att_idx] = -1
                new_selection_numeric_attributes[numeric_attributes[att_idx]][1] = \
                    selection_numeric_attributes[numeric_attributes[att_idx]][1]
                att_idx -= 1
                continue
            this_refinement[att_idx] = numeric_att_domain_to_relax[numeric_attributes[att_idx]][
                last_time_selection[att_idx] + 1]
            last_time_selection[att_idx] += 1
            new_selection_numeric_attributes[numeric_attributes[att_idx]][1] = this_refinement[att_idx]
            att_idx += 1
        else:  # categorical
            if last_time_selection[att_idx] == 1:
                last_time_selection[att_idx] = -1
                this_refinement[att_idx] = -1
                new_selection_categorical_attributes[
                    categorical_att_domain_too_add[att_idx - num_numeric_att][0]].remove(
                    categorical_att_domain_too_add[att_idx - num_numeric_att][1])
                att_idx -= 1
                continue
            this_refinement[att_idx] += 1
            last_time_selection[att_idx] += 1
            if this_refinement[att_idx] == 1:
                new_selection_categorical_attributes[
                    categorical_att_domain_too_add[att_idx - num_numeric_att][0]].append(
                    categorical_att_domain_too_add[att_idx - num_numeric_att][1])
            att_idx += 1
    return minimal_refinements, numeric_att_domain_to_relax, categorical_att_domain_too_add



#########################################################################################################


# data = pd.read_csv("toy_examples/example2.csv")
# print(data)
# with open('toy_examples/selection2.json') as f:
#     info = json.load(f)

data = pd.read_csv("../InputData/Pipelines/healthcare/before_selection.csv")
print(data)
with open("../InputData/Pipelines/healthcare/incomeK/relaxation/selection1.json") as f:
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

pd.set_option('display.float_format', '{:.3f}'.format)

time1 = time.time()
relaxed_conditions, numeric_att_domain_to_relax, categorical_att_domain_too_add \
    = LatticeTraversal(data, selected_attributes, sensitive_attributes, fairness_constraints,
                       numeric_attributes, categorical_attributes, selection_numeric_attributes,
                       selection_categorical_attributes, time_limit=5 * 60)

# print("\nrelaxed_conditions: \n")
# print(relaxed_conditions)

minimal_refinements = transform_to_refinement_format(relaxed_conditions, numeric_attributes,
                                                     selection_numeric_attributes, selection_categorical_attributes,
                                                     numeric_att_domain_to_relax, categorical_att_domain_too_add)

time2 = time.time()
print(*minimal_refinements, sep="\n")
print("running time = {}".format(time2 - time1))

