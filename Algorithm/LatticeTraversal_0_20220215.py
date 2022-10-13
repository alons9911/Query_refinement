import math

import pandas as pd
import time
from itertools import combinations
import pattern_count
from pandas.io import sql
from sqlalchemy import create_engine
import pymysql
import numpy as np
import copy

host = 'localhost'
port = 3306
user = 'root'
passwd = 'ljy19980228'
db = 'Proj2'
charset = 'utf8'
table = 'healthcare'


def num2string(pattern):
    st = ''
    for i in pattern:
        if i != -1:
            st += str(i)
        st += '|'
    st = st[:-1]
    return st


#
# # TODO: nan is made "" for now
# def selection2sql(selection, df, connection, cursor, table):
#     sql_stmt = "SELECT * FROM " + table + " WHERE\n"
#     num_att = 0
#     for k, values in selection.items():
#         if num_att > 0:
#             sql_stmt += "\n and \n"
#         if '' in values:
#             value_removed = values[:]
#             print()
#             value_removed.remove("")
#             print(value_removed)
#             sql_stmt += "(" + k + " in " + str(tuple(value_removed)) + " or " + k + " is null )"
#         else:
#             sql_stmt += "(" + k + " in " + str(tuple(values)) + ")"
#         num_att += 1
#     sql_stmt += ";"
#     print("sql_stmt: {}".format(sql_stmt))
#     return sql_stmt
#


# TODO: nan is made "" for now
def selection2sql(original_conditions_categorical, original_conditions_numeric, df, connection, cursor, table):
    sql_stmt = "SELECT * FROM " + table + " WHERE\n"
    num_att = 0
    for k, values in original_conditions_categorical.items():
        if num_att > 0:
            sql_stmt += "\n and \n"
        if '' in values:
            value_removed = values[:]
            print()
            value_removed.remove("")
            print(value_removed)
            sql_stmt += "(" + k + " in " + str(tuple(value_removed)) + " or " + k + " is null )"
        else:
            sql_stmt += "(" + k + " in " + str(tuple(values)) + ")"
        num_att += 1
    # if len(original_conditions_categorical) > 0:
    #     sql_stmt += "\n and \n"
    for k, value in original_conditions_numeric.items():
        if num_att > 0:
            sql_stmt += "\n and \n"
        sql_stmt += "(" + k + " >= " + str(value) + ")"
        num_att += 1
    sql_stmt += ";"
    print("sql_stmt: {}".format(sql_stmt))
    return sql_stmt


# TODO:
# fairness_constraints = [["race", "race3", 100]]
def satisfy_fairness_constraints(df, connection, cursor, table, fairness_constraints_no_less_than,
                                 original_conditions_categorical, original_conditions_numeric):
    print("Here is satisfy_fairness_constraints")
    print("cond_after_relaxation: {}".format(original_conditions_categorical, original_conditions_numeric))
    sql_stmt = selection2sql(original_conditions_categorical, original_conditions_numeric, df, connection, cursor,
                             table)
    cursor.execute(sql_stmt)
    sqlresult = cursor.fetchall()
    # TODO: how to handle index returned from db
    df_result = pd.DataFrame(sqlresult, columns=["index"] + df.columns.tolist())
    # see whether result satisfies the fairness constraints
    # fairness_constraints_no_less_than = [["race", "race3", 100]]
    for fc in fairness_constraints_no_less_than:
        df_result_fc = df_result[(df_result[fc[0]] == fc[1])]
        print("number of people satisfying fairness constraint = {}".format(len(df_result_fc)))
        if len(df_result_fc) < fc[2]:
            return False
    return True


# fairness_constraints = [["race", "race3", 100]]
def satisfy_original_fairness_constraints(df, connection, cursor, table, fairness_constraints_no_less_than,
                                          original_conditions_categorical, original_conditions_numeric):
    sql_stmt = selection2sql(original_conditions_categorical, original_conditions_numeric, df, connection, cursor,
                             table)
    cursor.execute(sql_stmt)
    sqlresult = cursor.fetchall()
    # TODO: how to handle index returned from db
    df_result = pd.DataFrame(sqlresult, columns=["index"] + df.columns.tolist())
    # see whether result satisfies the fairness constraints
    # fairness_constraints_no_less_than = [["race", "race3", 100]]
    for fc in fairness_constraints_no_less_than:
        values_assignment = list(set(df[fc[0]].unique()))
        for value in values_assignment:
            df_result_fc = df_result[(df_result[fc[0]] == value)]
            print("number {} = {}".format(value, len(df_result_fc)))


"""
format:
numeric_att_domain: [age: {10, 20, 30}, income: {50, 60, 70}]
categorical_att_domain: [state: MI, state: IL, state: WI, city: Detroit, city: AA]
"""


def LatticeTraversal(df, connection, cursor, table, original_conditions, original_conditions_categorical_att,
                     num_categorical_att, original_conditions_numeric_att, num_numeric_att,
                     categorical_att_domain_too_add,
                     numeric_att_domain_to_relax, fairness_constraints_no_less_than,
                     time_limit=5 * 60):
    original_conditions_categorical = {k: original_conditions[k] for k in original_conditions_categorical_att}
    original_conditions_numeric = {k: original_conditions[k] for k in
                                   original_conditions_numeric_att}
    # result in original conditions
    satisfy_original_fairness_constraints(df, connection, cursor, table, fairness_constraints_no_less_than,
                                          original_conditions_categorical, original_conditions_numeric)

    time1 = time.time()
    relaxed_conditions = []
    index_list = list(range(0, (len(categorical_att_domain_too_add) + num_numeric_att)))  # list[1, 2, ...13]
    for num_relaxations in range(1, len(categorical_att_domain_too_add) + num_numeric_att + 1):
        # print("----------------------------------------------------  num_att = ", num_att)
        have_results = False
        comb_num_att = list(
            combinations(index_list, num_relaxations))  # list of combinations of att index, length num_att
        for comb in comb_num_att:  # comb starts from 0
            if time.time() - time1 > time_limit:
                raise Exception("Lattice Traversal over time, with {} s timeout".format(time_limit))
            print("comb: {}".format(comb))
            if comb[num_relaxations - 1] < len(categorical_att_domain_too_add):  # no numeric
                values_to_add = list(categorical_att_domain_too_add[i] for i in comb)
                cond_after_relaxation = copy.deepcopy(original_conditions)
                for v in values_to_add:
                    cond_after_relaxation[v[0]].append(v[1])

                # cond_after_relaxation = list(categorical_att_domain_too_add[i] for i in comb) + original_conditions
                if satisfy_fairness_constraints(df, connection, cursor, table, fairness_constraints_no_less_than,
                                                cond_after_relaxation, original_conditions_numeric):
                    relaxed_conditions.append(cond_after_relaxation)
                    have_results = True
            else:  # have numeric
                index_start_numeric = 0
                for index_start_numeric in range(len(comb)):
                    if comb[index_start_numeric] >= len(categorical_att_domain_too_add):
                        break
                categorical_cond_after_relaxation = list(
                    categorical_att_domain_too_add[i] for i in comb[:index_start_numeric])
                numeric_indices = comb[index_start_numeric:]
                all_numeric_cond = [
                    []]  # [] * numeric_att_domain_to_relax[numeric_att_domain_to_relax.keys()[0]][0]
                for idx in numeric_indices:
                    idx_numeric_att_domain = idx - len(categorical_att_domain_too_add)
                    att = numeric_att_domain_to_relax.keys()[idx_numeric_att_domain]
                    for num_relaxed in numeric_att_domain_to_relax[att]:
                        new_all_numeric_cond = []
                        for nc in all_numeric_cond:
                            nc[att] = num_relaxed
                            new_all_numeric_cond.append(nc)
                        all_numeric_cond = new_all_numeric_cond
                for numeric_c in all_numeric_cond:
                    c = categorical_cond_after_relaxation + numeric_c
                    if satisfy_fairness_constraints(df, connection, cursor, table, fairness_constraints_no_less_than,
                                                    c):
                        relaxed_conditions.append(c)
                        have_results = True
        if have_results:
            break
    return relaxed_conditions


file = r"../InputData/Pipelines/healthcare/before_selection.csv"
df = pd.read_csv(file, keep_default_na=False)
print(df[:4])
original_conditions = dict()
original_conditions["county"] = ["county2", "county3"]
original_conditions["complications"] = 5

original_conditions_categorical_att = ["county"]
num_categorical_att = len(original_conditions_categorical_att)

original_conditions_numeric_att = ['complications']
num_numeric_att = len(original_conditions_numeric_att)

categorical_att_domain_too_add = []
numeric_att_domain_to_relax = dict()

for att in original_conditions_categorical_att:
    values_assignment = list(set(df[att].unique()) - set(original_conditions[att]))
    for value in values_assignment:
        print(value, type(value))
        categorical_att_domain_too_add.append((att, value))
print("categorical_att_domain_too_add", categorical_att_domain_too_add)

for att in original_conditions_numeric_att:
    numeric_att_domain_to_relax[att] = []
    unique_values = df[att].unique().tolist()
    unique_values.sort()  # descending
    domain = unique_values[::-1]
    for idx_domain in range(len(domain)):
        if domain[idx_domain] < original_conditions[att]:
            numeric_att_domain_to_relax[att] = domain[idx_domain:]
            break
print("numeric_att_domain_to_relax", numeric_att_domain_to_relax)

# TODO: expression for fairness constraints,
# TODO: add no more than constraints
fairness_constraints_no_less_than = [["race", "race3", 50]]

# To connect MySQL database
conn = pymysql.connect(
    host=host,
    user=user,
    password=passwd,
    db=db,
)

# preparing a cursor object
cur = conn.cursor()
"""
def LatticeTraversal(df, connection, cursor, original_conditions, categorical_att_domain_too_add,
                     numeric_att_domain_to_relax,
                     num_categorical_att,
                     num_numeric_att, fairness_constraints, time_limit=5 * 60)
"""
relaxed_conditions = LatticeTraversal(df, conn, cur, table, original_conditions, original_conditions_categorical_att,
                                      num_categorical_att, original_conditions_numeric_att, num_numeric_att,
                                      categorical_att_domain_too_add,
                                      numeric_att_domain_to_relax, fairness_constraints_no_less_than,
                                      time_limit=5 * 60)

print("\nrelaxed_conditions: \n")
print(relaxed_conditions)

conn.close()
