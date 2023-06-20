from __future__ import print_function

from Algorithm.ProvenanceSearchValues import searchPVT_refinement, whether_satisfy_fairness_constraints, \
    subtract_provenance, searchPVT_relaxation, build_PVT_contract_only, build_PVT_relax_only, searchPVT_contraction, \
    build_PVT_refinement

import sys
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass
from pympler import asizeof
import copy
from typing import List, Any
import numpy as np
import pandas as pd
import time
from intbitset import intbitset
import json


def FindMinimalRefinement(data_file, query_info, constraint_info, time_limit=5 * 60):
    time1 = time.time()
    global assign_to_provenance_num
    assign_to_provenance_num = 0
    data = pd.read_csv(data_file, index_col=False)

    numeric_attributes = []
    categorical_attributes = {}
    selection_numeric_attributes = {}
    selection_categorical_attributes = {}
    if 'selection_numeric_attributes' in query_info:
        selection_numeric_attributes = query_info['selection_numeric_attributes']
        numeric_attributes = list(selection_numeric_attributes.keys())
    if 'selection_categorical_attributes' in query_info:
        selection_categorical_attributes = query_info['selection_categorical_attributes']
        categorical_attributes = query_info['categorical_attributes']
    selected_attributes = numeric_attributes + [x for x in categorical_attributes]
    print("selected_attributes", selected_attributes)

    sensitive_attributes = constraint_info['all_sensitive_attributes']
    fairness_constraints = constraint_info['fairness_constraints']

    pd.set_option('display.float_format', '{:.2f}'.format)

    print(f"data={data},\n selected_attributes={selected_attributes},\n sensitive_attributes={sensitive_attributes}")

    if whether_satisfy_fairness_constraints(data, selected_attributes, sensitive_attributes, fairness_constraints,
                                            numeric_attributes, categorical_attributes, selection_numeric_attributes,
                                            selection_categorical_attributes):
        print("original query satisfies constraints already")
        return {}, time.time() - time1, assign_to_provenance_num, 0, 0
    fairness_constraints_provenance_greater_than, fairness_constraints_provenance_smaller_than, \
    data_rows_greater_than, data_rows_smaller_than, only_greater_than, only_smaller_than, contraction_threshold \
        = subtract_provenance(data, selected_attributes, sensitive_attributes, fairness_constraints,
                              numeric_attributes, categorical_attributes, selection_numeric_attributes,
                              selection_categorical_attributes)
    time_provenance2 = time.time()
    provenance_time = time_provenance2 - time1
    # print("provenance_expressions")
    # print(*fairness_constraints_provenance_greater_than, sep="\n")
    # print(*fairness_constraints_provenance_smaller_than, sep="\n")
    if time.time() - time1 > time_limit:
        print("time out")
        return [], time.time() - time1, assign_to_provenance_num, provenance_time, 0

    if only_greater_than:
        time_table1 = time.time()
        PVT, PVT_head, categorical_att_columns, max_index_PVT = build_PVT_relax_only(data, selected_attributes,
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
        print(PVT)
        # print("max_index_PVT: {}".format(max_index_PVT))
        time_table2 = time.time()
        table_time = time_table2 - time_table1
        time_search1 = time.time()
        if time.time() - time1 > time_limit:
            print("time out")
            return [], time.time() - time1, assign_to_provenance_num, provenance_time, 0

        checked_assignments_satisfying = []
        checked_assignments_unsatisfying = []
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
        # print("table time = {}".format(table_time))
        print("searching time = {}".format(time2 - time_table1))
        # print("minimal_added_relaxations:{}".format(minimal_added_refinements))
        return minimal_refinements, time2 - time1, assign_to_provenance_num, provenance_time, time2 - time_search1

    elif only_smaller_than:
        time_table1 = time.time()
        PVT, PVT_head, categorical_att_columns, \
        max_index_PVT = build_PVT_contract_only(data, selected_attributes, numeric_attributes,
                                                categorical_attributes,
                                                selection_numeric_attributes,
                                                selection_categorical_attributes,
                                                sensitive_attributes,
                                                fairness_constraints,
                                                fairness_constraints_provenance_greater_than,
                                                fairness_constraints_provenance_smaller_than,
                                                data_rows_greater_than,
                                                data_rows_smaller_than)
        # print("max_index_PVT: {}".format(max_index_PVT))
        time_table2 = time.time()
        table_time = time_table2 - time_table1
        # print("delta table:\n{}".format(delta_table))
        time_search1 = time.time()
        if time.time() - time1 > time_limit:
            print("time out")
            return [], time.time() - time1, assign_to_provenance_num, provenance_time, 0

        checked_assignments_satisfying = []
        checked_assignments_unsatisfying = []

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
        # print("table time = {}".format(table_time))
        print("searching time = {}".format(time2 - time_table1))
        # print("minimal_added_relaxations:{}".format(minimal_added_refinements))
        return minimal_refinements, time2 - time1, assign_to_provenance_num, provenance_time, time2 - time_search1

    time_table1 = time.time()

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
                                                                data_rows_greater_than,
                                                                data_rows_smaller_than, contraction_threshold)
    # print("max_index_PVT: {}".format(max_index_PVT))
    time_table2 = time.time()
    table_time = time_table2 - time_table1
    # print("delta table:\n{}".format(delta_table))
    time_search1 = time.time()
    if time.time() - time1 > time_limit:
        print("time out")
        return [], time.time() - time1, assign_to_provenance_num, provenance_time, 0

    checked_assignments_satisfying = []
    checked_assignments_unsatisfying = []
    minimal_refinements = searchPVT_refinement(PVT, PVT_head, possible_values_lists, numeric_attributes,
                                               categorical_attributes, selection_numeric_attributes,
                                               selection_categorical_attributes, len(PVT_head),
                                               fairness_constraints_provenance_greater_than,
                                               fairness_constraints_provenance_smaller_than, PVT, PVT_head,
                                               max_index_PVT,
                                               checked_assignments_satisfying,
                                               checked_assignments_unsatisfying, time_limit)
    time2 = time.time()
    print("provenance time = {}".format(provenance_time))
    # print("table time = {}".format(table_time))
    print("searching time = {}".format(time2 - time_table1))
    print("assign_to_provenance_num = {}".format(assign_to_provenance_num))
    return minimal_refinements, time2 - time1, assign_to_provenance_num, provenance_time, time2 - time_search1
