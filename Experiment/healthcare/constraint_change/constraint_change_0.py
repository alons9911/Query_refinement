"""
executable
without optimizations
"""
import copy
from typing import List, Any
import numpy as np
import pandas as pd
import time
from intbitset import intbitset
import json

from Algorithm import ProvenanceSearchValues_6_20220825 as ps
from Algorithm import LatticeTraversal_2_2022405 as lt


minimal_refinements1 = []
minimal_added_refinements1 = []
running_time1 = []

minimal_refinements2 = []
minimal_added_refinements2 = []
running_time2 = []


data_file = r"../../../InputData/Pipelines/healthcare/incomeK/before_selection_incomeK.csv"
query_file = r"../../../InputData/Pipelines/healthcare/incomeK/constraint_change/query1.json"
constraint_file_prefix = r"../../../InputData/Pipelines/healthcare/incomeK/constraint_change/constraint"





time_output_file = r"./constraint_change_0.csv"
time_output = open(time_output_file, "w")
time_output.write("income,PS,LT\n")

result_output_file = r"./result_1.txt"
result_output = open(result_output_file, "w")
result_output.write("selection file, result\n")



for i in range(1, 11):

    constraint_file = constraint_file_prefix + str(i) + ".json"

    print("========================== provenance search ===================================")
    minimal_refinements1, running_time1, assign_to_provenance_num, \
    provenance_time, search_time = \
        ps.FindMinimalRefinement(data_file, query_file, constraint_file)

    print("running time = {}".format(running_time1))

    print("========================== lattice traversal ===================================")

    minimal_refinements2, minimal_added_refinements2, running_time2 = \
        lt.FindMinimalRefinement(data_file, query_file, constraint_file)

    print("running time = {}".format(running_time2))


    print(*minimal_refinements1, sep="\n")

    result_output.write("\n")
    idx = i * 50
    time_output.write("{},{:0.2f},{:0.2f},{:0.2f}\n".format(idx,  provenance_time, search_time, running_time2))
    # time_output.write("{}, {:0.2f}, {:0.2f}\n".format(idx, running_time1, running_time2))
    result_output.write("{}\n".format(idx))
    result_output.write(", ".join(str(item) for item in minimal_added_refinements1))
    result_output.write("\n")
    result_output.write("\n".join(str(item) for item in minimal_refinements1))
    result_output.write("\n")




