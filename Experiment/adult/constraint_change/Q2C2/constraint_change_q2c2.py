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

from Algorithm import ProvenanceSearchValues_4_20220712_global as ps
from Algorithm import LatticeTraversal_2_2022405 as lt


minimal_refinements1 = []
minimal_added_refinements1 = []
running_time1 = []

minimal_refinements2 = []
minimal_added_refinements2 = []
running_time2 = []


data_file = r"../../../../InputData/Adult/adult.data"

constraint_file_prefix = r"constraint2"
time_limit = 5*60


def run_query(q):
    query_file = r"query" + str(q) + ".json"
    time_output_file = r"./constraint_change_q2c2.csv"
    time_output = open(time_output_file, "w")
    time_output.write("income,PS,LT\n")

    result_output_file = r"./result_q2c2.txt"
    result_output = open(result_output_file, "w")
    result_output.write("selection file, result\n")

    for i in range(1, 6):

        constraint_file = constraint_file_prefix + str(i) + ".json"

        print("========================== provenance search ===================================")
        minimal_refinements1, running_time1, assign_to_provenance_num = \
            ps.FindMinimalRefinement(data_file, query_file, constraint_file, time_limit)

        print("running time = {}".format(running_time1))
        print(*minimal_refinements1, sep="\n")
        # print("========================== lattice traversal ===================================")
        #
        # minimal_refinements2, minimal_added_refinements2, running_time2 = \
        #     lt.FindMinimalRefinement(data_file, query_file, constraint_file, time_limit)
        # print("running time = {}".format(running_time2))
        # if running_time2 < time_limit:
        #     print(*minimal_refinements2, sep="\n")
        running_time2 = 0
        result_output.write("\n")
        idx = 50 + i * 50
        time_output.write("{},{:0.2f},{:0.2f},{}\n".format(idx, running_time1, running_time2, assign_to_provenance_num))
        result_output.write("{}\n".format(idx))
        result_output.write(",".join(str(item) for item in minimal_added_refinements1))
        result_output.write("\n")
        result_output.write("\n".join(str(item) for item in minimal_refinements1))
        result_output.write("\n")
    result_output.close()
    time_output.close()



run_query(2)