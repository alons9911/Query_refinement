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

from Algorithm import ProvenanceSearch_15_20220429 as ps
from Algorithm import LatticeTraversal_2_2022405 as lt







data_file = r"../../../InputData/Pipelines/adult/adult_preprocessed.csv"
query_file_prefix = r"../../../InputData/Pipelines/adult/relaxation/query"
constraint_file_prefix = r"../../../InputData/Pipelines/adult/relaxation/constraint"

time_output_file = r"./time_ps_1.csv"
time_output = open(time_output_file, "w")
time_output.write("selection file, running time ps, running time lt\n")

result_output_file = r"./result_ps_1.txt"
result_output = open(result_output_file, "w")
result_output.write("selection file, result\n")


def compare(q, c):
    query_file = query_file_prefix + str(q) + ".json"
    constraint_file = constraint_file_prefix + str(c) + ".json"

    print("========================== provenance search ===================================")
    minimal_refinements1, minimal_added_refinements1, running_time1 = \
        ps.FindMinimalRefinement(data_file, query_file, constraint_file)

    print("running time = {}".format(running_time1))

    # print("========================== lattice traversal ===================================")
    #
    # minimal_refinements2, minimal_added_refinements2, running_time2 = \
    #     lt.FindMinimalRefinement(data_file, query_file, constraint_file)
    #
    # print("running time = {}".format(running_time2))


    print(*minimal_refinements1, sep="\n")

    result_output.write("\n")
    idx = "q" + str(q) + "c" + str(c)
    time_output.write("{}, {:0.2f}\n".format(idx, running_time1))
    result_output.write("{}\n".format(idx))
    result_output.write(", ".join(str(item) for item in minimal_added_refinements1))
    result_output.write("\n")
    result_output.write("\n".join(str(item) for item in minimal_refinements1))
    result_output.write("\n")

compare(2, 1)
compare(2, 2)
compare(2, 3)


result_output.close()
time_output.close()
