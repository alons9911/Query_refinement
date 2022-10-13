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
from Algorithm import LatticeTraversal_4_20220901 as lt


minimal_refinements1 = []
minimal_added_refinements1 = []
# running_time1 = []
#
minimal_refinements2 = []
minimal_added_refinements2 = []
# running_time2 = []


time_limit = 5*60


time_output_file = r"time.csv"
time_output = open(time_output_file, "w")
time_output.write("data size,running time,provenance time,search time\n")

result_output_file = r"result.txt"
result_output = open(result_output_file, "w")
result_output.write("selection file, result\n")

data_size = ["100", "200", "400", "800"]

for i in range(1, 5):
    query_file = r"../../../InputData/Healthcare/incomeK/scale_2/query" + str(i) + ".json"
    constraint_file = r"../../../InputData/Healthcare/incomeK/scale_2/constraint" + str(i) + ".json"
    data_file = "../../../InputData/Healthcare/incomeK/scale_2/healthcare_" + data_size[i-1] + ".csv"

    print(data_file, constraint_file, query_file)

    print("========================== provenance search ===================================")
    minimal_refinements1, running_time1, assign_to_provenance_num,\
        provenance_time, search_time = \
        ps.FindMinimalRefinement(data_file, query_file, constraint_file, time_limit)
    print("running time = {}".format(running_time1), provenance_time, search_time)
    print(*minimal_refinements1, sep="\n")

    print("========================== lattice traversal ===================================")
    minimal_refinements2, minimal_added_refinements2, running_time2 = \
        lt.FindMinimalRefinement(data_file, query_file, constraint_file, time_limit)
    print("running time = {}".format(running_time2))
    if running_time2 < time_limit:
        print(*minimal_refinements2, sep="\n")
    running_time2 = 0
    result_output.write("\n")

    time_output.write("{},{:0.2f},{:0.2f},{:0.2f},{:0.2f}\n".format(2**(i-1), running_time1, provenance_time,
                                                                    search_time, running_time2))
    result_output.write("{}\n".format(2**(i-1)))
    result_output.write(",".join(str(item) for item in minimal_added_refinements1))
    result_output.write("\n")
    result_output.write("\n".join(str(item) for item in minimal_refinements1))
    result_output.write("\n")
result_output.close()
time_output.close()
