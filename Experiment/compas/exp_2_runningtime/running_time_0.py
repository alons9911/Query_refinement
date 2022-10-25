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

from Algorithm import ProvenanceSearchValues as ps
from Algorithm import Baseline as lt

data_file = r"../../../InputData/Compas/compas-scores.csv"
query_file_prefix = r"query"
constraint_file_prefix = r"constraint"

time_output_prefix = r"./result_"


def file(q, c):
    time_output_file = time_output_prefix + str(q) + str(c) + ".csv"
    time_output = open(time_output_file, "w")
    time_output.write("selection file, running time ps, running time lt\n")
    return time_output


time_limit = 60 * 5


def compare(q, c, time_output):
    print("run with query{} constraint{}".format(q, c))
    query_file = query_file_prefix + str(q) + ".json"
    constraint_file = constraint_file_prefix + str(c) + ".json"

    print("========================== provenance search ===================================")
    minimal_refinements1, running_time1, assign_to_provenance_num, \
    provenance_time, search_time = \
        ps.FindMinimalRefinement(data_file, query_file, constraint_file, time_limit)

    print("running time = {}".format(running_time1))
    print(*minimal_refinements1, sep="\n")

    # print("========================== lattice traversal ===================================")
    #
    # minimal_refinements2, minimal_added_refinements2, running_time2 = \
    #     lt.FindMinimalRefinement(data_file, query_file, constraint_file, time_limit)
    # if running_time2 > time_limit:
    #     print("naive alg out of time")
    # else:
    #     print("running time = {}".format(running_time2))
    #     print(*minimal_refinements2, sep="\n")

    time_output.write("\n")
    idx = "Q" + str(q) + "C" + str(c)
    time_output.write("{},{:0.2f},{:0.2f},{:0.2f}\n".format(idx, running_time1, provenance_time,
                                                            search_time))
    # if running_time2 < time_limit:
    #     time_output.write("{}, {:0.2f}\n".format(idx, running_time2))
    time_output.write("\n".join(str(item) for item in minimal_refinements1))
    time_output.write("\n")
    summary_file.write(("{},{:0.2f},".format(idx, running_time1)))
    # if running_time2 < time_limit:
    #     summary_file.write("{:0.2f}\n".format(running_time2))
    # else:
    #     summary_file.write("\n")


summary_file = open(r"time.csv", "w")
summary_file.write("file,PS,LT\n")


def run(q, c):
    time_output = file(q, c)
    compare(q, c, time_output)
    time_output.close()


# run(1, 1)
# run(1, 2)
# run(1, 3)
# run(2, 1)
# run(2, 2)
# run(2, 3)

summary_file.close()
