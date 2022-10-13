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

from Algorithm import ProvenanceSearch_10_20220421 as ps
from Algorithm import LatticeTraversal_2_2022405 as lt

minimal_refinements1 = []
minimal_added_refinements1 = []
running_time1 = []

minimal_refinements2 = []
minimal_added_refinements2 = []
running_time2 = []


output_path = r'running_time_income_0.csv'
output_file = open(output_path, "w")

output_file.write("running_time1,running_time2\n")


for i in range(1, 7):

    data_file = r"../../InputData/Pipelines/healthcare/incomeK/before_selection_incomeK.csv"
    selection_file = r"../../InputData/Pipelines/healthcare/incomeK/income_change/selection" + str(i) + ".json"

    print("========================== provenance search ===================================")
    minimal_refinements1_, minimal_added_refinements1_, running_time1_ = ps.FindMinimalRefinement(data_file, selection_file)
    minimal_refinements1.append(minimal_refinements1_)
    minimal_added_refinements1.append(minimal_added_refinements1_)
    running_time1.append(running_time1_)

    print("running time = {}".format(running_time1))

    print("========================== lattice traversal ===================================")

    minimal_refinements2_, minimal_added_refinements2_, running_time2_ = lt.FindMinimalRefinement(data_file, selection_file)
    minimal_refinements2.append(minimal_refinements2_)
    minimal_added_refinements2.append(minimal_added_refinements2_)
    running_time2.append(running_time2_)

    print("running time = {}".format(running_time2))

    print(*minimal_refinements1, sep="\n")

    output_file.write("{},{}\n".format(running_time1_, running_time2_))





