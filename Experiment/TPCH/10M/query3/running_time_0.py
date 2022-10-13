"""
executable
without optimizations
"""

import copy
from typing import List, Any
import numpy as np
import pandas as pd
import mysql.connector
import time
from intbitset import intbitset
import json

from Algorithm import ProvenanceSearch_10_20220421 as ps
from Algorithm import LatticeTraversal_4_2022427 as lt

host_args = {
    "host": "localhost",
    "user": "root",
    "password": "ljy19980228"
}


sql_connection = mysql.connector.connect(**host_args)


sql_filename = 'tpch10M-q3-param.sql'
sql_file = open(sql_filename, 'r')
sql_file_read = sql_file.read()
sql_file.close()


sql_getinfo_filename = 'tpch10M-q3-getinfo.sql'
sql_getinfo_file = open(sql_getinfo_filename, 'r')
sql_getinfo_file_read = sql_file.read()
sql_getinfo_file.close()


data_file = r"../../InputData/Pipelines/healthcare/incomeK/before_selection_incomeK.csv"
selection_file = r"../../InputData/Pipelines/healthcare/incomeK/selection4.json"

data = pd.read_csv(data_file)

print("========================== provenance search ===================================")
minimal_refinements1, minimal_added_refinements1, running_time1 = \
    ps.FindMinimalRefinement(data, selection_file)

print("running time = {}".format(running_time1))


print("========================== lattice traversal ===================================")

minimal_refinements2, minimal_added_refinements2, running_time2 = \
    lt.FindMinimalRefinement(data, selection_file, sql_file_read, sql_getinfo_file, sql_connection)

print("running time = {}".format(running_time2))


print(*minimal_refinements1, sep="\n")


