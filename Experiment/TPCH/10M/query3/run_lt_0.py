"""
executable
without optimizations
"""

import copy
from typing import List, Any
import numpy as np
import pandas as pd
import pymysql
import time
from intbitset import intbitset
import json

from Algorithm import LatticeTraversal_4_2022427 as lt

host_args = {
    "host": "localhost",
    "user": "root",
    "password": "ljy19980228",
    'db': 'tpch'
}


sql_connection = pymysql.connect(**host_args)


cursor = sql_connection.cursor()


sql_filename = 'tpch10M-q3-param.sql'
sql_file = open(sql_filename, 'r')
sql_file_read = sql_file.read()
sql_file.close()


sql_getinfo_filename = 'tpch10M-q3-getinfo.sql'
sql_getinfo_file = open(sql_getinfo_filename, 'r')
sql_getinfo_file_read = sql_getinfo_file.read()
sql_getinfo_file.close()


# params = {"att": "o_orderdate"}
# cursor = sql_connection.cursor()
# cursor.execute(sql_getinfo_file, params)
# fetch_result = cursor.fetchall()  # ascending
# print(fetch_result)

selection_file = r"selection1.json"
answer_column_file = r"columns_q3.csv"


print("========================== lattice traversal ===================================")

minimal_refinements2, minimal_added_refinements2, running_time2 = \
    lt.FindMinimalRefinement(selection_file, sql_file_read, sql_getinfo_file_read, sql_connection, answer_column_file)

print("running time = {}".format(running_time2))


print(*minimal_refinements2, sep="\n")




