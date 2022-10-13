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

from Algorithm import ProvenanceSearch_3_20220324 as ps
from Algorithm import LatticeTraversal_2_2022405 as lt




data_file = r"../../InputData/Pipelines/healthcare/before_selection.csv"
selection_file = r"../../InputData/Pipelines/healthcare/incomeK/relaxation/selection1.json"

minimal_refinements1, running_time1 = ps.FindMinimalRefinement(data_file, selection_file)

minimal_refinements2, running_time2 = lt.FindMinimalRefinement(data_file, selection_file)





