# utils.py
# ==============================
# Title: ADA Project Data Processing Utilities
# ==============================

import pandas as pd
import os
import matplotlib.pyplot as plt
import re
import numpy as np

def calculate_weighted_avg(row, global_mean,m):
    n = row['nbr_ratings']
    avg = row['avg']
    return np.round(((avg*n)+ (global_mean*m))/(n+m),3)

