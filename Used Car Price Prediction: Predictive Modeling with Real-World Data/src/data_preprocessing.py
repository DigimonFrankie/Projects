import numpy as np
import pandas as pd
import os
import re
import pickle
from collections import defaultdict, Counter

class DataPreprocessor:
    def __init__(self, df, assets_dir):
        self.df = df.copy()
        self.assets_dir = assets_dir
