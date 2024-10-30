# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# %%
pd.set_option("display.max_columns", 500)

train = pd.read_csv(
    r"C:\Users\ragha\Desktop\Competition\child-mind-institute-problematic-internet-use\train.csv"
)
test = pd.read_csv(
    r"C:\Users\ragha\Desktop\Competition\child-mind-institute-problematic-internet-use\test.csv"
)

data_dictionary = pd.read_csv(
    r"C:\Users\ragha\Desktop\Competition\child-mind-institute-problematic-internet-use\data_dictionary.csv"
)

# %%
