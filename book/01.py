import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
# %matplotlib inline

import seaborn as sns
sns.set(style='darkgrid')

cols = ['price', 'maint', 'doors', 'persons', 'lug_capacity', 'safety', 'output']
cars = pd.read_csv()