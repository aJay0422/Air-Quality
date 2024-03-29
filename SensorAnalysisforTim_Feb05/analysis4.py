import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import os
import argparse

from matplotlib.widgets import RectangleSelector
from utils import average_hour, corrcoef_nan

# Create a parser
parser = argparse.ArgumentParser(description='Time Series Analysis')
# Add arguments
parser.add_argument('--idx1', type=int, default=4)
parser.add_argument('--idx2', type=int, default=5)
parser.add_argument('--month', type=str, default='Oct')
# Parse the arguments
args = parser.parse_args()

month2slice = {'Oct': slice(0, 31*24),
               'Nov': slice(31*24, 61*24),
               'Dec': slice(61*24, 92*24),
               'Jan': slice(92*24, 122*24)}


# Prepare data
data_dir = "../InterpolationBaseline/data/Oct0123_Jan3024/"

area1_ids = ['JAJWJnroQCSJz0Dr9uVC1g', '_e49pbOSQseqTE5lu-6NMA']
area2_ids = ['pCPex6DkSdS0f5K2f7jyHg', 'xudEmbncQ7iqwy3sZ0jZvQ']
area3_ids = ['6nBLCf6WT06TOuUExPkBtA', 'JKiLhziTQ4eiYHQq3x01uw']

df_full = pd.read_csv(os.path.join(data_dir, area1_ids[0] + ".csv"))
df_full = average_hour(df_full)
df_full = df_full.loc[:, ['month', 'day', 'hour']]

area1_dfs = []
area2_dfs = []
area3_dfs = []

for id in area1_ids:
    df = pd.read_csv(data_dir + id + ".csv")
    df = average_hour(df)
    if len(df) == 24 * 122:
        print("Length correct")
    else:
        df = df_full.merge(df, on=['month', 'day', 'hour'], how='left')
        print("Length incorrect")
    area1_dfs.append(df)

for id in area2_ids:
    df = pd.read_csv(data_dir + id + ".csv")
    df = average_hour(df)
    if len(df) == 24 * 122:
        print("Length correct")
    else:
        df = df_full.merge(df, on=['month', 'day', 'hour'], how='left')
        print("Length incorrect")
    area2_dfs.append(df)

for id in area3_ids:
    df = pd.read_csv(data_dir + id + ".csv")
    df = average_hour(df)
    if len(df) == 24 * 122:
        print("Length correct")
    else:
        df = df_full.merge(df, on=['month', 'day', 'hour'], how='left')
        print("Length incorrect")
    area3_dfs.append(df)
all_dfs = area1_dfs + area2_dfs + area3_dfs

# Select the data
slc = month2slice[args.month]
seq1 = all_dfs[args.idx1]["pm25"][slc]
seq2 = all_dfs[args.idx2]["pm25"][slc]



# define a callback function for period selection
def oneselect(eclick, erelease):
    if eclick.dblclick:
        print("Reset selction")
        return
    
    x1, x2 = int(eclick.xdata), int(erelease.xdata)
    if x1 < 0:
        x1 = 0
    elif x1 >= len(seq1):
        x1 = len(seq1) - 1
    if x2 < 0:
        x2 = 0
    elif x2 >= len(seq1):
        x2 = len(seq1) - 1
    start, end = np.sort([x1, x2])
    print(start, end)
    selected_seq1 = seq1[start:end]
    selected_seq2 = seq2[start:end]
    # print(start, end)
    # print(selected_seq1, selected_seq2)
    if len(selected_seq1) != 0 and len(selected_seq2) != 0:
        corr = corrcoef_nan(selected_seq1, selected_seq2)[0, 1]
        ax.set_title("Correlation: {:.2f}".format(corr), fontsize=20)
        fig.canvas.draw_idle()
        # print(f"Correlation: {corr}")
    else:
        corr = corrcoef_nan(seq1, seq2)[0, 1]
        ax.set_title("Global Correlation: {:.2f}".format(corr), fontsize=20)
        fig.canvas.draw_idle()

fig, ax = plt.subplots()
label1 = 'Sensor {}'.format(args.idx1)
label2 = 'Sensor {}'.format(args.idx2)
ax.plot(np.arange(len(seq1)), seq1, label=label1)
ax.plot(np.arange(len(seq2)), seq2, label=label2)
ax.legend()

rect_selector = RectangleSelector(ax, oneselect, useblit=True,
                                  button=[1], # left mouse button,
                                  minspanx=5, minspany=5,
                                  spancoords='pixels',
                                  interactive=True)

plt.show()
