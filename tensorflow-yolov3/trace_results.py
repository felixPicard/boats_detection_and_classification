import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

files_directory = "csvFiles/saves/"
files_identity = "pandas"
files_names = ["0797.csv", "0799.csv", "0801.csv"]
colors = ["r", "g", "b"]
mode = "."

plt.figure(0)
f_scores = np.linspace(0.2, 0.8, num=4)
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

for i in range(len(files_names)):

    file_path = files_directory + files_identity + files_names[i]
    current_table = pd.read_csv(file_path, sep="\t")

    plt.figure(0)
    plt.plot(current_table["Recall"], current_table["Precision"], colors[i]+mode, label=files_names[i])


    plt.figure(1)
    plt.plot(current_table["Threshold"], current_table["f1_score"], colors[i]+mode, label=files_names[i])

plt.figure(0)
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()

plt.figure(1)
plt.ylim(0, 1)
plt.xlim(0 ,1)
plt.xlabel("Threshold")
plt.ylabel("f1_score")
plt.legend()

plt.show()