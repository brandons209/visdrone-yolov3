import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_graph(x, y, x_label, y_label, label, title, markersize):
    plt.title(title)
    plt.plot(x, y, linewidth=2, label=label, marker='o', markersize=markersize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
