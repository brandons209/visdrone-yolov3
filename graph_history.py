import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_graph(x, y, x_label, y_label, label, title, markersize):
    plt.title(title)
    plt.plot(x, y, linewidth=2, label=label, marker='o', markersize=markersize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)

parser = argparse.ArgumentParser()
parser.add_argument("--history_path", type=str, default="data/saved_weights/history.pkl", help="path to history pickle file")

args = parser.parse_args()

with open(args.history_path, 'rb') as f:
    history = pickle.load(f)

losses = {}
loss_names = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]
for name in loss_names:
    current_loss = []
    for e in range(len(history)):
        current_loss.append(history[e][name])
    losses[name] = current_loss

epochs = np.arange(len(history))
#bounding boxes
plt.figure(figsize=(15,15))
for loss in loss_names[:4]:
    plot_graph(epochs, losses[loss], "Epochs", "Loss", loss, "Bounding Box Lossess", 1)
    plt.xticks(np.arange(0, len(history), 5))
    plt.yticks(np.linspace(0, 1.5, num=15))
    plt.legend(title="Losses")
plt.tight_layout()
plt.show()

#object confidence and class
plt.figure(figsize=(15,15))
for loss in loss_names[4:6]:
    plot_graph(epochs, losses[loss], "Epochs", "Loss", loss, "Conf/Class Lossess", 1)
    plt.xticks(np.arange(0, len(history), 5))
    plt.yticks(np.linspace(0, 1, num=15))
    plt.legend(title="Losses")
plt.tight_layout()
plt.show()

#recall and precision
plt.figure(figsize=(15,15))
for loss in loss_names[6:]:
    plot_graph(epochs, losses[loss], "Epochs", "Loss", loss, "Recall and Precision", 1)
    plt.xticks(np.arange(0, len(history), 5))
    plt.yticks(np.linspace(0, 1, num=15))
    plt.legend(title="Accuracies")
plt.tight_layout()
plt.show()
