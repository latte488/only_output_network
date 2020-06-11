import argparse
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str, default='data.csv')
parser.add_argument('--png_path', type=str, default='image.png')
args = parser.parse_args()

losses = []
accuracys = []
with open(args.csv_path) as f:
    for column in csv.reader(f):
        column = [float(row) for row in column]
        losses.append(column[0])
        accuracys.append(column[1])
epoch = [i + 1 for i in range(len(losses))]
plt.figure()
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot(epoch, losses, label='loss')
plt.plot(epoch, accuracys, label='accuracy')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend()
plt.savefig(args.png_path)
