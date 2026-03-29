import matplotlib.pyplot as plt
import os

block_sizes = [2, 8, 16]
latencies = [0.6460, 0.1954, 0.1959]

plt.figure()
plt.plot(block_sizes, latencies, marker='o')

plt.xlabel("Block Size")
plt.ylabel("Latency (seconds)")
plt.title("Latency vs Block Size")

plt.grid()

save_path = os.path.join(os.path.dirname(__file__), "latency_plot.png")
plt.savefig(save_path)