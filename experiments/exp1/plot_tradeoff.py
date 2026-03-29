import matplotlib.pyplot as plt
import os

# -------------------------
# DATA (YOUR RESULTS)
# -------------------------

models = ["Transformer", "Mamba", "Soft Hybrid", "Sparse Hybrid"]

latencies = [0.3151, 0.1722, 0.1401, 0.1954]
losses = [1.0145, 0.1202, 0.0867, 0.4850]

# -------------------------
# PLOT
# -------------------------

plt.figure()
for i, name in enumerate(models):
    if "Sparse" in name:
        plt.scatter(latencies[i], losses[i], marker='*', s=200)
    else:
        plt.scatter(latencies[i], losses[i])

# Annotate points
for i, name in enumerate(models):
    plt.text(latencies[i] + 0.002, losses[i] + 0.02, name)

plt.xlabel("Latency (seconds)")
plt.ylabel("Loss")
plt.title("Performance vs Efficiency Tradeoff")

plt.grid()

# -------------------------
# SAVE
# -------------------------

save_path = os.path.join(os.path.dirname(__file__), "tradeoff_plot.png")
plt.savefig(save_path)