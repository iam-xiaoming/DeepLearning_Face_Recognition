import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image


# ----------------------------------------------------------
# Tính similarity, tạo label dự đoán cho từng cặp
# ----------------------------------------------------------
def eval_pairs(embs, labels, generate_pairs_fn, 
               max_per_class=50, threshold=0.2):

    embs = torch.cat(embs).cpu()
    labels = torch.cat(labels).cpu().numpy()

    pairs = np.array(generate_pairs_fn(labels, max_per_class))
    idx_a = pairs[:, 0].astype(int)
    idx_b = pairs[:, 1].astype(int)

    sim = torch.sum(embs[idx_a] * embs[idx_b], dim=1).numpy()
    preds = (sim >= threshold).astype(int)

    return preds, pairs, sim


def sample_correct_pairs(pairs, preds, sim_scores, dataset, n=10):
    result = []

    while len(result) < n:
        i = np.random.randint(0, len(pairs))
        idx1, idx2, lbl = pairs[i]

        if preds[i] != lbl:
            continue

        path1, _ = dataset.samples[idx1]
        path2, _ = dataset.samples[idx2]
        result.append((path1, path2, lbl, sim_scores[i]))

    return result


def sample_incorrect_pairs(pairs, preds, sim_scores, dataset, n=10):
    result = []

    while len(result) < n:
        i = np.random.randint(0, len(pairs))
        idx1, idx2, lbl = pairs[i]

        if preds[i] == lbl:
            continue

        path1, _ = dataset.samples[idx1]
        path2, _ = dataset.samples[idx2]
        result.append((path1, path2, lbl, sim_scores[i]))

    return result


# ----------------------------------------------------------
# Vẽ ảnh các cặp
# ----------------------------------------------------------
def show_pairs(paths, pairs_per_row=2, title="Pairs"):
    if not paths:
        print("Không có cặp nào để hiển thị.")
        return

    num_pairs = len(paths)
    num_rows = math.ceil(num_pairs / pairs_per_row)

    width_ratios = []
    gap = 0.3

    for i in range(pairs_per_row):
        width_ratios.extend([1, 1])
        if i < pairs_per_row - 1:
            width_ratios.append(gap)

    total_cols = len(width_ratios)

    fig = plt.figure(figsize=(16, 4.5 * num_rows))
    gs = gridspec.GridSpec(num_rows, total_cols,
                           width_ratios=width_ratios,
                           wspace=0.01, hspace=0.35)

    for idx, (p1, p2, lbl, sim) in enumerate(paths):
        row = idx // pairs_per_row
        col = (idx % pairs_per_row) * 3
        color = "green" if lbl == 1 else "red"
        text = "Same" if lbl == 1 else "Different"

        img1 = Image.open(p1)
        img2 = Image.open(p2)

        ax1 = fig.add_subplot(gs[row, col])
        ax2 = fig.add_subplot(gs[row, col + 1])

        ax1.imshow(img1); ax1.axis("off")
        ax2.imshow(img2); ax2.axis("off")

        title_ax = fig.add_subplot(gs[row, col:col + 2])
        title_ax.axis("off")
        title_ax.set_title(
            f"{text}\nSim={sim:.3f} | Dist={1-sim:.3f}",
            color=color, fontsize=12, fontweight='bold', y=1.05
        )

    plt.suptitle(title)
    plt.show()
