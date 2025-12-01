import torch
import numpy as np
import arc_scores
from collections import defaultdict
import itertools

def generate_balanced_pairs(labels, max_per_class=None, random_state=42):
    rng = np.random.RandomState(random_state)

    label2idx = defaultdict(list)
    for i, lb in enumerate(labels):
        label2idx[lb].append(i)

    pos_pairs = []
    for lb, idxs in label2idx.items():
        if len(idxs) < 2:
            continue

        idxs = np.array(idxs)
        if max_per_class and len(idxs) > max_per_class:
            idxs = rng.choice(idxs, max_per_class, replace=False)

        pos_pairs.extend(list(itertools.combinations(idxs, 2)))

    n_pos = len(pos_pairs)
    labels_unique = list(label2idx.keys())

    neg_pairs = []
    class_pairs = list(itertools.combinations(labels_unique, 2))

    for _ in range(n_pos):
        lb1, lb2 = class_pairs[rng.randint(len(class_pairs))]
        i = rng.choice(label2idx[lb1])
        j = rng.choice(label2idx[lb2])
        neg_pairs.append((i, j))

    pairs = [(i, j, 1) for (i, j) in pos_pairs] + \
            [(i, j, 0) for (i, j) in neg_pairs]

    rng.shuffle(pairs)
    return pairs


def evaluate(embs, labels, max_per_class=50, n_linspace=1000, epsilon=1e-6, random_state=42):
    embs = torch.cat(embs).cpu()
    labels = torch.cat(labels).cpu().numpy()

    pairs = generate_balanced_pairs(labels, max_per_class)
    pairs = np.array(pairs)

    idx_a = pairs[:, 0].astype(int)
    idx_b = pairs[:, 1].astype(int)
    similarity_scores = torch.sum(embs[idx_a] * embs[idx_b], dim=1).numpy()

    targets = pairs[:, 2].astype(int)

    # Best accuracy
    thresholds = np.linspace(
        similarity_scores.min() - epsilon,
        similarity_scores.max() + epsilon,
        n_linspace
    )
    preds = similarity_scores[None, :] >= thresholds[:, None]
    accs = (preds == targets).mean(axis=1)
    best_acc = accs.max()
    best_th = thresholds[accs.argmax()]

    # ROC & TAR
    roc_auc = arc_scores.compute_roc_auc(similarity_scores, targets)["auc"]
    tar_far_3 = arc_scores.tar_at_far(similarity_scores, targets)
    tar_far_4 = arc_scores.tar_at_far(similarity_scores, targets, far_target=1e-4)

    return {
        "accuracy": float(best_acc),
        "roc_auc": float(roc_auc),
        "tar_far_3": float(tar_far_3),
        "tar_far_4": float(tar_far_4),
        "threshold": float(best_th)
    }