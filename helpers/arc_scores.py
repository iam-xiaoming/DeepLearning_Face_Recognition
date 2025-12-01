import numpy as np
from sklearn.metrics import roc_curve, auc

def compute_roc_auc(scores, targets, pos_label=1):
    """
    distances: 1D array (N,)
    targets: 1D array {0,1}
    returns: dict with fpr, tpr, thresholds, auc
    """
    fpr, tpr, th = roc_curve(targets, scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    return {"fpr": fpr, "tpr": tpr, "thresholds": th, "auc": roc_auc}

def tar_at_far(scores, targets, far_target=1e-3):
    """
    Return TAR (TPR) at given FAR level.
    FAR = FP / N_neg
    We compute thresholds from ROC points and interpolate TPR at threshold s.t. FPR <= far_target
    """
    res = compute_roc_auc(scores, targets)
    fpr = res["fpr"]
    tpr = res["tpr"]

    # find max TPR where FPR <= far_target
    mask = fpr <= far_target
    if mask.sum() == 0:
        # If no point has FPR <= far_target, interpolate using first segment
        # linear interpolation across fpr/tpr
        tar = np.interp(far_target, fpr, tpr)
    else:
        tar = tpr[mask].max()
    return tar