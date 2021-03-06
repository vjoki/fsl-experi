from typing import cast, Tuple, List, Dict, Optional
import warnings
import torch
import matplotlib.pyplot as plt
from pytorch_lightning.metrics.functional import roc
from pytorch_lightning.metrics.functional.classification import auroc, precision_recall


def minDCF(fpr: torch.Tensor, fnr: torch.Tensor, thresholds: torch.Tensor,
           p_target: float = 0.05, c_miss: int = 1, c_fa: int = 1):
    c_det = (c_miss * p_target * fnr) + (c_fa * (1 - p_target) * fpr)
    min_c_det_idx = torch.argmin(c_det)
    min_c_det = c_det[min_c_det_idx]
    min_c_det_threshold = thresholds[min_c_det_idx]

    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def

    return min_dcf, min_c_det_threshold


def equal_error_rate(fpr: torch.Tensor, fnr: torch.Tensor,
                     thresholds: torch.Tensor,
                     warn_threshold: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Index of the nearest intersection point.
    idx = torch.argmin(torch.abs(fnr - fpr))

    # roc() can return fpr, tpr, thresholds with much shorter length than scores.size(0)
    # (e.g. cases where scores.size(0) = 2611 and fnr.size(0) = 4), which leads to a large
    # differences between fpr[e] and fnr[e]. Thus we take the mean of fpr[e] and fnr[e] as a
    # good enough approximation. Could also interpolate using scipy brentq and interp1d methods,
    # but this on the other hand leads to frequent division by zero errors in some cases.
    eer = 0.5 * (fpr[idx] + fnr[idx])
    eer_threshold = thresholds[idx]

    # Since we're averaging the EER, warn if the difference between FNR and FPR is too large.
    if torch.abs(fpr[idx] - fnr[idx]) > warn_threshold:
        warnings.warn('Inaccurate EER ({}), real EER is somewhere between or near [{}, {}].'.format(eer, fpr[idx], fnr[idx]))

    # https://yangcha.github.io/EER-ROC/
    # https://stackoverflow.com/questions/28339746/equal-error-rate-in-python
    # Unfortunately this seems to fail frequently, incorrectly returning 0.0 and nan.
    #
    # from scipy.optimize import brentq
    # from scipy.interpolate import interp1d
    # eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    # eer_threshold = interp1d(fpr, thresholds)(eer)

    return eer, eer_threshold, idx


def compute_evaluation_metrics(outputs: List[List[torch.Tensor]],
                               plot: bool = False,
                               prefix: Optional[str] = None) -> Dict[str, torch.Tensor]:
    scores = torch.cat(list((scores for step in outputs for scores in step[0])))
    # NOTE: Need sigmoid here because we skip the sigmoid in forward() due to using BCE with logits for loss.
    #scores = torch.sigmoid(scores)
    print('Score range: [{}, {}]'
          .format(torch.min(scores).item(),
                  torch.max(scores).item()))
    labels = torch.cat(list((labels for step in outputs for labels in step[1])))

    auc = auroc(scores, labels, pos_label=1)
    fpr, tpr, thresholds = roc(scores, labels, pos_label=1)
    prec, recall = precision_recall(scores, labels)

    # mypy massaging, single tensors when num_classes is not specified (= binary case).
    fpr = cast(torch.Tensor, fpr)
    tpr = cast(torch.Tensor, tpr)
    thresholds = cast(torch.Tensor, thresholds)

    fnr = 1 - tpr
    eer, eer_threshold, idx = equal_error_rate(fpr, fnr, thresholds)
    min_dcf, min_dcf_threshold = minDCF(fpr, fnr, thresholds)

    # Accuracy based on EER and minDCF thresholds.
    eer_preds = (scores >= eer_threshold).long()
    min_dcf_preds = (scores >= min_dcf_threshold).long()
    eer_acc = torch.sum(eer_preds == labels).float() / labels.numel()
    min_dcf_acc = torch.sum(min_dcf_preds == labels).float() / labels.numel()

    if plot:
        assert idx.dim() == 0 or (idx.dim() == 1 and idx.size(0) == 1)
        i = int(idx.item())
        fpr = fpr.cpu().numpy()
        tpr = tpr.cpu().numpy()
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.plot([0, 1], [0, 1], 'r--', label='Reference', alpha=0.6)
        plt.plot([1, 0], [0, 1], 'k--', label='EER line', alpha=0.6)
        plt.plot(fpr, tpr, label='ROC curve')
        plt.fill_between(fpr, tpr, 0, label='AUC', color='0.8')
        plt.plot(fpr[i], tpr[i], 'ko', label='EER = {:.2f}%'.format(eer * 100))  # EER point
        plt.legend()
        plt.show()

    if prefix:
        prefix = '{}_'.format(prefix)
    else:
        prefix = ''

    return {
        '{}eer'.format(prefix): eer,
        '{}eer_acc'.format(prefix): eer_acc,
        '{}eer_threshold'.format(prefix): eer_threshold,
        '{}auc'.format(prefix): auc,
        '{}min_dcf'.format(prefix): min_dcf,
        '{}min_dcf_acc'.format(prefix): min_dcf_acc,
        '{}min_dcf_threshold'.format(prefix): min_dcf_threshold,
        '{}prec'.format(prefix): prec,
        '{}recall'.format(prefix): recall
    }


# torch.utils.data.DataLoader collate_fn
def collate_var_len_tuples_fn(batch):
    a, b, labels = zip(*batch)
    # lengths = torch.as_tensor(list(map(lambda t1, t2: (t1.size(0), t2.size(0)), a, b)))
    a = torch.nn.utils.rnn.pad_sequence(a, batch_first=True)
    b = torch.nn.utils.rnn.pad_sequence(b, batch_first=True)
    return a, b, torch.utils.data.dataloader.default_collate(labels)
