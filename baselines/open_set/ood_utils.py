import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def euclidean_distance(a, b):
    ''' Memory efficient implementation: ||a||² + ||b||² - 2<a,b> '''

    a_norm = torch.sum(a**2, dim=1, keepdim=True)
    b_norm = torch.sum(b**2, dim=1)
    distances = a_norm + b_norm - 2.0 * torch.matmul(a, b.T)
    distances = torch.clamp(distances, min=0.0)
    return torch.sqrt(distances)


def score_nearest_mean(train_features, train_labels, val_features):
    device = train_features.device

    id_labels = np.unique(train_labels)
    
    centers = []
    for c in id_labels:
        centers.append(train_features[train_labels == c].mean(dim=0))
    centers = torch.stack(centers).to(device)
    distances = euclidean_distance(val_features, centers)
    score_matrix = -distances

    score, pred = torch.max(score_matrix, dim=1)
    return score


def score_linear_probe(train_features, train_labels, val_features, lr_kwargs):
    classifier = LogisticRegression(**lr_kwargs)
    classifier.fit(train_features, train_labels)

    score_matrix = torch.from_numpy(classifier.decision_function(val_features))
    score, pred = torch.max(score_matrix, dim=1)
    return score


def visualise_ood(pred_id, pred_ood, normalize=True):
    plt.figure(figsize=(10,6))
    plt.hist(pred_id, bins=30, alpha=0.5, label='ID', density=normalize)
    plt.hist(pred_ood, bins=30, alpha=0.5, label='OOD', density=normalize)
    plt.xlabel('ID Score')
    plt.ylabel('Frequency')
    
    # Add legend
    plt.legend()
    
    # Show the plot
    plt.show()


def get_curve_online(known, novel, stypes = ['Bas']):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    for stype in stypes:
        known.sort()
        novel.sort()
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        fp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k+num_n):
            if k == num_k:
                tp[stype][l+1:] = tp[stype][l]
                fp[stype][l+1:] = np.arange(fp[stype][l]-1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l+1:] = np.arange(tp[stype][l]-1, -1, -1)
                fp[stype][l+1:] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l+1] = tp[stype][l]
                    fp[stype][l+1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l+1] = tp[stype][l] - 1
                    fp[stype][l+1] = fp[stype][l]
        tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
        tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
    return tp, fp, tnr_at_tpr95


def metric_ood(known, novel, stypes = ['Bas'], verbose=True):
    tp, fp, tnr_at_tpr95 = get_curve_online(known, novel, stypes)
    results = dict()
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    if verbose:
        print('      ', end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')
        
    for stype in stypes:
        if verbose:
            print('{stype:5s} '.format(stype=stype), end='')
        results[stype] = dict()
        
        # TNR
        mtype = 'TNR'
        results[stype][mtype] = 100.*tnr_at_tpr95[stype]
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        
        # AUROC
        mtype = 'AUROC'
        tpr = np.concatenate([[1.], tp[stype]/tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype]/fp[stype][0], [0.]])
        results[stype][mtype] = 100.*(-np.trapz(1.-fpr, tpr))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        
        # DTACC
        mtype = 'DTACC'
        results[stype][mtype] = 100.*(.5 * (tp[stype]/tp[stype][0] + 1.-fp[stype]/fp[stype][0]).max())
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        
        # AUIN
        mtype = 'AUIN'
        denom = tp[stype]+fp[stype]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[stype]/denom, [0.]])
        results[stype][mtype] = 100.*(-np.trapz(pin[pin_ind], tpr[pin_ind]))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        
        # AUOUT
        mtype = 'AUOUT'
        denom = tp[stype][0]-tp[stype]+fp[stype][0]-fp[stype]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[stype][0]-fp[stype])/denom, [.5]])
        results[stype][mtype] = 100.*(np.trapz(pout[pout_ind], 1.-fpr[pout_ind]))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
            print('')

    return results