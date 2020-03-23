from matplotlib import pyplot as plt
import numpy as np
import os
import pickle


def stack_tests(lrs, all_losses, all_min_lr, all_max_lr):
    n = len(lrs)
    all_losses_res = []
    for l in all_losses:
        pad_width = (0, n - len(l))
        padded_lo = np.pad(l, pad_width, mode='constant', constant_values=(0, l[-1]))
        all_losses_res.append(padded_lo)

    lrs_res = lrs
    all_losses_res = np.vstack(all_losses_res)
    all_min_lr_res = np.vstack(all_min_lr)
    all_max_lr_res = np.vstack(all_max_lr)
    return lrs_res, all_losses_res, all_min_lr_res, all_max_lr_res


def plot_lr_loss(lrs_res, all_losses_res, all_min_lr_res, all_max_lr_res,
                 epoch_ratio, optimizer,
                 bs=512, prefix='cifar10',
                 data_folder=None,
                 ylim_top_ratio=0.95):

    losses_m = np.median(all_losses_res, axis=0)
    losses_lo = np.quantile(all_losses_res, q=0.1, axis=0)
    losses_hi = np.quantile(all_losses_res, q=0.9, axis=0)

    min_lr_m = np.median(all_min_lr_res, axis=0)[0]
    min_lr_lo = np.quantile(all_min_lr_res, q=0.1, axis=0)[0]
    min_lr_hi = np.quantile(all_min_lr_res, q=0.9, axis=0)[0]

    max_lr_m = np.median(all_max_lr_res, axis=0)[0]
    max_lr_lo = np.quantile(all_max_lr_res, q=0.1, axis=0)[0]
    max_lr_hi = np.quantile(all_max_lr_res, q=0.9, axis=0)[0]

    plt.figure(figsize=(20, 10))
    plt.plot(lrs_res, losses_m)
    plt.grid()
    plt.fill_between(lrs_res, losses_lo, losses_hi, alpha=0.1, color="b")

    min_idx = (np.abs(lrs_res - min_lr_m)).argmin()
    max_idx = (np.abs(lrs_res - max_lr_m)).argmin()
    min_lr_pt = (min_lr_m, losses_m[min_idx])
    max_lr_pt = (max_lr_m, losses_m[max_idx])

    plt.plot(min_lr_m, losses_m[min_idx], markersize=10, marker='*', color='red')
    plt.plot(max_lr_m, losses_m[max_idx], markersize=10, marker='*', color='red')

    plt.annotate(xy=min_lr_pt, s=f'{min_lr_m:.2E}')
    plt.annotate(xy=max_lr_pt, s=f'{max_lr_m:.2E}')

    plt.plot([min_lr_lo, min_lr_hi], [losses_m[min_idx], losses_m[min_idx]], color='red')
    plt.plot([max_lr_lo, max_lr_hi], [losses_m[max_idx], losses_m[max_idx]], color='red')

    plt.xlabel("Learning Rate", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.xscale('log')
    gap = losses_m[0] - np.min(losses_m)
    plt.ylim(bottom=np.min(losses_m) - 0.5 * gap, top=losses_m[0] + ylim_top_ratio * gap)

    plt.xlim(left=1e-7, right=100)

    plt.title(f"{prefix.title()} LR Range Test {epoch_ratio:.3} epochs with {optimizer} and batch size {bs}", fontsize=20)

    if data_folder:
        filename = f"{prefix}_lrrt_{epoch_ratio:.3}_{optimizer}_{bs}.png"
        plt.savefig(os.path.join(data_folder, filename))
        
        
def plot_lrrt_curves(c, bins, lrs_res, all_losses_res):
    plt.figure(figsize=(20,10))
    n = all_losses_res.shape[0]
    for i in range(n):
        plt.plot(lrs_res, all_losses_res[i,:])
    
    losses_m = np.median(all_losses_res, axis=0)
    gap = losses_m[0] - np.min(losses_m)
    plt.ylim(bottom=np.min(losses_m) - 0.25 * gap, top=losses_m[0] + 0.25 * gap)
    plt.xlim(left=1e-7, right=100)
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.xscale('log')
    plt.grid("on")
    plt.title(f"Minimum of loss in [{bins[c-1]}, {bins[c]}]")
    
    
def plot_lrrt_curves_by_max_lr(lrs_res, all_losses_res, all_max_lr_res):
    n_curves = all_losses_res.shape[0]
    lrs_of_min_loss = all_max_lr_res*10
    bins = 10**np.linspace(-7, 2, 10)
    clusters_idx = np.digitize(lrs_of_min_loss, bins, right=True).reshape((n_curves,))
    clusters = np.unique(clusters_idx) 
    for c in clusters:
        idx_in_c = np.where(clusters_idx==c)
        all_losses_res_in_c = all_losses_res.take(idx_in_c, axis=0)[0]
        plot_lrrt_curves(c, bins, lrs_res, all_losses_res_in_c)
        plt.pause(0.05)
