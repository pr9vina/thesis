import matplotlib.pyplot as plt
import numpy as np


def get_effect_plots(estimated_effect, true_effect):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.hist(estimated_effect, bins=30, alpha=0.7, color='blue')
    ax1.axvline(np.mean(estimated_effect), color='black', linestyle='--', linewidth=2)
    mean_estimated = np.mean(estimated_effect)
    ax1.set_title(f'Estimated effects: {np.round(mean_estimated, 3)}')

    ax2.hist(true_effect, bins=30, alpha=0.7, color='green')
    ax2.axvline(np.mean(true_effect), color='black', linestyle='--', linewidth=2)
    mean_true = np.mean(true_effect)
    ax2.set_title(f'True effects: {np.round(mean_true, 3)}')

    perc_diff = np.abs((mean_true - mean_estimated) / ((mean_true + mean_estimated) / 2))*100
    print(f"Pecrentage difference between true and estimated effect: {np.round(perc_diff, 3)}")
