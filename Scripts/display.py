
import scipy.stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

#to display the scores of comparison
def display_scores(results):
   mean = np.mean(results)
   sigma = scipy.stats.sem(results)
   sigma = sigma * scipy.stats.t.ppf((1 + 0.95) / 2., 5-1)
   print('Final Score: ', f'{mean} \xB1 {sigma}')

# to plot the data and compare the original and generated data
# input: original data, generated data, analysis type, compare size
# output: plot
def visualization(original_data, generated_data, analysis, save_path, compare=3000):
    # Analysis sample size (for faster computation)
    sample_size = min([compare, original_data.shape[0]])
    idx = np.random.permutation(original_data.shape[0])[:sample_size]

    original_data = original_data[idx]
    generated_data = generated_data[idx]

    _, seq_len, _ = original_data.shape

    for i in range(sample_size):
        if (i == 0):
            prep_data = np.reshape(np.mean(original_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data, np.reshape(np.mean(original_data[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat, np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

    # Visualization parameter
    colors = ["red" for i in range(sample_size)] + ["blue" for i in range(sample_size)]

    if analysis == 'tsne':

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        _, ax = plt.subplots(1)

        plt.scatter(tsne_results[:sample_size, 0], tsne_results[:sample_size, 1], c=colors[:sample_size], alpha=0.2, label="Original")
        plt.scatter(tsne_results[sample_size:, 0], tsne_results[sample_size:, 1], c=colors[sample_size:], alpha=0.2, label="Synthetic")

        ax.legend()

        plt.title('t-SNE plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
        plt.savefig(save_path+f"tsne.png", dpi=300)
        plt.show()

    elif analysis == 'kernel':

        _, ax = plt.subplots(1)
        sns.distplot(prep_data, hist=False, kde=True, kde_kws={'linewidth': 5}, label='Original', color="red")
        sns.distplot(prep_data_hat, hist=False, kde=True, kde_kws={'linewidth': 5, 'linestyle':'--'}, label='Synthetic', color="blue")

        plt.legend()
        plt.xlabel('Data Value')
        plt.ylabel('Data Density Estimate')
    
        plt.savefig(save_path+f"histo.png", dpi=300)
        plt.show()
        plt.close()

if __name__ == '__main__':
   pass
