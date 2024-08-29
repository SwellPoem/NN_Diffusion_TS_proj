# Description: This script calculates the Frechet Inception Distance (FID) between two sets of data. 
# The FID is a measure of similarity between two sets of data, 
# Calculated by comparing the mean and covariance statistics of two sets of data.
import numpy as np
from Scripts.ts2vec.ts2vec import TS2Vec

# function to calculate the Frechet Inception Distance (FID) between two sets of data
def calculate_fid(data1, data2):
    # calculate mean and covariance statistics
    mu1, sigma1 = data1.mean(axis=0), np.cov(data1, rowvar=False)
    mu2, sigma2 = data2.mean(axis=0), np.cov(data2, rowvar=False)
    # calculate sum squared difference between means
    sum_sqaured_diff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    # using Eigenvalue Decomposition
    eigvals, eigvecs = np.linalg.eig(sigma1.dot(sigma2))
    covmean = eigvecs.dot(np.sqrt(np.diag(eigvals))).dot(eigvecs.T)
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = sum_sqaured_diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def Context_FID(ori_data, generated_data, device='cpu'):
    model = TS2Vec(input_dims=ori_data.shape[-1], device=device, batch_size=8, lr=0.001, output_dims=320,
                   max_train_length=3000)
    #train the model on the original data
    model.fit(ori_data, verbose=False)
    #transform the two data in their vector representation
    original_vector = model.encode(ori_data, encoding_window='full_series')
    generated_vector = model.encode(generated_data, encoding_window='full_series')
    #order of the original data is shuffled to avoid bias
    idx = np.random.permutation(ori_data.shape[0])
    original_vector = original_vector[idx]
    generated_vector = generated_vector[idx]
    #calculus of FID
    results = calculate_fid(original_vector, generated_vector)
    return results