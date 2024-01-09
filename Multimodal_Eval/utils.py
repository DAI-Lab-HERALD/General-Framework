import numpy as np
import scipy.stats as stats
import ot

from scipy.special import logsumexp


def create_random_data_splt(data, rnd_seed=0, n_samples=20000):
    np.random.seed(rnd_seed)

    # Randomly select n_samples from data
    rnd_idx = np.random.choice(len(data), n_samples, replace=False)

    # Split the selected data into training and testing sets 50/50
    data = data[rnd_idx]
    data_train = data[:int(n_samples/2)]
    data_test = data[int(n_samples/2):]

    return data_train, data_test


def calculate_JSD(log_like_Ptrue, log_like_Ptest, log_like_Qtrue, log_like_Qtest):
    # combine log-likelihoods of true and test data for P and Q
    log_like_P = np.concatenate((log_like_Ptrue, log_like_Ptest))
    log_like_Q = np.concatenate((log_like_Qtrue, log_like_Qtest))

    # calculate the log likelihood of the combined data; can be seen as a form of importance sampling
    log_like_PQ = logsumexp(np.stack((log_like_P, log_like_Q), axis=0), axis=0) - np.log(2) 

    helpP = log_like_P - log_like_PQ
    helpQ = log_like_Q - log_like_PQ

    # calculate the KLD of P and Q
    KLD_P = np.mean(np.exp(helpP) * helpP)
    KLD_Q = np.mean(np.exp(helpQ) * helpQ)

    # calculate the JSD of P and Q
    JSD = (KLD_P + KLD_Q) / 2
    JSD /= np.log(2)

    return JSD

def calculate_Wasserstein(log_like_Ptrue, log_like_Ptest):
    # calculate the Wasserstein distance between the log-likelihoods of the true and test data
    Wasserstein = stats.wasserstein_distance(log_like_Ptrue, log_like_Ptest)

    return Wasserstein

def calculate_multivariate_Wasserstein(X_true, X_pred):
    # calculate the Wasserstein distance between the samples of the true and test data
    Wasserstein = ot.dist(X_true, X_pred, metric='euclidean')
    sample_weights1 = np.ones(X_true.shape[0]) / X_true.shape[0]  # Uniform weights for true distribution
    sample_weights2 = np.ones(X_pred.shape[0]) / X_pred.shape[0]  # Uniform weights for pred distribution
    
    # Calculate the optimal transport plan
    transport_plan = ot.emd(sample_weights1, sample_weights2, Wasserstein)
    
    # Calculate the Wasserstein distance as the optimal transport cost
    multivariate_wasserstein_distance = np.sum(transport_plan * Wasserstein)
    return multivariate_wasserstein_distance
