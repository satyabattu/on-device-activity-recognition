import numpy as np

def mean_feature(signal):
    return np.mean(signal)

def std_feature(signal):
    return np.std(signal)

def energy_feature(signal):
    return np.sum(signal ** 2) / len(signal)

def extract_features(signal):
    """
    Extract features from a 1D signal window
    """
    return np.array([
        mean_feature(signal),
        std_feature(signal),
        energy_feature(signal)
    ])
