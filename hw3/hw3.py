from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Your implementation goes here!
    x = np.load(filename)
    x = x - np.mean(x, axis=0)
    return x

def get_covariance(dataset):
    n = len(dataset)
    x = np.dot(np.transpose(dataset), dataset)
    x = x / (n - 1)
    return x

def get_eig(S, m):
    w, v = eigh(S, subset_by_index=[len(S) - m, len(S) - 1])
    w = np.flip(w, 0)
    w_mtx = np.diag(w)
    v[:, [1, 0]] = v[:, [0, 1]]
    return w_mtx, v

def get_eig_prop(S, prop):
    w = eigh(S, eigvals_only=True)
    w_tot = sum(w)
    lower_bound = w_tot * prop
    w, v = eigh(S, subset_by_value=[lower_bound, np.inf])
    w = np.flip(w, 0)
    w_mtx = np.diag(w)
    v[:, [1, 0]] = v[:, [0, 1]]
    return w_mtx, v

def project_image(image, U):
    alpha = np.dot(np.transpose(U), image)
    return np.dot(U, alpha)

def display_image(orig, proj):
    orig = np.transpose(np.reshape(orig, (32, 32)))
    proj = np.transpose(np.reshape(proj, (32, 32)))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Original')
    ax2.set_title('Projection')
    bar_1 = ax1.imshow(orig, aspect = 'equal', cmap='viridis')
    bar_2 = ax2.imshow(proj, aspect = 'equal', cmap='viridis')
    fig.colorbar(bar_1, ax = ax1)
    fig.colorbar(bar_2, ax = ax2)
    plt.show()
