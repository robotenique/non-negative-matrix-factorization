import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF

def recsys():
    training_dataset()

def training_dataset(training_data):
    # Supress scientific notation
    np.set_printoptions(suppress=True, linewidth=300) 
    num_latent_features = 2
    X_sparse = np.array([   [5, 3, 0, 1],
                            [4, 0, 0, 1],
                            [1, 1, 0, 5],
                            [1, 0, 0, 4],
                            [0, 1, 5, 4],
                         ])
    model = NMF(n_components=num_latent_features, init='random', random_state=0)
    W = model.fit(X_sparse)
    result = model.inverse_transform(model.transform(X_sparse))
    print("\n========= Original User x movie matrix =========")
    print(X_sparse)
    print("\n========= Reconstructed matrix after regularization =========")
    print(np.around(result, decimals=2))

if __name__ == '__main__':
    recsys()

