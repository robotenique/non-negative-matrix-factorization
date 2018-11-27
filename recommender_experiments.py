import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

class MF():

    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))

        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)


def recsys():
    df = pd.read_csv("ml-latest-small/ratings.csv")
    df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    R_df = df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
    R = R_df.values
    A = training_dataset()
    B = svd_MF()
    #"""TODO:  Check how to do a train test split + how to evaluate each matrix factorization
    #"""
    ## MSE like this is a very poor way of checking...
    #mse_svds = np.mean((R - A)**2)
    #mse_mf = np.mean((R - B)**2)
    #print(f"MSE between R itself: {np.mean((R - R)**2)}")
    #print(f"MSE for SVDS: {mse_svds}")
    #print(f"MSE for MF: {mse_mf}")




def training_dataset():
    # Supress scientific notation
    np.set_printoptions(suppress=True, linewidth=300)
    # Number of latent features of the Matrix Factorization
    num_latent_features = 50
    # Percentage of the data to use for training. (1 - train_size) will be for testing
    train_size = .85
    df = pd.read_csv("ml-latest-small/ratings.csv")
    df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    whole_data = df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
    df = df.drop(['Timestamp'], axis=1)
    uniq_movies = np.unique(df.values[:, 1]).astype(int)
    # Mapping movie indices
    movie_to_index = {k:v for k, v in zip(uniq_movies, range(len(uniq_movies)))}
    # Prepare training and test data
    rnd_permutation = np.random.permutation(df.values)
    last_pos = int(train_size*len(df.values)) + 1
    X_train = rnd_permutation[:last_pos]
    X_train[:, 0] = X_train[:, 0] - 1
    X_train[:, 1] = np.array(list(map(lambda m_id: movie_to_index[int(m_id)], X_train[:, 1])))
    X_test = rnd_permutation[last_pos:]
    X_test[:, 0] = X_test[:, 0] - 1
    X_test[:, 1] = np.array(list(map(lambda m_id: movie_to_index[int(m_id)], X_test[:, 1])))
    new = np.zeros(whole_data.values.shape)
    # Add training data to our matrix to be trained. The other data are zero values
    new[X_train[:, 0].astype(int), X_train[:, 1].astype(int)] = X_train[:, 2]
    R = new
    user_ratings_mean = np.mean(R, axis = 1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)
    U, sigma, Vt = svds(R_demeaned, k = num_latent_features)
    sigma = np.diag(sigma)
    recc = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    #preds_df = pd.DataFrame(recc, columns = R_df.columns)
    #preds_df = pd.DataFrame(recc)
    #print(preds_df.head())
    # Testing for user 0
    num_recommendations = 10
    ratings = recc[0]
    # Negative because we want the max
    ind = np.argpartition(ratings, -num_recommendations)[-num_recommendations:]
    print(f"Recommendations for user [0]:")
    print(f"Indexes: {ind}")
    print(f"Ratings: {ratings[ind]}")
    # Calculate the MSE using the Frobenius norm
    # Correct indexes from the training
    u_id = X_test[:, 0].astype(int)
    m_id = X_test[:, 1].astype(int)
    # Frobenius norm (https://stats.stackexchange.com/questions/97411/evaluating-matrix-factorization-algorithms-for-netflix)
    mse = np.mean((X_test[:, 2] - recc[u_id, m_id])**2)
    print(f"Mean Squared Error: {mse}")
    return recc, mse

def svd_MF():
    # Supress scientific notation
    np.set_printoptions(suppress=True, linewidth=300)
    # Number of latent features of the Matrix Factorization
    num_latent_features = 50
    # Percentage of the data to use for training. (1 - train_size) will be for testing
    train_size = .85
    df = pd.read_csv("ml-latest-small/ratings.csv")
    df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    whole_data = df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
    df = df.drop(['Timestamp'], axis=1)
    uniq_movies = np.unique(df.values[:, 1]).astype(int)
    # Mapping movie indices
    movie_to_index = {k:v for k, v in zip(uniq_movies, range(len(uniq_movies)))}
    # Prepare training and test data
    rnd_permutation = np.random.permutation(df.values)
    last_pos = int(train_size*len(df.values)) + 1
    X_train = rnd_permutation[:last_pos]
    X_train[:, 0] = X_train[:, 0] - 1
    X_train[:, 1] = np.array(list(map(lambda m_id: movie_to_index[int(m_id)], X_train[:, 1])))
    X_test = rnd_permutation[last_pos:]
    X_test[:, 0] = X_test[:, 0] - 1
    X_test[:, 1] = np.array(list(map(lambda m_id: movie_to_index[int(m_id)], X_test[:, 1])))
    new = np.zeros(whole_data.values.shape)
    # Add training data to our matrix to be trained. The other data are zero values
    new[X_train[:, 0].astype(int), X_train[:, 1].astype(int)] = X_train[:, 2]
    R = new
    mf = MF(R, K=num_latent_features, alpha=0.001, beta=0.01, iterations=30)
    training_process = mf.train()
    """ print()
    print("P x Q:")
    print(mf.full_matrix())
    print()
    print("Global bias:")
    print(mf.b)
    print()
    print("User bias:")
    print(mf.b_u)
    print()
    print("Item bias:")
    print(mf.b_i)"""
    recc = mf.full_matrix()
    num_recommendations = 10
    ratings = recc[0]
    # Negative because we want the max
    ind = np.argpartition(ratings, -num_recommendations)[-num_recommendations:]
    print(f"Recommendations for user [0]:")
    print(f"Indexes: {ind}")
    print(f"Ratings: {ratings[ind]}")
    # Calculate the MSE using the Frobenius norm
    # Correct indexes from the training
    u_id = X_test[:, 0].astype(int)
    m_id = X_test[:, 1].astype(int)
    # Frobenius norm
    mse = np.mean((X_test[:, 2] - recc[u_id, m_id])**2)
    print(f"Mean Squared Error: {mse}")

    """ x = [x for x, y in training_process]
    y = [y for x, y in training_process]
    plt.figure(figsize=((16,4)))
    plt.plot(x, y)
    plt.xticks(x, x)
    plt.xlabel("Iterations")
    plt.ylabel("Mean Square Error")
    plt.grid(axis="y")
    plt.show() """
    return recc







if __name__ == '__main__':
    recsys()

