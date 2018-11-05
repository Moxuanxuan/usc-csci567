import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
    #    '''
    #         Finds n_cluster in the data x
    #         params:
    #             x - N X D numpy array
    #         returns:
    #             A tuple
    #             (centroids a n_cluster X D numpy array, y a size (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates an Int)
    #         Note: Number of iterations is the number of time you update the assignment
    #     ''' 
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        initrow = np.random.choice(N,self.n_cluster)
        means = x[initrow,:]
        J=10**10
        number_of_updates = 0
        for i in range(self.max_iter):
            dists = -2*np.dot(x, means.T) + np.sum(np.square(means), axis = 1) + np.transpose([np.sum(np.square(x), axis = 1)])
            membership = np.argmin(dists,axis=1)
            number_of_updates = number_of_updates + 1
            Jnew = np.sum(np.square(x-means[membership,:]))/N
            if abs(J-Jnew) <= self.e:
                break
            J = Jnew
            for j in range(self.n_cluster):
                belong_vec = (membership == j)
                if np.sum(belong_vec) == 0:
                    continue
                means[j,:] = np.sum(x[belong_vec,:],axis=0)/np.sum(belong_vec)
        return (means, membership, number_of_updates)
        # DONOT CHANGE CODE BELOW THIS LINE

class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float) 
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by 
                    majority voting ((N,) numpy array) 
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        initrow = np.random.choice(N,self.n_cluster)
        means = x[initrow,:]
        J=10**10
        number_of_updates = 0
        for i in range(self.max_iter):
            dists = -2*np.dot(x, means.T) + np.sum(np.square(means), axis = 1) + np.transpose([np.sum(np.square(x), axis = 1)])
            membership = np.argmin(dists,axis=1)
            number_of_updates = number_of_updates + 1
            Jnew = np.sum(np.square(x-means[membership,:]))/N
            if abs(J-Jnew) <= self.e:
                break
            J = Jnew
            for j in range(self.n_cluster):
                belong_vec = (membership == j)
                if np.sum(belong_vec) == 0:
                    continue
                means[j,:] = np.sum(x[belong_vec,:],axis=0)/np.sum(belong_vec)
        centroids = means
        centroid_labels = np.zeros(self.n_cluster)
        for i in range(self.n_cluster):
            belong_vec = (membership == i)
            if np.sum(belong_vec) == 0:
                centroid_labels[i] = 0
                continue
            labels = y[belong_vec]
            tu=sorted([(np.sum(labels==j),j) for j in set(labels)],key = lambda x:(x[0], -x[1]))
            centroid_labels[i] = tu[-1][1]
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        labels = np.zeros(N)
        dists = -2*np.dot(x, self.centroids.T) + np.sum(np.square(self.centroids), axis = 1) + np.transpose([np.sum(np.square(x), axis = 1)])
        membership = np.argmin(dists,axis=1)
        labels = self.centroid_labels[membership]
        # DONOT CHANGE CODE BELOW THIS LINE
        return labels

