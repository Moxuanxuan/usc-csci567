import numpy as np
from kmeans import KMeans

class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures (Int)
            e : error tolerance (Float) 
            max_iter : maximum number of updates (Int)
            init : initialization of means and variance
                Can be 'random' or 'kmeans' 
            means : means of Gaussian mixtures (n_cluster X D numpy array)
            variances : variance of Gaussian mixtures (n_cluster X D X D numpy array) 
            pi_k : mixture probabilities of different component ((n_cluster,) size numpy array)
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k (see P4.pdf)

            # DONOT MODIFY CODE ABOVE THIS LINE
            kinit = KMeans(n_cluster = self.n_cluster,e = self.e)
            self.means, membership, number_of_updates = kinit.fit(x)
            Nk = [np.sum(membership==j) for j in range(self.n_cluster)]
            Nk = np.array(Nk)
            self.pi_k = Nk/N
            variances = np.zeros((self.n_cluster,D,D))
            for i in range(self.n_cluster):
                pre_var = x[(membership==i),:]-self.means[i,:]
                variances[i,:,:] = np.dot(pre_var.T,pre_var)
            self.variances = variances / Nk.reshape(self.n_cluster,1,1)
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - initialize variance to be identity and pi_k to be uniform

            # DONOT MODIFY CODE ABOVE THIS LINE
            self.pi_k = np.ones(self.n_cluster)/self.n_cluster
            self.variances = np.zeros((self.n_cluster,D,D))
            for i in range(self.n_cluster):
                self.variances[i,:,:] = np.eye(D)
            self.means = np.random.rand(self.n_cluster,D)
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - Use EM to learn the means, variances, and pi_k and assign them to self
        # - Update until convergence or until you have made self.max_iter updates.
        # - Return the number of E/M-Steps executed (Int) 
        # Hint: Try to separate E & M step for clarity
        # DONOT MODIFY CODE ABOVE THIS LINE
        l = self.compute_log_likelihood(x)
        # print(l)
        # print(self.log_mat)
        iterations = 0
        for i in range(self.max_iter):
            # print(self.log_mat)
            r_mat = self.log_mat/np.sum(self.log_mat,axis=0).reshape(1,N)
            # print(r_mat)
            Nk = np.sum(r_mat, axis=1)
            # print(Nk)
            self.pi_k = Nk/N
            self.means = np.dot(r_mat,x)/Nk.reshape(self.n_cluster,1)
            variances = np.zeros((self.n_cluster,D,D))
            for i in range(self.n_cluster):
                pre_var = x-self.means[i,:]
                variances[i,:,:] = np.dot(pre_var.T*r_mat[i,:],pre_var)
            self.variances = variances / Nk.reshape(self.n_cluster,1,1)
            lnew = self.compute_log_likelihood(x)
            iterations = iterations + 1
            if abs(l-lnew)<=self.e :
                break
            l = lnew
            # print(l)
        return iterations
        # DONOT MODIFY CODE BELOW THIS LINE

		
    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        D = self.means.shape[1]
        samples = np.zeros((N,D))
        rand_w = np.random.rand(N)
        for i in range(N):
            p_plus = 0
            for j in range(self.n_cluster):
                p_plus = p_plus + self.pi_k[j]
                if rand_w[i]<=p_plus:
                    break
            samples[i,:] = np.random.multivariate_normal(self.means[j,:],self.variances[j,:,:])
        # DONOT MODIFY CODE BELOW THIS LINE
        return samples        

    def compute_log_likelihood(self, x, means=None, variances=None, pi_k=None):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        if means is None:
            means = self.means
        if variances is None:
            variances = self.variances
        if pi_k is None:
            pi_k = self.pi_k    
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood (Float)
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        Gaussians = [self.Gaussian_pdf(means[i,:],variances[i,:,:]) for i in range(self.n_cluster)]
        N,D = x.shape
        log_mat = np.zeros((self.n_cluster, N))
        for i in range(self.n_cluster):
            for j in range(N):
                log_mat[i,j] = Gaussians[i].getLikelihood(x[j,:])
        log_mat = pi_k.reshape(self.n_cluster,1)*log_mat
        self.log_mat = log_mat
        log_likelihood = np.sum(np.log(np.sum(self.log_mat,axis=0)))
        log_likelihood = float(log_likelihood)
        # DONOT MODIFY CODE BELOW THIS LINE
        return log_likelihood

    class Gaussian_pdf():
        def __init__(self,mean,variance):
            self.mean = mean
            self.variance = variance
            self.c = None
            self.inv = None
            '''
                Input: 
                    Means: A 1 X D numpy array of the Gaussian mean
                    Variance: A D X D numpy array of the Gaussian covariance matrix
                Output: 
                    None: 
            '''
            # TODO
            # - comment/remove the exception
            # - Set self.inv equal to the inverse the variance matrix (after ensuring it is full rank - see P4.pdf)
            # - Set self.c equal to ((2pi)^D) * det(variance) (after ensuring the variance matrix is full rank)
            # Note you can call this class in compute_log_likelihood and fit
            # DONOT MODIFY CODE ABOVE THIS LINE
            D = self.variance.shape[0]
            def invertible():
                if np.linalg.det(self.variance)==0:
                    self.variance = self.variance + 10**(-3)*np.eye(D)
                    invertible()
                else:
                    return
            invertible()
            self.inv = np.linalg.inv(self.variance)
            self.c = (2*np.pi)**D*np.linalg.det(self.variance)
            # DONOT MODIFY CODE BELOW THIS LINE

        def getLikelihood(self,x):
            '''
                Input: 
                    x: a 1 X D numpy array representing a sample
                Output: 
                    p: a numpy float, the likelihood sample x was generated by this Gaussian
                Hint: 
                    p = e^(-0.5(x-mean)*(inv(variance))*(x-mean)'/sqrt(c))
                    where ' is transpose and * is matrix multiplication
            '''
            #TODO
            # - Comment/remove the exception
            # - Calculate the likelihood of sample x generated by this Gaussian
            # Note: use the described implementation of a Gaussian to ensure compatibility with the solutions
            # DONOT MODIFY CODE ABOVE THIS LINE
            p = np.exp(-0.5*np.dot(np.dot((x-self.mean),self.inv),(x-self.mean).T))/np.sqrt(self.c)
            # DONOT MODIFY CODE BELOW THIS LINE
            return p
