####
"""
Monte Carlo algorithms
"""

import numpy as np
import itertools, time, tools, math
import minimizers

class CrossEntropy(minimizers.Ranker):
    """
    The Cross-Entropy Monte Carlo approach as introduced by Pihur et al. BMC Bioinformatics 2009.
    This implementation is still in the experimental stage. 
    """
    def start(self,k=9,N=5000,rho=0.01,w=0.1,epsilon=1.0e-5):
        #top-k list: 0 will compute k = n (# of candidates)
        self.method_name = "Cross Entropy Monte Carlo"
        m = self.m.a_ranks
        self.N = N
        n = len(m[0])
        if k==0:
            k = n
        self.k = k
        print "Starting Cross Entropy Monte Carlo simulation for top-%d list" % k
        start_time = float(time.time())
        x = self.get_X(n,k)
        # initialize p
        p = np.empty([n,k])
        p.fill(1.0/float(n))

        x_old = x
        print "Entering the main loop."
        while True:
            t_init = float(time.time())
            #This block loops to draw N samples from P with MCMC
            count = N
            #generate random numbers to use later 
            u = np.random.rand(N) 
            sample_matrices = [] #stores the N ranking matrices
            while count:
               count -=1
               x_new = self.get_X(n,k)
               prob_A = 1
               for j in range(n):
                   for r in range(k):
                       prob_A *= p[j,r]**(x_new[j,r]-x_old[j,r])
#               print prob_A, u[count]
               if prob_A > u[count]:
                   x_old = x_new
               sample_matrices.append(x_old)
               # sample_list has N samples.
            # Now we should find rankings of these N samples and get scores
            sample_rankings = []
            for matrix in sample_matrices:
                ranks = self.get_ranking(matrix)       
                score = self.get_score(ranks)
                sample_rankings.append((matrix,ranks,score))
            # we now have tuples (matrix,ranks,score) in sample_rankings
            # samples sorted by their scores (last element of the tuple)
            sorted_samples = sorted(sample_rankings,key=lambda v: v[2])
            # pick the top rho quantile
            a = int(len(sorted_samples)*rho)
            print a
            # Update p and accumulate convergence criterion
            conv = 0.0
            for j in range(n):
                for r in range(k):
                    q = 0
                    for i in range(a):
                        q += sorted_samples[i][0][j,r]
                    corr = w*float(q)/a
                    conv += np.fabs(corr - w*p[j,r]) # This is simply the difference p(t+1)-p(t)
                    p[j,r] = (1-w)*p[j,r] + corr
            conv = conv/n/k
            t_end = float(time.time())
            print "Current score: %.2f. Convergence criterion: %f. Loop time: %f sec." % (sorted_samples[0][2],conv,t_end-t_init)
            if conv < epsilon:
                super_tuple = sorted_samples[0]
                break
            else: 
                continue
        return super_tuple

    @staticmethod
    def get_ranking(matrix):
        # given a n x k matrix for a top-k list, returns a list with n elements with ranks in top-k, and -1 otherwise
        n = matrix.shape[0]
        k = matrix.shape[1]
        l = np.empty(n)
        l.fill(-1)
        for i in range(k):
            candidate = np.where(matrix[:,i]==1.0)[0][0]
            l[candidate] = i+1
        return l

    def get_score(self,l):       
        dist = 0.0
        for i in self.m.a_ranks:
            # we first replace any -1 in l with the same ranking in m_aranks
            for j in np.where(l==-1)[0]:
                l[j] == i[j]
            dist += self.distance_method(i,l)
        return dist*2/self.k**2/len(self.m.a_ranks)

    @staticmethod
    def get_X(n,k):
        # generates a random ranking matrix n x k for top-k
        X = np.zeros((n,k))     
#        print 'x',X,'\n\n'
        r = np.arange(n)
        np.random.shuffle(r)
#        print 'r',r,'\n\n'
        for i in range(k):
            X[r[i],i] = 1.0
        return X
