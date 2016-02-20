####
"""
Minimizer algorithms
"""

import numpy as np
import scipy.linalg as LA
import itertools, time, tools, math

class Ranker:
    """
    Base class for all ranking algorithms
    """
    def __init__(self,m,distance_method='Spearman'):
        self.m = m #is the data object which has all relevant information
        self.n = len(self.m.a_l)
        if distance_method == 'Kendall':
            self.distance_method = tools.Kendall_tau
            self.distance_method.name = 'Kendall'
        else: 
            self.distance_method = tools.Spearman
            self.distance_method.name = 'Spearman'
        self.psize = math.factorial(len(self.m.a_l[0]))

    def summary(self):
        print "%s super-list search summary" % self.method_name
        print "List length = %d" % len(self.super_list)
        print "Permutations = %d" % self.psize
        print "Time elapsed = %.2f seconds" % self.time
        print "Scoring method = %s" % self.distance_method.name
        print "Minimum score = %.2f" % self.score

class RandomWeights(Ranker):
    """
    A ranking algorithm that samples the multi-objective function
    F(x) = Sum(w_i*f_i) with random w_i for N times. This is a simple
    multi-objective algorithm to the select best candidates without
    having to specify weights initially. Features have to be scaled.
    """
    def start(self,scaling_method='MinMax_scale',N=10,t=100):
        self.scaling_method = scaling_method
        if self.scaling_method == 'MinMax_scale':
            self.scaled_data = self.m.minmax_scaled
        elif self.scaling_method == 'Z_scale':
            self.scaled_data = self.m.z_scaled
        else:
            raise Exception("Scaling method not specified correctly")
        self.method_name = 'Random-weights optimizer'
        self.N = N*len(self.scaled_data[0])
        count = self.N
        s_list = np.zeros(len(self.scaled_data[0]))

        print "Starting Multi-Objective Optimization with %d random weights" % self.N
        start_time = float(time.time())
        while count:
            w = np.random.rand(self.n)
            w = w/np.sum(w)
            b = np.zeros(len(self.scaled_data[0]))
            for weight,i in zip(w,self.scaled_data):
                b=b+weight*i
            s_list = s_list + tools.get_ranks([b])[0]
            count-=1
            if not count%t: print str(count) + ' ',
        self.super_list = tools.get_ranks([s_list])[0]
        d = 0
        for l in tools.get_ranks(self.scaled_data):
            d=d+self.distance_method(self.super_list,l)
        self.score = d*2/len(self.scaled_data[0])**2/self.n
        self.time = float(time.time())-start_time
        print "Done. Time elapsed %.2f seconds" % self.time
        #s_list has accumulated ranks for N iterations of rand.weights

class RankBrute(Ranker):
    """
    Brute-force rank aggregation. Will be computationally intractable even
    with the lists with >= 10 elements.
    """
    def start(self, freq=5000):
        print "Starting Brute-Force rank aggregation"
        self.method_name = 'Brute-force'
        m = self.m.a_ranks
        start_time = float(time.time())
        dist = 1e129
        s_list = []
        count = 0
        total = self.psize
        for b in itertools.permutations(range(1,len(m[0])+1)):
            c = list(b)
            d = 0
            for l in m:
                d=d+self.distance_method(c,l)
            if d < dist:
                s_list = list(c)
                dist = d
            count+=1
            if count%freq==0:
                print "Finished %.2f percent. Current score: %.2f" \
                       % (100.0*(count+1)/total,dist)
        print "Done. Time elapsed: %.2f seconds." % (float(time.time())-start_time)
        self.super_list = s_list
        self.score = dist*2/len(self.m.a_ranks[0])**2/self.n
        self.time = float(time.time())-start_time


class RankBorda(Ranker):
    """
    A fast aggergation algorithm which uses the rank of a candidate in each list as a
    measure of its score in that list, and accumulates the total score for each candidate
    to come up with a master list where candidates are ranked by this total score.
    """
    def start(self, freq=5000):
        print "Starting Rank Aggregation with Borda's method"
        self.method_name = "Borda's method"
        m = self.m.a_ranks
        start_time = float(time.time())
        b = np.zeros(len(m[0]))
        for i in m:
            b=b+i
        s_list = tools.get_ranks([b])[0]
        dist = 0.0
        for i in m:
            dist += self.distance_method(i,s_list)
        print "Done. Time elapsed: %.2f seconds." % (float(time.time())-start_time)
        self.super_list = s_list
        self.score = dist*2/len(self.m.a_ranks[0])**2/self.n
        self.time = float(time.time())-start_time

class RankRandom(Ranker):
    """
    A benchmark algorithm to evaluate the performance of rank aggregation algorithms in a given problem.
    It will generate a set of N completely randomly ranked super-list and return an expectation
    value for the upper bound for the distance. If any method yields a total distance worse (larger) than 
    this, there must be something wrong!
    """
    def start(self, N=100):
        print "Starting random ranking with %d random weights" % N
        #average 100 random rankings
        self.method_name = "Random ranking"
        m = self.m.a_ranks
        start_time = float(time.time())
        count = N
        d_t = 0.0
        while count:
            count-=1
            v = np.random.rand(len(m[0]))
            s_list = tools.get_ranks([v])[0]
            dist = 0.0
            for i in m:
                dist += self.distance_method(i,s_list)
            d_t += dist
        d_t = d_t/N
        print "Done. Time elapsed: %.2f seconds." % (float(time.time())-start_time)
        self.super_list = s_list
        self.score = dist*2/len(self.m.a_ranks[0])**2/self.n
        self.time = float(time.time())-start_time

class RankMC4(Ranker):
    """
    A python implementation of the Markov Chain 4 method proposed by Dwork et al. 2001.
    This implementation is still in the experimental stage.
    """
    def start(self, N=1000, t=1000):
        print "Starting Markov Chain Rank Aggregation"
        self.method_name = "Markov Chain Ranking"
        m = self.m.a_ranks
        start_time = float(time.time())
        n = len(m[0])
        self.N = N*n
        self.t = t*n
        # initialize transition matrix
        M = np.zeros((n,n))
        n_list_over2 = len(m)/2
        P = np.random.randint(n)
        #check for absorbing state every t steps
        absorbing_states=[]
        count = self.N
        while count:
            count -= 1
            # Initial Check if P is in the list of absorbing states
            if P in absorbing_states:
                while True:
                    P = np.random.randint(n)
                    if P not in absorbing_states:
                        break

            # Mainf MC4 algorithm starts
            # Pick a new state Q randomly
            Q = np.random.randint(n)
            if Q in absorbing_states: continue
            ct = 0
            for l in m:
                # check if Q has a higher rank in majority of lists
                if l[Q]<l[P]:
                    ct += 1
            if ct >= n_list_over2:
                # Q is the new state
                M[Q,P]+=1
                P = Q
            else:
                M[P,P]+=1

            # Check absotbing states and print status every t steps
            if not (count%self.t):
                #First checking for absorbing states
                for i in range(len(M)):
                    s = np.sort(M[i])
                    if s[-1]>1000 and s[-1] == M[i,i] and s[-2]<=1:
                        print "Encountered absorbing state %d, removing it" % i
                        absorbing_states.append(i)
                        M = np.zeros((n,n))
                        count = self.N 
                print "%d current absorbing states %s" % ( count, str( absorbing_states ) )
                # Second compute current score
                eigen = tools.getlefteigen(M)
                s_list = tools.get_ranks([1-eigen])[0]
                dist = 0.0
                for l in m:
                    dist += self.distance_method(l,s_list)
                print "Current score: %.2f" % dist

        eigen = tools.getlefteigen(M)
        s_list = tools.get_ranks([1-eigen])[0]
        dist = 0.0
        for l in m:
            dist += self.distance_method(l,s_list)
        print "Done. Time elapsed: %.2f seconds." % (float(time.time())-start_time)
        self.super_list = s_list
        self.score = dist*2/len(self.m.a_ranks[0])**2/self.n
        self.time = float(time.time())-start_time
        self.absorbing_states = absorbing_states
