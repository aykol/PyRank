import numpy as np
import math
import scipy.linalg as LA
import scipy.stats as ss
import itertools
import time
import tools, minimizers, montecarlo

class FeatureSet:
    """

    Basic class to input feature lists and perform basic operations on the lists.
    Can load data from a CSV file with a header row such as "candidate, list1, list2, ..."
    left-most column stores the names for candidates. Can perform Z-scaling and Min-Max
    scaling on data for multi-objective optimization.

    """
    def __init__(self):
        print "Initialized a set of feature lists"

    def load_data(self,filename):
        f = open(filename,'r')
        self.headers = f.readline().rstrip('\n').split(',')
        self.c_l = [] # candidate list m candidates
        self.a_l = [] # attribute list, m x n
        for i in range(len(self.headers)-1):
            self.a_l.append([])   
        while True:
            l = f.readline().rstrip('\n').split(',')
            if len(l) <= 1:
                break
            self.c_l.append(l[0])
            for i in range(0,len(l)-1):
                self.a_l[i].append(float(l[i+1]))
        print "Found %d attributes, and %d candidates to rank." % ((len(self.headers)-1),len(self.a_l[0]))
        self.call_ranks()
    
    def call_ranks(self):
        self.a_ranks = tools.get_ranks(self.a_l)

    def Z_scale(self):
        self.z_scaled = []
        for i in self.a_l:
            mean,std = np.mean(i),np.std(i)
            self.z_scaled.append([(x-mean)/std for x in i])
        self.z_scaled = np.array(self.z_scaled)
    
    def MinMax_scale(self):
        self.minmax_scaled = []
        for i in self.a_l:
            min, max = np.amin(i), np.amax(i)
            self.minmax_scaled.append([(x-min)/(max-min) for x in i])
        self.minmax_scaled = np.array(self.minmax_scaled)



if __name__ == "__main__":
    new_test = FeatureSet()
    new_test.load_data('data_files/toyset2')
    print new_test.headers
    print new_test.c_l
    new_test.Z_scale()
    new_test.MinMax_scale()
    mo = minimizers.RandomWeights(new_test)
    mo.start()    
    mc = minimizers.RankMC4(new_test)
    mc.start()
    brute = minimizers.RankBrute(new_test)
    brute.start()
    borda = minimizers.RankBorda(new_test)
    borda.start()
    random = minimizers.RankRandom(new_test)
    random.start()
    mcmc = montecarlo.CrossEntropy(new_test,40)
    x = mcmc.start()
#