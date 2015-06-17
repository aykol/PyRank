import numpy as np
import math
import scipy.stats as ss
import itertools
import time

def get_ranks(m):
    ranks = []    
    for i in m:
        ranks.append(ss.rankdata(i))
    return ranks        

def Spearman(list1,list2):
    name = 'Spearman'
    return np.linalg.norm(list1-list2,ord=1)

def Kendall_tau(list1,list2,p=0):
    name = 'Kendall_tau'
    K = 0.0
    for i in itertools.combinations(range(len(list1)),2):
        c = list(i)
        t,u = c[0],c[1]
# Syntactically this is what is done but the finel else statemen# is sufficient to ensure the first two commented conditions
#           if list1[t]<list1[u] and list2[t]<list2[u]:
#               K+=0.0
#           elif list1[t]>list1[u] and list2[t]>list2[u]:
#               K+=0.0
        if list1[t]<list1[u] and list2[t]>list2[u]:
            K+=1.0
        elif list1[t]>list1[u] and list2[t]<list2[u]:
            K+=1.0
        elif list1[t]==list1[u] or list2[t]<list2[u]:
            K+=p
        else:
            continue
    return K
            
def GetLeftEigen(M):
    # Takes a square materix M and returns left-eigenvector
    print "Getting l-eigenvector"
    # normalize each row such that summation is equal to p = 1
    for i in range(len(M)):
        sum = np.sum(M[i])
        if sum == 0: continue
        M[i] = np.array([float(x)/sum for x in M[i,:]])
    #   Power iteration converges to dominant left eigenvector extremely fast
    l_eig2 =  PowerIterate(M)
    s2 = np.sum(l_eig2)
    l_eig2=l_eig2/s2
    return l_eig2
 
# A Power Iterater that to compute dominant left eigenvector of square matrix m
def PowerIterate(m,iteration=10):
    start =float(time.time())
    n = len(m)
    v = np.random.rand(n)
    v = v/sum(v)
    k = 0
    while iteration:
        v = np.dot(v,m)
        iteration -=1
    print "Power iteration finished in %.2f seconds" % (float(time.time())-start)
    return v

