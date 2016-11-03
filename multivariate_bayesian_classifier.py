#import needed modules

import numpy as np
import math
import matplotlib.pyplot as plt

# Read training data and test data from Path

f = np.loadtxt('/home/jun/PycharmProjects/bayesianclassifier/data/trn.txt')
g = np.loadtxt('/home/jun/PycharmProjects/bayesianclassifier/data/tst.txt')

# Separate data columns from label column

# training data

f_trans = f.T
trn_data = f_trans[0:len(f_trans)-1,:]
trn_label = f_trans[-1,:]

# Test data

g_trans = g.T
tst_data = g_trans[0:len(g_trans)-1,:]
tst_label = g_trans[-1,:]

T = tst_data.T # test data matrix

N_t,d_t = T.shape

# Generate training Data Matrix D

D = trn_data.T #Training data matrix D : 60290x13

# number of examples (N) and number of features (d)

N,d = D.shape

# Generate label vector r_i for class 1 and class 2
# Class 1 = (label == 0) & Class 2 = (label ==1)

# Training label vector

# label vector for class 1

r_1 = np.array([])

for i in range(len(trn_label)):
    if trn_label[i] == 0:
        r_1 = np.append(r_1,1)
    else:
        r_1 = np.append(r_1,0)
#r_1 : 60290 x 1

# label vector for class 2

r_2 = trn_label # r_2 : 60290x1

# Testing label vector

# label vector for class 1

t_1 = np.array([])

for i in range(len(tst_label)):
    if tst_label[i] == 0:
        t_1 = np.append(t_1,1)
    else:
        t_1 = np.append(t_1,0)

# label vector for class 2

t_2 = tst_label

# Generate Prior probability P_i for Class 1 and Class 2

P_1 = np.sum(r_1)/float(N) # P_1 : ~ 0.452
P_2 = np.sum(r_2)/float(N) # P_2 : ~ 0.547

# Generate sample mean vector m_i for Class 1 and Class 2

m_1 = np.array([])
m_2 = np.array([])

for i in range(d):
    m_1 = np.append(m_1,D.T[i,:].dot(r_1)/float(np.sum(r_1)))
    m_2 = np.append(m_2,D.T[i,:].dot(r_2)/float(np.sum(r_2)))

#m_1 : 13x1
#m_2 : 13x1

# Generate Sample covariance matrix S_i for Class 1 and Class 2

# Delete elements from different class

D_1 = D.T[:,r_1==1]
D_2 = D.T[:,r_2==1]

D_1 = D_1.T # D_1 : 27262 x 13
D_2 = D_2.T # D_2 : 33028 x 13

# Calculate S_i

S_1 = np.cov(D_1.T) # S_1 : 13x13
S_2 = np.cov(D_2.T) # s_2 : 13x13


'''
Generate quadratic discriminant function

g_i = x.T*W_i*x + w_i.T*x + w_i0

where

W_i = -inv(S_i)/2
w_i = inv(S_i)*m_i
w_i0 = -m_i.T*inv(S_i)*m_i/2 - log(det(S_i)) + log(P_i)
'''

W_1 = np.linalg.inv(S_1)*(-0.5)
W_2 = np.linalg.inv(S_2)*(-0.5)

w_1 = (np.linalg.inv(S_1)).dot(m_1)
w_2 = (np.linalg.inv(S_2)).dot(m_2)

w_10 = -(0.5)*m_1.dot(np.linalg.inv(S_1)).dot(m_1)-math.log(np.linalg.det(S_1))*0.5+math.log(P_1)
w_20 = -(0.5)*m_2.dot(np.linalg.inv(S_2)).dot(m_2)-math.log(np.linalg.det(S_2))*0.5+math.log(P_2)

def quad_disc(x):
    g_1 = x.dot(W_1).dot(x)+w_1.dot(x)+w_10
    g_2 = x.dot(W_2).dot(x)+w_2.dot(x)+w_20
    C = -(0.5*d)*math.log(2*math.pi)
    if g_1>g_2:
        return 0.0
    else:
        return 1.0


# Calculate Error rate using test data and elements for confusion matrix

ans = 0
TN = 0
TP = 0
FN = 0
FP = 0
TPR = np.array([])
FPR = np.array([])

result = np.array([])

for i in range(N_t):
    clf = quad_disc(T[i,:])
    result = np.append(result,clf)
    if clf == tst_label[i]:
        ans += 1
emp_error_rate = (1-ans/float(N_t))*100

for i in range(N_t):
    if (tst_label[i]==0)&(result[i]==0):
        TN += 1
    elif (tst_label[i]==0)&(result[i]==1):
        FP += 1
    elif (tst_label[i]==1)&(result[i]==1):
        TP += 1
    else:
        FN += 1

    TPR = np.append(TPR,TP)
    FPR = np.append(FPR,FP)

TPR = TPR/float(TP+FN)
FPR = FPR/float(FP+TN)

print "True Negative : "+str(TN)
print "True Positive : "+str(TP)
print "False Positive : "+str(FP)
print "False Negative : "+str(FN)
print "Empirical Error for test data!: "+str(emp_error_rate)+"%"

# Draw Receiver Operating Characteristics (ROC) Curve

plt.figure(figsize=(4, 4), dpi=80)
plt.xlabel("FPR", fontsize=14)
plt.ylabel("TPR", fontsize=14)
plt.title("ROC Curve", fontsize=14)
plt.plot(FPR,TPR,linewidth=2)
plt.show()



