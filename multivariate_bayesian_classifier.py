import numpy as np
import math

# Read data from trn.txt

f = np.loadtxt('/home/jun/PycharmProjects/bayesianclassifier/data/trn.txt')

# Separate data columns from label column

f_trans = f.T
trn_data = f_trans[0:len(f_trans)-1,:]
trn_label = f_trans[-1,:]

# Generate Data Matrix D

D = np.asmatrix(trn_data.T)

# number of examples (N) and number of features (d)

N,d = D.shape

# Generate label vector r_i for class 1 and class 2
# Class 1 = (label == 0) & Class 2 = (label ==1)

# label vector for class 1

r_1 = np.array([])

for i in range(len(trn_label)):
    if trn_label[i] == 0:
        r_1 = np.append(r_1,1)
    else:
        r_1 = np.append(r_1,0)

r_1 = np.asmatrix(r_1).T
# label vector for class 2

r_2 = trn_label
r_2 = np.asmatrix(r_2).T
# Generate Prior probability P_i for Class 1 and Class 2

P_1 = np.sum(r_1)/float(N)
P_2 = np.sum(r_2)/float(N)

# Generate sample mean vector m_i for Class 1 and Class 2

m_1 = np.array([])
m_2 = np.array([])

for i in range(d):
    m_1 = np.append(m_1,D.T[i,:]*(r_1)/float(np.sum(r_1)))
    m_2 = np.append(m_2,D.T[i,:]*(r_2)/float(np.sum(r_2)))

m_1 = np.asmatrix(m_1).T
m_2 = np.asmatrix(m_2).T

# Generate Sample covariance matrix S_i for Class 1 and Class 2

# Delete elements from different class

D_1 = D[r_1==1,:]
D_2 = D[r_2==1,:]

# Calculate S_i

S_1 = np.cov(D_1.T)
S_2 = np.cov(D_2.T)

S_1 = np.asmatrix(S_1)
S_2 = np.asmatrix(S_2)

'''
Generate quadratic discriminant function

g_i = x.T*W_i*x + w_i.T*x + w_i0

where

W_i = -inv(S_i)/2
w_i = inv(S_i)*m_i
w_i0 = -m_i.T*inv(S_i)*m_i/2 - log(det(S_i)) + log(P_i)
'''

W_1 = -np.linalg.inv(S_1)/2.0
W_2 = -np.linalg.inv(S_2)/2.0

w_1 = (np.linalg.inv(S_1))*m_1
w_2 = (np.linalg.inv(S_2))*m_2

w_10 = -m_1.T*(np.linalg.inv(S_1))*m_1-math.log(np.linalg.det(S_1))/2.0+math.log(P_1)
w_20 = -m_2.T*(np.linalg.inv(S_2))*m_2-math.log(np.linalg.det(S_2))/2.0+math.log(P_2)

def quad_disc(x):
    g_1 = x.T*W_1*x+w_1.T*x+w_10
    g_2 = x.T*W_2*x+w_2.T*x+w_20
    C = -(d/2.0)*math.log(2*math.pi)
    print math.exp(g_1 + C)
    print math.exp(g_2 + C)
    if g_1>g_2:
        return 0.0
    else:
        return 1.0


ans = 0
for i in range(N):
    if quad_disc(D[i,:].T) == trn_label[i]:
        ans += 1

