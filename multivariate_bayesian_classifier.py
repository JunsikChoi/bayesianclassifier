import numpy as np
import math

# Read data from trn.txt

f = np.loadtxt('/home/jun/PycharmProjects/bayesianclassifier/data/trn.txt')

# Separate data columns from label column

f_trans = f.T
trn_data = f_trans[0:len(f_trans)-1,:]
trn_label = f_trans[-1,:]

# Generate Data Matrix D

D = trn_data.T #D : 60290x13
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
#r_1 : 60290 x 1

# label vector for class 2

r_2 = trn_label # r_2 : 60290x1
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
    #print math.exp(g_1 + C)
    #print math.exp(g_2 + C)
    if g_1>g_2:
        return 0.0
    else:
        return 1.0


ans = 0
for i in range(N):
    if quad_disc(D[i,:]) == trn_label[i]:
        ans += 1

print ans/float(N)