import numpy as np

# Read data from trn.txt

f = np.loadtxt('C:/Users/leich/PycharmProjects/bayesianclassifier/data/trn.txt')
f_trans = f.T
trn_data = f_trans[0:len(f_trans)-1,:]
trn_label = f_trans[-1,:]

# Making Data Matrix

trn_data_matrix = np.asmatrix(trn_data.T)

# Making mean vector
'''
print trn_data_matrix.T.shape
for j in range(len(trn_data[:,0])):
    for i in range(len(trn_data[j, :])):
        sum += (trn_data[j, i])
    mean = sum / len(trn_data[j, :])
    mean_vector = np.append(mean_vector,mean)
print mean_vector
'''
d,N = trn_data_matrix.T.shape
sum = 0
mean_vector = np.array([])
for j in range(d):
    for i in range(N):
        sum += (trn_data_matrix.T[j, i])
    mean = sum / N
    mean_vector = np.append(mean_vector,mean)
print mean_vector
# Making Covariance Matrix

#print "See"
#print trn_data_matrix
#print trn_data_matrix.T
cov_matrix = np.cov(trn_data_matrix.T)
#print cov_matrix
#print cov_matrix.shape
#print trn_data_matrix.shape
#print trn_data_matrix[1,:]
#print trn_data[0,:][0]
#print len(trn_data[:,0])
'''
for j in range(len(trn_data[:,0])): #13
    dev_sum = 0.
    for i in range(len(trn_data[j,:])):
        dev_sum += (trn_data[j.:][i] - mean_vector[j])**2
    var = dev_sum/float(i+1)
'''
'''
cov_matrix = np.array([])
dev_sum = 0
#print trn_data
for i in range(13):
    for j in range(13):
        print i,j
        for k in range(len(trn_data[i,:])):
            print k
            dev_sum += np.sum((trn_data[i,:]-mean_vector[i])*(trn_data[j,:]-mean_vector[j]))
        cov_ij = dev_sum/float(k+1)
        print cov_ij
        cov_matrix = np.append(cov_matrix,cov_ij)
        dev_sum = 0
print cov_matrix
print "end"
'''
