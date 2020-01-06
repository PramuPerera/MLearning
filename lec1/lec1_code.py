#https://www.python-course.eu/neural_network_mnist.php
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity

image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = ""
train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")


train_imgs = np.asfarray(train_data[:, 1:]) 
train_labels = np.squeeze(np.array(train_data[:, :1])).astype(int)
test_imgs = np.asfarray(test_data[:, 1:]) 
test_labels = np.squeeze(np.array(test_data[:, :1])).astype(int)

zero_train = train_imgs[train_labels==1,:]
one_train = train_imgs[train_labels==0,:]
zero_test = test_imgs[test_labels==1,:]
one_test = test_imgs[test_labels==0,:]


#counting white pixels
train_count_zero = np.sum(zero_train>25,axis=1)
train_count_one = np.sum(one_train>25,axis=1)
test_count_zero = np.sum(zero_test>25,axis=1)
test_count_one = np.sum(one_test>25,axis=1)

kde1 = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(train_count_one.reshape(-1,1))
kde0 = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(train_count_zero.reshape(-1,1))

onescores_one = np.array([ kde1.score(i.reshape(-1,1)) for i in test_count_one])
onescores_zero = np.array( [ kde0.score(i.reshape(-1,1)) for i in test_count_one])
zeroscores_one = np.array([ kde1.score(i.reshape(-1,1)) for i in test_count_zero])
zeroscores_zero =  np.array([ kde0.score(i.reshape(-1,1)) for i in test_count_zero])


plt.subplot(3,1,1)
plt.hist(train_count_zero,100)
plt.title('Histogram of Pixel Count for Digit 0')
plt.subplot(3,1,2)
plt.hist(train_count_one,100)
plt.title('Histogram of Pixel Count for Digit 1')
plt.subplot(3,1,3)
sns.distplot(train_count_zero)
sns.distplot(train_count_one)
plt.title('Probability Density Function')
plt.show()


t = 118.5
print('Accuracy for Maximum Likelihood:')
print((np.mean((onescores_one>onescores_zero)==True)+np.mean((zeroscores_one>zeroscores_zero)==False))/2)
print('Accuracy for Naive Method:')
print(np.mean(test_count_one>t)+np.mean(test_count_one<t)/2)

