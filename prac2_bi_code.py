#https://www.python-course.eu/neural_network_mnist.php
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
import scipy.stats as stats

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

train_imgs = train_imgs[train_labels<=1,:]
train_labels = train_labels[train_labels<=1]
test_imgs= test_imgs[test_labels<=1,:]
test_labels = test_labels[test_labels<=1]

#counting white pixels
train_imgs = np.sum(train_imgs>25,axis=1)
test_imgs = np.sum(test_imgs>25,axis=1)

means=[]
stds =[]
scores = []
x = np.linspace(0, 350, 10000)
for i in range(2):
   train = train_imgs[train_labels==i]
   prior = len(train)
   means = np.mean(train)
   stds = np.std(train)
   plt.plot(x, stats.norm.pdf(x, means, stds))
   scores.append(stats.norm.pdf(test_imgs, means, stds)*prior)
plt.show()

scores= np.exp(scores)
scores = scores/np.sum(scores, axis=0)
predictions = np.argmax(scores, axis=0)
print('Accuracy : ' + str(np.mean(predictions==test_labels)))



