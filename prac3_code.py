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

accs= []
losses = []

#counting white pixels
w = np.random.normal(0,  0.1, (28*28, 10))
b = np.random.normal(0, 0.1, (1,10))
grad_w = np.zeros((28*28, 10))
grad_b = np.zeros((10, 1))

#Training
for epoch in range(10):
   loss = 0
   temp_grad_w = np.zeros((28*28, 10))
   temp_grad_b = np.zeros((10,1))
   for img, lbl in zip(train_imgs, train_labels):
      output = np.zeros((1,10))
      for sublbl in range(10):
         output[0,sublbl] = np.sum(img*w[:,sublbl])+b[0,sublbl]
      output = np.exp(output-np.max(output))
      sum_d = np.sum(output, 1)
      output = output/sum_d
      output+=0.01
      loss+= -1*np.log(output[0,lbl])     
      temp_grad_w += np.repeat(np.expand_dims(img,1),10, axis=1)/sum_d
      temp_grad_b[:] += 1/sum_d
      temp_grad_w[:,lbl] -= img
      temp_grad_b[lbl] -= 1   
   grad_w = temp_grad_w/len(train_imgs)
   grad_b = temp_grad_b/len(train_imgs)
   w = w-grad_w*0.01
   b = b-grad_b*0.01
   losses.append(loss/len(train_imgs))
   print('epoch' + str(epoch)+ ' loss = '+ str(loss/len(train_imgs)))
   acc = 0
   for img, lbl in zip(test_imgs, test_labels):   
      output = np.zeros((1,10))
      for sublbl in range(10):
         output[0,sublbl] = np.sum(img*w[:,sublbl])+b[0,sublbl]
      if np.argmax(output)==lbl:
         acc+=1
   accs.append(acc/len(test_imgs))
   print('Accuracy : ' + str(acc/len(test_imgs)))
plt.plot(losses)
plt.title('Evolution of Loss')
plt.show()

