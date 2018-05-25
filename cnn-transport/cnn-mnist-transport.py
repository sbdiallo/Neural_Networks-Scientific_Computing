
# coding: utf-8

# In[41]:



import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import csv


# In[42]:


### devbut fonctions predef
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) 
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape) 
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    

def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b) 

def full_layer(input, size):
    in_size = int(input.get_shape()[1]) 
    W = weight_variable([in_size, size]) 
    b = bias_variable([size])
    return tf.matmul(input, W) + b
       
#### fin fonctions predef


# In[43]:


# Definition constantes

MINIBATCH_SIZE = 50
STEPS = 1000
batch_size = 1000

# marche pour un multiple
MULTIPLE = 1
IMAGE_SIZE = 28 * MULTIPLE
TAILLE_MAX = IMAGE_SIZE * IMAGE_SIZE
TAILLE_MIN = IMAGE_SIZE
num_classes = 10

training_data = "/tmp/csv/mnist_butch_train.csv"
testing_data = "/tmp/csv/mnist_butch_test.csv"


# In[44]:


# debut def reseau

x = tf.placeholder(tf.float32, shape=[None, TAILLE_MAX])

y_ = tf.placeholder(tf.float32, shape=[None, num_classes])

x_image = tf.reshape(x, [-1, TAILLE_MIN, TAILLE_MIN, 1])

conv1 = conv_layer(x_image, shape=[5*MULTIPLE, 5*MULTIPLE, 1, 32]) #32
conv1_pool = max_pool_2x2(conv1)

conv2 = conv_layer(conv1_pool, shape=[5*MULTIPLE,5*MULTIPLE, 32, 64]) # 32,64
conv2_pool = max_pool_2x2(conv2)

conv2_flat = tf.reshape(conv2_pool, [-1, 7*7*64*MULTIPLE*MULTIPLE]) 
full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

keep_prob = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

y_conv = full_layer(full1_drop, num_classes)

##### fin def reseau


# In[45]:



#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels=y_))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,logits=y_conv))
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv,y_))

train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy) 

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

test_accuracy = 0

with tf.Session() as sess: 
    
    sess.run(tf.global_variables_initializer())
    
    compteur_gene = 0
    inf = open(training_data)
    inf.seek(0)
    
    for i in range(STEPS): 
        
        bu0=np.array(MINIBATCH_SIZE*[np.array(TAILLE_MAX *[np.float32(0.)])])
        bu1=np.array(MINIBATCH_SIZE*[np.array(num_classes*[np.float64(0.)])])
        butch=[bu0,bu1]
        
        
        if compteur_gene > 3000:  # A généraliser
            inf.seek(0)
            compteur_gene=0
        compteur_gene = compteur_gene + MINIBATCH_SIZE
        
        j1=0
        while j1 < MINIBATCH_SIZE:
            zzz = inf.readline(-1).split(",")   
        
            for jj in range(TAILLE_MAX):
                butch[0][j1][jj] = zzz[jj+1]
                    
            for jj in range(num_classes):
                butch[1][j1][jj]=0
            butch[1][j1][int(zzz[0])]=1
                
            j1=j1+1
    
                    
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: butch[0],y_: butch[1],keep_prob: 1.0}) 
            
            print "step {}, training accuracy {}".format(i, train_accuracy)
            
        sess.run(train_step, feed_dict={x: butch[0], y_: butch[1], keep_prob: 0.5}) 
        
 
    bu0 = np.array(batch_size *[np.array(784*[np.float32(0.)])])
    bu1 = np.array(batch_size *[np.array(num_classes*[np.float64(0.)])])
    botch = [bu0,bu1]
    
    with open(testing_data) as inf:
        j1 = 0
       
        while j1 < batch_size:
                
            zzz = inf.readline(-1).split(",") 
            n = len(zzz) - 1 
            
            for jj in range(TAILLE_MAX):
                if jj < n :
                    botch[0][j1][jj] = zzz[jj+1]
            
            for jj in range(num_classes):
                botch[1][j1][jj] = 0
            
            botch[1][j1][int(zzz[0])]=1
                
            j1=j1+1   
            
            
    test_accuracy= sess.run(accuracy, feed_dict={x: botch[0],y_: botch[1],keep_prob: 1.0}) 

print "test accuracy: {}".format(test_accuracy)
print ("This is the END")


# In[ ]:




 

