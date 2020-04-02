### Copyright (C) <2019>  <Dimah Dera>


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
from scipy.misc import imsave
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import time, sys
import pickle
import timeit
#from Adding_noise import random_noise

plt.ioff()
#%matplotlib inline
mnist = input_data.read_data_sets("./MNIST_data/", one_hot= True)
mnist.train.images.reshape(-1,28,28,1).shape
        
image_size = 28
patch_size = 5
num_channel = 1
num_labels = 10

# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
    
 
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides= [1,1,1,1], padding= "VALID")#without padding

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize= [1,2,2,1], strides= [1,2,2,1], padding= "SAME")


# The function that propagates the mean and covariance matrix of the variational distribution through the convolutional neural network that consists of
# one convolutional layer followed by activation function, one max-pooling layer, one fully-connected layer and a soft-max layer.
# Inputs:
#       x: the input image (tensor) from the dataset
#       conv1_weight_M: the variable that represents the mean tensor of the convolutional kernels
#       conv1_weight_sigma: the variable that represents the variance tensor of the convolutional kernels
#       fc1_weight_mu: the variable that represents the mean tensor of the fully-connected layer weights
#       fc1_bias_mu: the variable that represents the mean tensor of the fully-connected layer bias 
#       fc1_weight_sigma: the variable that represents the variance tensor of the fully-connected layer weights
#       fc1_bias_sigma: the variable that represents the variance tensor of the fully-connected layer bias
#       new_size: The size of the feature map after the max-pooling layer # ((image_size - filter_size + 2P)/stride)+1) 
#       image_size: the size of the input image (28 x 28)
#       num_channel:the number of channels in the input image. It is 1 for MNIST data as the dataset are grayscale images
#       patch_size: the size of the convolutional kernels (5 x 5)
#       num_filters: the number of kernels or filters in the convolutional layer (32)
#       num_labels: the number of classes in the dataset. It is 10 for the MNIST dataset
# Outputs: 
#       mu_y: The output mean vector (predictive mean).
#       mu_f_fc1: The logit or class score.
#       sigma_y: The output covariance matrix (predictive covariance matrix).
#       sigma_f: The covariance matrix before the soft-max layer.


def Model_with_uncertainty_computation(x, conv1_weight_M, conv1_weight_sigma, fc1_weight_mu, fc1_bias_mu,fc1_weight_sigma,
            fc1_bias_sigma, new_size,  image_size=28, patch_size=5, num_channel=1, num_filters=[32],num_labels=10):   

    ## Propagation through the convolutional layer
    # Propagate the mean    
    mu_z = conv2d(x, conv1_weight_M)# shape=[1, image_size,image_size,num_filters[0]]
    # Propagate the covariance matrix
    x_train_patches = tf.extract_image_patches(x, ksizes=[1, patch_size, patch_size, 1], strides=[1,1,1,1], rates=[1,1,1,1], padding = "VALID")# shape=[1, image_size, image_size, patch_size*patch_size*num_channel]
    x_train_matrix = tf.reshape(x_train_patches,[1, -1, patch_size*patch_size*num_channel])# shape=[1, image_size*image_size, patch_size*patch_size*num_channel]    
    X_XTranspose = tf.matmul(x_train_matrix, tf.transpose(x_train_matrix, [0, 2, 1]))# shape=[1, image_size*image_size, image_size*image_size ] dimension of vectorized slice in the tensor z
    X_XTranspose=tf.expand_dims(X_XTranspose, 1)
    X_XTranspose = tf.tile(X_XTranspose, [1, num_filters[0],1,1])#shape=[1, num_filter[0], image_size*image_size, image_size*image_size]
    X_XTranspose = tf.transpose(X_XTranspose, [0, 2, 3, 1])#shape=[1,image_size*image_size, image_size*image_size, num_filter[0]]
    X_XTranspose = tf.squeeze(X_XTranspose) #shape=[image_size*image_size, image_size*image_size, num_filter[0]]
    sigma_z = tf.multiply(tf.log(1. + tf.exp(conv1_weight_sigma)), X_XTranspose)#shape=[image_size*image_size, image_size*image_size, num_filter[0]]        
    ######################################################
    ######################################################   
    ## propagation through the activation function  
    # Propagate the mean  
    mu_g = tf.nn.relu(mu_z)#shape=[1, image_size,image_size,num_filters[0]]    
    # Propagate the covariance matrix
    activation_gradiant = tf.gradients(mu_g, mu_z)[0] # shape =[1, image_size,image_size,num_filters[0]]    
    gradient_matrix = tf.reshape(activation_gradiant,[1, -1, num_filters[0]])# shape =[1, image_size*image_size, num_filters[0]]    
    gradient_matrix=tf.expand_dims(gradient_matrix, 3)
    grad1 = tf.transpose(gradient_matrix, [0,2,1,3])
    grad2 = tf.transpose(grad1,[0,1,3,2])
    grad_square = tf.matmul(grad1,grad2)
    grad_square = tf.transpose(grad_square,[0,2,3,1])# shape =[1, image_size*image_size, image_size*image_size, num_filters[0]]
    grad_square = tf.squeeze(grad_square)    
    sigma_g = tf.multiply(sigma_z, grad_square)# shape =[image_size*image_size,image_size*image_size, num_filters[0]]   
    ######################################################
    ######################################################
    image_size = image_size - patch_size + 1    
    ## propagation through the max-pooling layer 
    # Propagate the mean      
    mu_p, argmax = tf.nn.max_pool_with_argmax(mu_g, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #shape=[1, new_size,new_size,num_filters[0]]
    # Propagate the covariance matrix    
    argmax1= tf.transpose(argmax, [0, 3, 1, 2])
    argmax2 = tf.reshape(argmax1,[1, num_filters[0], -1])#shape=[1, num_filters[0], new_size*new_size]
    new_sigma_g= tf.transpose(sigma_g, [2, 0, 1]) # shape =[num_filters[0], image_size*image_size, image_size*image_size]
    new_sigma_g =  tf.reshape(new_sigma_g,[ num_filters[0]*image_size*image_size,-1])    
    
    x_index = tf.mod(tf.floor_div(argmax2,tf.constant(num_filters[0],shape=[1,num_filters[0], new_size*new_size], dtype='int64')),tf.constant(image_size ,shape=[1,num_filters[0], new_size*new_size], dtype='int64')) 
    
    aux = tf.floor_div(tf.floor_div(argmax2,tf.constant(num_filters[0],shape=[1,num_filters[0], new_size*new_size], dtype='int64')),tf.constant(image_size,shape=[1,num_filters[0], new_size*new_size], dtype='int64'))    
    y_index = tf.mod(aux,tf.constant(image_size,shape=[1,num_filters[0],new_size*new_size], dtype='int64'))
    index = tf.multiply(y_index,image_size) + x_index
    index = tf.squeeze(index) # shape=[num_filters[0],new_size*new_size]      
    for i in range(num_filters[0]):
        if(i==0):
            ind1 = tf.gather(index, tf.constant(i))
            new_ind = ind1
        else:
            ind1 = (image_size*image_size*i)+ tf.gather(index, tf.constant(i))
            new_ind = tf.concat([new_ind,ind1],0) # shape=[num_filters[0]*new_size*new_size] 
    column1 = tf.gather(new_sigma_g,new_ind) 
    column2 = tf.reshape(column1, [num_filters[0], new_size*new_size, -1])
    column3 = tf.transpose(column2, [0, 2, 1]) 
    column4 = tf.reshape(column3, [num_filters[0]*image_size*image_size, -1])
    final = tf.gather(column4,new_ind)
    sigma_p = tf.reshape(final,[num_filters[0],new_size*new_size,new_size*new_size]) #shape=[num_filters[0],new_size*new_size, new_size*new_size] 
    ######################################################
    ######################################################
    # # Flatten the feature map after the max-pooling layer
    
    mu_b = tf.reshape(mu_p, [-1, new_size*new_size*num_filters[0]]) #shape=[1, new_size*new_size*num_filters[0]]        
    diag_elements = tf.matrix_diag_part(sigma_p) #shape=[num_filters[0], new_size*new_size]     
    diag_sigma_b =tf.reshape(diag_elements,[-1]) #shape=[new_size*new_size*num_filters[0]]       
    ######################################################
    ## propagation through the Fully Connected 
    # Propagate the mean    
    mu_f_fc1 = tf.matmul(mu_b, fc1_weight_mu) + fc1_bias_mu #shape=[1, num_labels]    
    # Propagate the covariance matrix
    fc1_weight_mu1 = tf.reshape(fc1_weight_mu, [num_filters[0],new_size*new_size,num_labels]) #shape=[num_filters[0],new_size*new_size,num_labels]
    fc1_weight_mu1T = tf.transpose(fc1_weight_mu1,[0,2,1]) #shape=[num_filters[0],num_labels,new_size*new_size]
    
    muhT_sigmab = tf.matmul(fc1_weight_mu1T,sigma_p)#shape=[num_filters[0],num_labels,new_size*new_size]
    muhT_sigmab_mu = tf.matmul(muhT_sigmab,fc1_weight_mu1)#shape=[num_filters[0],num_labels,num_labels]
    muhT_sigmab_mu = tf.reduce_sum(muhT_sigmab_mu, 0) #shape=[num_labels,num_labels]
       
    tr_sigma_b = tf.reduce_sum(diag_sigma_b)#shape=[1]
    mu_bT_mu_b = tf.reduce_sum(tf.multiply(mu_b, mu_b),1)  
    mu_bT_mu_b = tf.squeeze(mu_bT_mu_b)#shape=[1]    
    tr_sigma_h_sigma_b = tf.multiply(tf.log(1. + tf.exp(fc1_weight_sigma)), tr_sigma_b) # shape=[num_labels] 
    mu_bT_sigma_h_mu_b = tf.multiply(tf.log(1. + tf.exp(fc1_weight_sigma)), mu_bT_mu_b) # shape=[num_labels]     
    tr_sigma_h_sigma_b = tf.diag(tr_sigma_h_sigma_b) #shape=[num_labels,num_labels]
    mu_bT_sigma_h_mu_b = tf.diag(mu_bT_sigma_h_mu_b) #shape=[num_labels,num_labels]    
    sigma_f = tr_sigma_h_sigma_b + muhT_sigmab_mu + mu_bT_sigma_h_mu_b #shape=[num_labels,num_labels]     
    ######################################################  
    ######################################################       
    ## propagation through the soft-max layer    
    # Propagate the mean
    mu_y = tf.nn.softmax(mu_f_fc1) #shape=[1, num_labels] 
    # Propagate the covariance matrix
    # compute the gradient of softmax manually  
    grad_f1 = tf.matmul(tf.transpose(mu_y), mu_y)  
    diag_f = tf.diag(tf.squeeze(mu_y))
    grad_soft = diag_f - grad_f1 #shape=[num_labels,num_labels]
    sigma_y = tf.matmul(grad_soft,   tf.matmul(sigma_f, tf.transpose(grad_soft)))#shape=[num_labels,num_labels] 
    return mu_y, mu_f_fc1, sigma_y, sigma_f
   

# the log-likelihood of the objective function
# Inputs: 
#       y_pred_mean: The output mean vector (predictive mean).
#       y_pred_sd: The output covariance matrix (predictive covariance matrix).
#       y_test: The ground truth prediction vector
#       num_labels: the number of classes in the dataset. It is 10 for the MNIST dataset
# Output:
#       the expected log-likelihood term of the objective function  
def nll_gaussian(y_pred_mean,y_pred_sd,y_test, num_labels=10): 
    y_pred_sd_inv = tf.matrix_inverse(y_pred_sd + tf.diag(tf.constant(1e-3, shape=[num_labels])))   
    mu_ = y_pred_mean - y_test
    mu_sigma = tf.matmul(mu_ ,  y_pred_sd_inv) 
    ms = 0.5*tf.matmul(mu_sigma , tf.transpose(mu_))+ 0.5*tf.log(tf.matrix_determinant(y_pred_sd + tf.diag(tf.constant(1e-3, shape=[num_labels]))))      
    ms = tf.reduce_mean(ms)
    return(ms)


# Visualization function: It visualize nine images from the test set with the true and predicted value written under each image.
def plot_images(images, sigma_std, epoch, cls_true, cls_pred=None, noise=0.0):
    assert len(images[:,0,0,0]) == len(cls_true) == 9    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        # Get the i'th image and reshape the array.
        image = images[i,:,:,:]        
        # Add the adversarial noise to the image.
        image += noise          
        # Ensure the noisy pixel-values are between 0 and 1.
        image = np.clip(image, 0.0, 1.0)
        image1 = np.squeeze(image)
        # Plot image.
        ax.imshow(image1, cmap='binary', interpolation='nearest')
        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.   
    #gaussain_noise_var = 0.1   
    plt.savefig('./EVI_with_sigma_{}/epochs_{}/EVI_CNN_on_MNIST_Tested_images.png'.format(sigma_std, epoch))    
    plt.close(fig)       
 
 
#The main function: 
# Inputs: 
#      image_size: the size of the input image (28 x 28).
#      num_channel:the number of channels in the input image. It is 1 for MNIST data as the dataset are grayscale images.
#      patch_size: the size of the convolutional kernels (5 x 5).
#      num_filters: the number of kernels or filters in the convolutional layer.
#      num_labels: the number of classes in the dataset. It is 10 for the MNIST dataset.
#      batch_size: the batch size. It is hyper-parameters. We choose it to be 50. 
#      noise_limit: the level of the adversarial noise 
#      noise_l2_weight: the weighting parameters to regularize the adversarial cost function.
#      adversary_target_cls: the targeted attack class.
#      Adv_lr: the learning rate of adversarial cost function.
#      lr : the learning rate of the trainting.
#      init_sigma_std: the initial condition of the weights' variance.
#      init_mu: the initial condition of the weights' mean. 
#      epochs: the number of epochs. 
#      Adversarial_noise: decide if we want to apply adversarial noise or not (True or False).
#      Random_noise: decide if we want to apply Gaussian noise or not (True or False).
#      gaussain_noise_var: the variance of the Gaussian noise.
#      Training = decide if we want to do training or loading a trained model for testing (True or False).
#      repeat_initial: decide if we want to load previous initail coditions that we used before or not (True or False). 
#      continue_train: decide if we want to continue training or start training from scratch (True or False).
# Output:
#       The function saves the model after training, the weights, training, validation and test accuracies and a text file that shows for example the following: 
#                    Number of kernels : 32
#                   ---------------------------------
#                    Averaged Test Accuracy : 0.9744
#                    Best Test Accuracy : 1.0
#                    Total run time in sec : 6367.633367589209
#                    Averaged Training  Accuracy : 0.969938181818
#                    Averaged Validation Accuracy : 0.969890909091
#                   ---------------------------------
#                   ---------------------------------
#                    Initial std of sigma of the weights : -2.2
#                    Initial log(1+exp(sigma)) : 0.1
#                    Rate of Convergence : 10 epochs 
#       The main function also plot the training, validation and test accuracies vs. number of epochs, nine test images and their predicted and true classes and the adversarial example.
#   
def main_function(image_size=28, num_channel=1, patch_size=5, num_filters=[32],num_labels=10, batch_size = 50, noise_limit = 0.1,
         noise_l2_weight = 0.01, adversary_target_cls=3, Adv_lr = 0.8e-4, lr = 1e-4, init_sigma_std=-2.2 , init_mu=0.1, epochs =6,
        Adversarial_noise=True, Random_noise=False, gaussain_noise_var=0.1, Training = True, repeat_initial=False, continue_train = False):   
         
    init_std = 0.1    
    x = tf.placeholder(tf.float32, shape = (1, image_size,image_size,num_channel), name='x')
    y = tf.placeholder(tf.float32, shape = (1,num_labels), name='y_true')   
    
    y_true_cls = tf.argmax(y, axis=1) 
    new_size = np.int(np.ceil(np.float(image_size - patch_size+1)/2))# ((input_size-filter_size+2P)/stride)+1
    # initialize the variables
    if continue_train:
        file11 = open('./EVI_with_sigma_{}/epochs_{}/BayesCNN_weights_features.pkl'.format(init_std, epochs), 'rb')        
        conv_m, conv_s, fc_m, fc_s, b_m, b_s  =   pickle.load(file11) 
        file11.close()
        conv1_weight_M = tf.Variable(tf.constant(conv_m, shape= [patch_size,patch_size,num_channel,num_filters[0]], dtype=tf.float32))
        fc1_weight_mu = tf.Variable(tf.constant(fc_m, shape=[new_size*new_size*num_filters[0], num_labels], dtype=tf.float32))
        conv1_weight_sigma = tf.Variable(tf.constant(conv_s, shape=[num_filters[0]]))
        fc1_weight_sigma = tf.Variable(tf.constant(fc_s, shape=[num_labels]))
    elif repeat_initial: 
        init_file1 = open('./Initial_deterministic_CNN_weights_withValid.pkl', 'rb')
        W11, W22, b22 =   pickle.load(init_file1)
        init_file1.close()
        conv1_weight_M = tf.Variable(tf.constant(W11, shape= [patch_size,patch_size,num_channel,num_filters[0]], dtype=tf.float32))
        fc1_weight_mu = tf.Variable(tf.constant(W22, shape=[new_size*new_size*num_filters[0], num_labels], dtype=tf.float32))
        conv1_weight_sigma = tf.Variable(tf.constant(init_sigma_std, shape=[num_filters[0]]), name='conv1_weight_sigma')#, trainable=False)
        fc1_weight_sigma = tf.Variable(tf.constant(init_sigma_std, shape=[num_labels]), name='fc1_weight_sigma')#, trainable=False)
    else:
        conv1_weight_M = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channel, num_filters[0]], stddev=init_mu))#, trainable=False, collections=collection)     
        fc1_weight_mu = tf.Variable(tf.truncated_normal([new_size*new_size*num_filters[0], num_labels], stddev=init_mu))  
        #conv1_weight_sigma = tf.Variable(tf.constant(init_sigma_std, shape=[num_filters[0]]), name='conv1_weight_sigma')#, trainable=True)
        conv1_weight_sigma = tf.Variable(tf.random.uniform(shape=[num_filters[0]], minval=-12.,  maxval=init_sigma_std))
        #fc1_weight_sigma = tf.Variable(tf.constant(init_sigma_std, shape=[num_labels]), name='fc1_weight_sigma')#, trainable=True) 
        fc1_weight_sigma = tf.Variable(tf.random.uniform(shape=[num_labels], minval=-12.,  maxval=init_sigma_std))
        
    fc1_bias_mu = tf.Variable(tf.constant(0.00001, shape=[num_labels]))
    fc1_bias_sigma = tf.Variable(tf.constant(0.00001, shape=[num_labels])) 
    
    if Adversarial_noise:
        ADVERSARY_VARIABLES = 'adversary_variables'
        collections = [tf.GraphKeys.GLOBAL_VARIABLES, ADVERSARY_VARIABLES]
        x_noise = tf.Variable(tf.zeros([image_size, image_size, num_channel]),  name='x_noise', trainable=False, collections=collections) 
        x_noise_clip = tf.assign(x_noise, tf.clip_by_value(x_noise, -noise_limit,  noise_limit))
        x_noisy_image = x + x_noise        
        x_noisy_image = tf.clip_by_value(x_noisy_image, 0.0, 1.0) 
        print('Call the model ....')
        prediction, class_score1, output_sigma, sigma_f = Model_with_uncertainty_computation(x_noisy_image, conv1_weight_M, conv1_weight_sigma,   fc1_weight_mu, fc1_bias_mu,fc1_weight_sigma, fc1_bias_sigma, new_size)
        adversary_variables = tf.get_collection(ADVERSARY_VARIABLES)  
        l2_loss_noise = noise_l2_weight * tf.nn.l2_loss(x_noise)  
    else:
        print('Call the model ....')
        prediction, class_score1, output_sigma, sigma_f= Model_with_uncertainty_computation(x, conv1_weight_M, conv1_weight_sigma,   fc1_weight_mu, fc1_bias_mu,fc1_weight_sigma, fc1_bias_sigma, new_size)
    ######################################################     
    # KL-divergence regularization term
    f_s = tf.log(1. + tf.exp(fc1_weight_sigma))
    c_s = tf.log(1. + tf.exp(conv1_weight_sigma))
    
    kl_loss_conv1 =-0.5*tf.reduce_mean(patch_size*patch_size+(patch_size*patch_size)*tf.log(c_s)-tf.reduce_sum(tf.abs(conv1_weight_M)) - (patch_size*patch_size)*c_s, axis=-1) 
    kl_loss_fc1 =-0.5*tf.reduce_mean(new_size*new_size*num_filters[0]+(new_size*new_size*num_filters[0])*tf.log(f_s)-tf.reduce_sum(tf.abs(fc1_weight_mu)) - (new_size*new_size*num_filters[0])*f_s, axis=-1) 
    
    ######################################################   
    output_sigma1 = tf.clip_by_value(t=output_sigma, clip_value_min=tf.constant(1e-10),
                                   clip_value_max=tf.constant(1e+10))
    y_pred_cls = tf.argmax(prediction, axis=1)     
    tau = 0.002
    print('Compute Cost Function ....')  
    log_likelihood = nll_gaussian(prediction, output_sigma1, y)   
    loss = log_likelihood +  tau *(kl_loss_conv1 + kl_loss_fc1 )
    print('Compute Optm ....')
    global_step = tf.Variable(0, trainable=False)
    #starter_learning_rate = 1e-3
   # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
    #                                       1000, 0.96, staircase=True)
    #optm = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)   
    optm = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss) 
    print('Compute Accuracy ....')
    corr = tf.equal(y_pred_cls , tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
    if Adversarial_noise:          
        loss_adversary = log_likelihood + l2_loss_noise
        optimizer_adversary = tf.train.AdamOptimizer(learning_rate= Adv_lr).minimize(loss_adversary, var_list=adversary_variables)#AdamOptimizer
        
    print('Initialize Variables ....')   
    saver = tf.train.Saver()  # initialize saver for saving weight and bias values
    init = tf.global_variables_initializer()
    if not os.path.exists('./EVI_with_sigma_{}/epochs_{}/model.meta'.format(init_std, epochs)):     
        sess = tf.Session()# initialize tensorflow session    
        sess.run(init)
        #saver.restore(sess, tf.train.latest_checkpoint('./Extended_VI_CNN/'))
        if Adversarial_noise:
            sess.run(tf.variables_initializer([x_noise])) 

        start = timeit.default_timer()
        print("Starting training ....") 
        train_acc = np.zeros(epochs) 
        valid_acc = np.zeros(epochs) 
        
        for k in range(epochs): 
            print(k+1 ,'/', epochs)
            acc1 = 0 
            acc_valid1 = 0 
            err1 = 0
            err_valid1 = 0           
            
            for minibatch in range(int(mnist.train.num_examples/(batch_size))):
                update_progress(minibatch / int(mnist.train.num_examples / (batch_size)) )   
                batch_data, batch_labels = mnist.train.next_batch(batch_size)
                valid_batch_data, valid_batch_labels = mnist.validation.next_batch(batch_size)
                valid_batch_data = valid_batch_data.reshape(-1,28,28,1)
                batch_data = batch_data.reshape(-1,28,28,1)
                acc2 = 0  
                acc_valid2 = 0 
                err2 = 0
                err_valid2 = 0
                start_time = time.time()
                for i in range(batch_size):
                    xx_ = np.expand_dims(batch_data[i,:,:,:],axis=0) 
                    yy_ = np.expand_dims(batch_labels[i,:], axis=0)  
                    valid_xx_ = np.expand_dims(valid_batch_data[i,:,:,:],axis=0)
                    valid_yy_ = np.expand_dims(valid_batch_labels[i,:], axis=0)                  
                    sess.run([optm],feed_dict = {x: xx_, y: yy_ })                   
                    acc = sess.run(accuracy ,feed_dict = {x: xx_, y: yy_}) 
                    acc_valid = sess.run(accuracy, feed_dict = {x:valid_xx_, y:valid_yy_})                 
                    acc2 += acc                                      
                    acc_valid2 += acc_valid
              
                    if (minibatch % 100 == 0) or (minibatch == (int(mnist.train.num_examples/(batch_size)) - 1)):
                        err = sess.run(loss, feed_dict = {x: xx_, y: yy_}) 
                        va_err = sess.run(loss ,feed_dict = {x:valid_xx_, y:valid_yy_})  
                        err2 += err
                        err_valid2 +=  va_err
                    
                end_time = time.time()                
                if (minibatch % 100 == 0) or (minibatch == (int(mnist.train.num_examples/(batch_size)) - 1)): 
                    print('Train Acc:', acc2/batch_size,  'Valid Acc:', acc_valid2/batch_size, 'Train err:', err2/batch_size, 'valid err:', err_valid2/batch_size)                   
                acc1 += acc2                           
                acc_valid1 += acc_valid2
                ################################
            train_acc[k] = acc1 / (int(mnist.train.num_examples))       
            valid_acc[k] = acc_valid1/ (int(mnist.train.num_examples) )              
            print('Training Acc  ', train_acc[k])
            print('Validation Acc  ', valid_acc[k])
            
        if Adversarial_noise:
            print('Building Adversarial Noise .....')            
            for minibatch1 in range(int(mnist.train.num_examples/(batch_size))):
                update_progress(minibatch1/int(mnist.train.num_examples/(batch_size)))
                batch_data, batch_labels = mnist.train.next_batch(batch_size)
                batch_data = batch_data.reshape(-1,28,28,1)
                y_true_batch = np.zeros_like(batch_labels)
                y_true_batch[:, adversary_target_cls] = 1.0
                
                for i in range(batch_size):
                    xx_ = np.expand_dims(batch_data[i,:,:,:],axis=0)
                    yy_ = np.expand_dims(y_true_batch[i,:], axis=0)
                    sess.run([optimizer_adversary], feed_dict={x: xx_, y: yy_ })
                    sess.run(x_noise_clip)             
            ####################################            
        stop = timeit.default_timer()
        print('Total Training Time: ', stop - start)
        if (epochs > 1):
            fig = plt.figure(figsize=(15,7))
            plt.plot(train_acc, 'b', label='Training acc')
            plt.plot(valid_acc,'r' , label='Validation acc')
            plt.ylim(0, 1.1)
            plt.title("Extended Variational Inference on MNIST Data")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend(loc='lower right')
            plt.savefig('./EVI_with_sigma_{}/epochs_{}/EVI_CNN_on_MNIST_Data_acc.png'.format(init_std, epochs))         
            plt.close(fig)       
    
        print('Training is Completed')
        if (epochs==1):
            print('Training Accuracy', train_acc)
            print('Validation Accuracy', valid_acc)
        else:
            print('Training Accuracy',np.mean(train_acc))
            print('Validation Accuracy',np.mean(valid_acc))
        print('---------------------')
        f = open('./EVI_with_sigma_{}/epochs_{}/training_validation_acc.pkl'.format(init_std, epochs), 'wb')         
        pickle.dump([train_acc, valid_acc], f)                                                   
        f.close()                                                               
        save_path = saver.save(sess,'./EVI_with_sigma_{}/epochs_{}/model'.format(init_std, epochs))              
    else:  
        sess = tf.Session()
        saver.restore(sess, tf.train.latest_checkpoint('./EVI_with_sigma_{}/epochs_{}/'.format(init_std, epochs)))   
   
    nm_test_ima = 100    
    test_acc = np.zeros(nm_test_ima)      
    for l in range(nm_test_ima):
        update_progress(l/nm_test_ima)
        #print(l+1, '/', nm_test_ima)
        test_data, test_labels = mnist.test.next_batch(batch_size)
        test_data = test_data.reshape(-1,28,28,1)          
        cls_true =  np.argmax(test_labels, axis = 1)   
        cls_pred = np.zeros_like(cls_true)       
        accu1 = 0       
        for i in range(batch_size):
            #update_progress(i/batch_size)
            if Random_noise:
                test_data[i,:,:,:] = random_noise((test_data[i,:,:,:]),mode='gaussian', var=gaussain_noise_var)            
            test_xx_ = np.expand_dims(test_data[i,:,:,:],axis=0)          
            test_yy_ = np.expand_dims(test_labels[i,:], axis=0) 
            cls_pred[i], accu  = sess.run( [y_pred_cls, accuracy], feed_dict={x: test_xx_, y: test_yy_})           
            accu1 += accu           
        test_acc[l] = accu1/batch_size
        #print('Test Acc', test_acc[l])            
        
    fig  = plt.figure(figsize=(15,7))
    plt.plot(test_acc, 'r', label='Test acc')
    plt.ylim(0, 1.2)   
    plt.title("Bayesian CNN on MNIST Data Test Acc")
    plt.xlabel("Test Image Number")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right')
    plt.savefig('./EVI_with_sigma_{}/epochs_{}/EVI_CNN_on_MNIST_Data_test_acc.png'.format( init_std, epochs))      
    plt.close(fig) 
    print('Maximum Test Accuracy', np.amax(test_acc))
    print('Average Test Accuracy', np.mean(test_acc))  
    
    f1 = open('./EVI_with_sigma_{}/epochs_{}/test_acc.pkl'.format( init_std, epochs), 'wb')    
    pickle.dump(test_acc, f1)
    f1.close()    
    ################################
    test_file = open('./test_images_mnist.pkl', 'rb')    
    img_test, test_label =   pickle.load(test_file) 
    test_file.close()
    img_test = img_test.reshape(-1,28,28,1) 
    cls_true =  np.argmax(test_label, axis = 1)
    cls_pred = np.zeros_like(cls_true) 
    uncert = np.zeros([9, num_labels, num_labels]) 
    mean_val = np.zeros([9, num_labels])
    class_score11 = np.zeros([9, num_labels])    
    sigma_f1 = np.zeros([9, num_labels, num_labels]) 
    for j in range(9):
        if Random_noise:
            img_test[j,:,:,:] = random_noise(img_test[j,:,:,:], mode='gaussian', var = gaussain_noise_var) 
        test_xx_ = np.expand_dims(img_test[j,:,:,:],axis=0)
        test_yy_ = np.expand_dims(test_label[j,:], axis=0)   
        cls_pred[j],mean_val[j,:], class_score11[j,:], uncert[j,:,:], sigma_f1[j,:,:] = sess.run([y_pred_cls, prediction, class_score1, output_sigma, sigma_f], feed_dict={x: test_xx_, y: test_yy_})
        

    images = img_test
    f2 = open('./EVI_with_sigma_{}/epochs_{}/test_uncert_info.pkl'.format(init_std, epochs), 'wb')    
    pickle.dump([uncert, mean_val, cls_pred, class_score11, sigma_f1], f2)
    f2.close()
    
    if Adversarial_noise:
        adver_example = sess.run(x_noise)
        file1 = open('./EVI_with_sigma_{}/epochs_{}/Bayes_CNN_x_noise.pkl'.format(init_std, epochs), 'wb')        
        pickle.dump( adver_example , file1)
        file1.close()
    
    con_m, conv_s, fc_m, fc_s, b_m, b_s  = sess.run([conv1_weight_M, conv1_weight_sigma, fc1_weight_mu, fc1_weight_sigma, fc1_bias_mu, fc1_bias_sigma])
    file11 = open('./EVI_with_sigma_{}/epochs_{}/BayesCNN_weights_features.pkl'.format(init_std, epochs), 'wb')    
    pickle.dump([ con_m, conv_s, fc_m, fc_s, b_m, b_s], file11)
    file11.close()
        
    if Adversarial_noise:  
        noise = sess.run(x_noise)
        noise = noise.reshape(28,28,1)
        noise1 = np.squeeze(noise)
        print("Noise:")
        print("- Min:", noise1.min())
        print("- Max:", noise1.max())
        
        # Plot the noise.
        plt.axis('off')
        plt.imsave('./EVI_with_sigma_{}/epochs_{}/EVI_on_MNIST_noise.png'.format( init_std, epochs), noise1,  cmap='seismic', vmin=-1.0, vmax=1.0)       
        plot_images(images=images[0:9,:,:,:], sigma_std=init_std, epoch=epochs, cls_true=cls_true[0:9], cls_pred=cls_pred[0:9],  noise=noise)
    else:
        plot_images(images=images[0:9,:,:,:], sigma_std=init_std, epoch=epochs, cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])
        
    textfile = open('./EVI_with_sigma_{}/epochs_{}/Related_info.txt'.format( init_std, epochs),'w')    
    textfile.write(' Number of kernels : ' +str(num_filters[0]))     
    textfile.write("\n---------------------------------")
    textfile.write("\n Averaged Test Accuracy : "+ str(np.mean(test_acc))) 
    textfile.write("\n Best Test Accuracy : "+ str(np.amax(test_acc))) 
    if Training: 
        textfile.write('\n Total run time in sec : ' +str(stop - start))
        if(epochs == 1):
            textfile.write("\n Averaged Training  Accuracy : "+ str( train_acc))
            textfile.write("\n Averaged Validation Accuracy : "+ str(valid_acc ))
        else:
            textfile.write("\n Averaged Training  Accuracy : "+ str(np.mean(train_acc)))
            textfile.write("\n Averaged Validation Accuracy : "+ str(np.mean(valid_acc)))
    textfile.write("\n---------------------------------")
    if Random_noise:
        textfile.write('\n Random Noise var: '+ str(gaussain_noise_var))
    if Adversarial_noise:
        textfile.write('\n Noise limit: '+ str(noise_limit))        
    textfile.write("\n---------------------------------")
    textfile.write('\n Initial std of sigma of the weights : ' +str(init_sigma_std)) 
    textfile.write('\n Initial log(1+exp(sigma)) : ' +str(init_std)) 
    textfile.write('\n Rate of Convergence : ' + str(epochs)+ ' epochs')
    textfile.close()
    sess.close()          
if __name__ == '__main__':
    main_function()    
