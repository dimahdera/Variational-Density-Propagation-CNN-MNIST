import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
#from scipy.misc import imsave
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import time, sys
import pickle
import timeit
from Adding_noise import random_noise

plt.ioff()
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
    return tf.nn.conv2d(x, W, strides= [1,1,1,1], padding= "VALID")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize= [1,2,2,1], strides= [1,2,2,1], padding= "SAME")

def Model_with_uncertainty_computation(x, conv1_weight_M, conv1_weight_sigma, fc1_weight_mu, fc1_bias_mu,fc1_weight_sigma,
       fc1_bias_sigma, new_size, image_size=28, patch_size=5, num_channel=1, num_filters=[4],num_labels=10): 
    
    
    mu_z = conv2d(x, conv1_weight_M)# shape=[1, image_size,image_size,num_filters[0]]
    x_train_patches = tf.extract_image_patches(x, ksizes=[1, patch_size, patch_size, 1], strides=[1,1,1,1], rates=[1,1,1,1], padding = "VALID")# shape=[1, image_size, image_size, patch_size*patch_size*num_channel]
    x_train_matrix = tf.squeeze(tf.reshape(x_train_patches,[1, -1, patch_size*patch_size*num_channel]))# shape=[image_size*image_size, patch_size*patch_size*num_channel]
    X_XTranspose =   tf.reduce_sum(tf.multiply (x_train_matrix, x_train_matrix) ,1)# shape=[image_size*image_size]   
    X_XTranspose=tf.expand_dims(X_XTranspose, 1)
    X_XTranspose = tf.tile(X_XTranspose, [1, num_filters[0]])#shape=[image_size*image_size, num_filter[0] ]    
    sigma_z = tf.multiply(X_XTranspose , tf.log(1. + tf.exp(conv1_weight_sigma)))#shape=[image_size*image_size, num_filter[0]]        
    ###################################################### 
    # propagation through the activation function    
    image_size = image_size - patch_size + 1    
    n_g = image_size * image_size  
    sigma_z_diag = tf.transpose(sigma_z )#shape=[num_filter[0], image_size*image_size]
   # L_gg = tf.sqrt(tf.clip_by_value(sigma_z_diag, 1e-25, 1e+25)   )    

    non_zero = tf.not_equal(sigma_z_diag, 0.)
    l_gg_mask = tf.boolean_mask(sigma_z_diag, non_zero)
    out_sqrt = tf.sqrt(l_gg_mask)
    idx_l_gg = tf.to_int32(tf.where(non_zero))
    L_gg = tf.scatter_nd(idx_l_gg, out_sqrt, tf.shape(non_zero))

    L_g = tf.matrix_diag(L_gg)   

    L = np.sqrt(n_g)* L_g 
    x_hat1 = tf.transpose(tf.reshape(tf.squeeze(mu_z), [-1, num_filters[0]]))#shape=[num_filter[0], image_size*image_size]
        
    x_hat = tf.ones([1,1, image_size*image_size]) * tf.expand_dims(x_hat1, axis=-1)
    sigma_points1 = x_hat  + L 
    sigma_points2 = x_hat  - L
        
    mu_g1 = (1/(2*n_g))*(tf.reduce_sum(tf.nn.relu(sigma_points1) + tf.nn.relu(sigma_points2), 2)) #shape=[num_filter[0], image_size*image_size]    
    mu_g = tf.reshape(tf.transpose(mu_g1), [ image_size, image_size,num_filters[0] ]) #shape=[ image_size, image_size,num_filters[0] ]
    mu_g = tf.expand_dims(mu_g, 0) #shape=[1,image_size, image_size,num_filters[0] ]
    
    
    mu_g2 = tf.ones([1,1, image_size*image_size]) * tf.expand_dims(mu_g1, axis=-1) 
    
    P_ga1 = tf.nn.relu(sigma_points1) - mu_g2 #shape=[num_filter[0], image_size*image_size,image_size*image_size]    
    P_gb1 = tf.matmul(P_ga1, P_ga1,  transpose_b=True) 
  
    P_ga2 = tf.nn.relu(sigma_points2) - mu_g2 #shape=[num_filter[0], image_size*image_size,image_size*image_size]    
    P_gb2 = tf.matmul(P_ga2, P_ga2, transpose_b=True)
    P_gg = (1/(2*n_g))*(P_gb1 + P_gb2)#shape=[num_filter[0], image_size*image_size, image_size*image_size]     
    ######################################################
    # propagation through the pooling layer    
    mu_p, argmax = tf.nn.max_pool_with_argmax(mu_g, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #shape=[1, new_size,new_size,num_filters[0]]
        
    argmax1= tf.transpose(argmax, [0, 3, 1, 2])
    argmax2 = tf.reshape(argmax1,[1, num_filters[0], -1])#shape=[1, num_filters[0], new_size*new_size]    
    new_sigma_g =  tf.reshape(P_gg,[ num_filters[0]*image_size*image_size,-1])    
    
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
    # # propagation through the Fully Connected
    mu_b = tf.reshape(mu_p, [-1, new_size*new_size*num_filters[0]]) #shape=[1, new_size*new_size*num_filters[0]]    
    diag_elements = tf.matrix_diag_part(sigma_p) #shape=[num_filters[0], new_size*new_size]              
    diag_sigma_b =tf.reshape(diag_elements,[-1]) #shape=[new_size*new_size*num_filters[0]]       
    ######################################################
    # # propagation through the Fully Connected      
    mu_f_fc1 = tf.matmul(mu_b, fc1_weight_mu) + fc1_bias_mu #shape=[1, num_labels]   
    ######################################################  
    fc1_weight_mu1 = tf.reshape(fc1_weight_mu, [new_size*new_size,num_filters[0],num_labels]) #shape=[num_filters[0],new_size*new_size,num_labels]
    fc1_weight_mu1T = tf.transpose(fc1_weight_mu1,[1,2,0]) #shape=[num_filters[0],num_labels,new_size*new_size]
    
    muhT_sigmab = tf.matmul(fc1_weight_mu1T,sigma_p)#shape=[num_filters[0],num_labels,new_size*new_size]
    muhT_sigmab_mu = tf.matmul(muhT_sigmab,tf.transpose(fc1_weight_mu1,[1,0,2]))#shape=[num_filters[0],num_labels,num_labels]
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
    mu_y = tf.nn.softmax(mu_f_fc1) #shape=[1, num_labels] 
    # compute the gradient of softmax manually  
    grad_f1 = tf.matmul(tf.transpose(mu_y), mu_y)  
    diag_f = tf.diag(tf.squeeze(mu_y))
    grad_soft = diag_f - grad_f1 #shape=[num_labels,num_labels]
    sigma_y = tf.matmul(grad_soft,   tf.matmul(sigma_f, tf.transpose(grad_soft)))#shape=[num_labels,num_labels]   
    return mu_y, mu_f_fc1, sigma_y, sigma_f
   

def nll_gaussian(y_pred_mean,y_pred_sd,y_test, num_labels=10):    
    y_pred_sd_inv = tf.matrix_inverse(y_pred_sd + tf.diag(tf.constant(1e-3, shape=[num_labels])))   
    mu_ = y_pred_mean - y_test
    mu_sigma = tf.matmul(mu_ ,  y_pred_sd_inv) 
    ms = 0.5*tf.matmul(mu_sigma , tf.transpose(mu_))+ 0.5*tf.log(tf.matrix_determinant(y_pred_sd + tf.diag(tf.constant(1e-3, shape=[num_labels]))))      
    ms = tf.reduce_mean(ms)
    return(ms)
def plot_images(images, sigma_std, epoch, cls_true,path, cls_pred=None, noise=0.0):
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

    plt.savefig(os.path.join(path, "UVI_CNN_on_MNIST_adv.png"))   
    plt.close(fig)                                                       

def main_function(image_size=28, num_channel=1, patch_size=5, num_filters=[4],num_labels=10, batch_size = 50, noise_limit = 0, 
                  noise_l2_weight = 0.01, adversary_target_cls=3,
                  init_sigma_std=-2.3 , init_mu=0.1, epochs = 1, Adversarial_noise=False, Random_noise=False, gaussain_noise_std=0.48,
                  Training = True, repeat_initial=False):
    init_std = 0.1
    gaussain_noise_var = gaussain_noise_std**2    
    saved_result_path = './UVI_CNN_with_sigma_{}_noise_{}/epochs_{}/'.format(init_std, noise_limit, epochs)#
    
    x = tf.placeholder(tf.float32, shape = (1, image_size,image_size,num_channel), name='x')
    y = tf.placeholder(tf.float32, shape = (1,num_labels), name='y_true')   
    y_true_cls = tf.argmax(y, axis=1)
    new_size = np.int(np.ceil(np.float(image_size - patch_size+1)/2))# ((input_size-filter_size+2P)/stride)+1  
    if repeat_initial:

        init_file1 = open('./UVI_CNN_with_sigma_{}_noise_{}/epochs_{}/UVI_CNN_weights_features.pkl'.format(init_std, noise_limit, 4), 'rb')
        con_m, conv_s, fc_m, fc_s, b_m, b_s =  pickle.load(init_file1)
        init_file1.close()
        conv1_weight_M = tf.Variable(tf.constant(con_m, shape= [patch_size,patch_size,num_channel,num_filters[0]], dtype=tf.float32))
        fc1_weight_mu = tf.Variable(tf.constant(fc_m, shape=[new_size*new_size*num_filters[0], num_labels], dtype=tf.float32))
        conv1_weight_sigma = tf.Variable(tf.constant(conv_s, shape=[num_filters[0]]))
        fc1_weight_sigma = tf.Variable(tf.constant(fc_s, shape=[num_labels]))  
        
    else:
        conv1_weight_M = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channel, num_filters[0]], stddev=init_mu))#, trainable=False, collections=collection)     
        fc1_weight_mu = tf.Variable(tf.truncated_normal([new_size*new_size*num_filters[0], num_labels], stddev=init_mu)) 
        conv1_weight_sigma = tf.Variable(tf.constant(init_sigma_std, shape=[num_filters[0]]))
        fc1_weight_sigma = tf.Variable(tf.constant(init_sigma_std, shape=[num_labels]))  

      
    fc1_bias_mu = tf.Variable(tf.constant(0.00001, shape=[num_labels]))
    fc1_bias_sigma = tf.Variable(tf.constant(0.0, shape=[num_labels])) 
    
    if Adversarial_noise:
        ADVERSARY_VARIABLES = 'adversary_variables'
        collections = [tf.GraphKeys.GLOBAL_VARIABLES, ADVERSARY_VARIABLES]
        x_noise = tf.Variable(tf.zeros([image_size, image_size, num_channel]),  name='x_noise', trainable=False, collections=collections) 
        x_noise_clip = tf.assign(x_noise, tf.clip_by_value(x_noise, -noise_limit,  noise_limit))
        x_noisy_image = x + x_noise 
        x_noisy_image = tf.clip_by_value(x_noisy_image, 0.0, 1.0) 
        print('Call the model ....')
        prediction, class_score, output_sigma, sigma_f = Model_with_uncertainty_computation(x_noisy_image,conv1_weight_M,conv1_weight_sigma,fc1_weight_mu, fc1_bias_mu,fc1_weight_sigma, fc1_bias_sigma, new_size)
        adversary_variables = tf.get_collection(ADVERSARY_VARIABLES)  
        l2_loss_noise = noise_l2_weight * tf.nn.l2_loss(x_noise)  
    else:
        print('Call the model ....')
        prediction, class_score, output_sigma, sigma_f = Model_with_uncertainty_computation(x,conv1_weight_M,conv1_weight_sigma,fc1_weight_mu,fc1_bias_mu,fc1_weight_sigma,fc1_bias_sigma, new_size)
    ######################################################     
    # KL-divergence 
    f_s = tf.log(1. + tf.exp(fc1_weight_sigma))
    c_s = tf.log(1. + tf.exp(conv1_weight_sigma))
    kl_loss_conv1 = - 0.5 * tf.reduce_mean(patch_size*patch_size + (patch_size*patch_size)*tf.log(c_s) -  tf.reduce_sum(tf.abs(conv1_weight_M)) - (patch_size*patch_size)*c_s, axis=-1) 
    kl_loss_fc1 = - 0.5 * tf.reduce_mean(new_size*new_size*num_filters[0] + (new_size*new_size*num_filters[0])*tf.log(f_s) - tf.reduce_sum(tf.abs(fc1_weight_mu)) - (new_size*new_size*num_filters[0])*f_s, axis=-1)
    ######################################################   
    output_sigma1 = tf.clip_by_value(t=output_sigma, clip_value_min=tf.constant(1e-10),
                                   clip_value_max=tf.constant(1e+10))
    y_pred_cls = tf.argmax(prediction, axis=1)
    N = 55000
    n_batches = N/ float(batch_size)
    tau = 0.002 
    print('Compute Cost Function ....')  
    log_likelihood = nll_gaussian(prediction, output_sigma1, y)   
    loss = log_likelihood +  tau *(kl_loss_conv1 + kl_loss_fc1 )
    print('Compute Optm ....')
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 1e-3
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 2000, 0.96, staircase=True)
    optm = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)     
    #optm = tf.train.AdamOptimizer(learning_rate = 1e-3).minimize(loss) 
    print('Compute Accuracy ....')
    corr = tf.equal(tf.argmax(prediction, 1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
    if Adversarial_noise:         
        loss_adversary = log_likelihood + l2_loss_noise
        adv_lr = 2.5e-5
        optimizer_adversary = tf.train.AdamOptimizer(learning_rate=adv_lr).minimize(loss_adversary, var_list=adversary_variables)
        
    print('Initialize Variables ....')   
    saver = tf.train.Saver()  # initialize saver for saving weight and bias values
    init = tf.global_variables_initializer()     
    
    if not os.path.exists(os.path.join(saved_result_path, "model.meta")):
        sess = tf.Session()# initialize tensorflow session    
        #saver.restore(sess, tf.train.latest_checkpoint('./UVI_CNN_with_sigma_{}_noise_{}/epochs_{}/'.format(init_std, noise_limit, 4)))
        sess.run(init) 
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
            for minibatch in range(int(mnist.train.num_examples/batch_size)):
                update_progress(minibatch / int(mnist.train.num_examples / batch_size))    
                batch_data, batch_labels = mnist.train.next_batch(batch_size)
                valid_batch_data, valid_batch_labels = mnist.validation.next_batch(batch_size)
                valid_batch_data = valid_batch_data.reshape(-1,28,28,1)
                batch_data = batch_data.reshape(-1,28,28,1)
                acc2 = 0  
                acc_valid2 = 0 
                err2 = 0
                err_valid2 = 0
                for i in range(batch_size):                    
                    xx_ = np.expand_dims(batch_data[i,:,:,:],axis=0) 
                    yy_ = np.expand_dims(batch_labels[i,:], axis=0)  
                    valid_xx_ = np.expand_dims(valid_batch_data[i,:,:,:],axis=0)
                    valid_yy_ = np.expand_dims(valid_batch_labels[i,:], axis=0)                  
                    sess.run([optm],feed_dict = {x: xx_, y: yy_ })                   
                    acc = sess.run(accuracy,feed_dict = {x: xx_, y: yy_}) 
                    acc_valid = sess.run(accuracy,feed_dict = {x:valid_xx_, y:valid_yy_})         
                    acc2 += acc                                      
                    acc_valid2 += acc_valid   
                    if (minibatch % 100 == 0) or (minibatch == (int(mnist.train.num_examples/(batch_size)) - 1)):                    
                        err = sess.run(loss, feed_dict = {x: xx_, y: yy_}) 
                        va_err = sess.run(loss ,feed_dict = {x:valid_xx_, y:valid_yy_})  
                        err2 += err
                        err_valid2 +=  va_err  
                if (minibatch % 50 == 0) or (minibatch == (int(mnist.train.num_examples/(batch_size)) - 1)): 
                    print('Train Acc:', acc2/batch_size,  'Valid Acc:', acc_valid2/batch_size, 'Train err:', err2/batch_size, 'valid err:', err_valid2/batch_size)                     
                  
                acc1 += acc2                           
                acc_valid1 += acc_valid2  
            train_acc[k] = acc1 / (int(mnist.train.num_examples))         
            valid_acc[k] = acc_valid1/ (int(mnist.train.num_examples))            
            print('Training Acc  ', train_acc[k])
            print('Validation Acc  ', valid_acc[k])


        if Adversarial_noise:
            print('Building Adversarial Noise .....')  
            adv_acc1 = np.zeros(int(mnist.train.num_examples/(batch_size)))
            for minibatch1 in range(int(mnist.train.num_examples/(batch_size))):
                update_progress(minibatch1/int(mnist.train.num_examples/(batch_size)))
                batch_data, batch_labels = mnist.train.next_batch(batch_size)
                batch_data = batch_data.reshape(-1,28,28,1)
                y_true_batch = np.zeros_like(batch_labels)
                y_true_batch[:, adversary_target_cls] = 1.0
                adv_acc2 = 0
                for i in range(batch_size):
                    xx_ = np.expand_dims(batch_data[i,:,:,:],axis=0)
                    yy_ = np.expand_dims(y_true_batch[i,:], axis=0)                    
                    sess.run([optimizer_adversary], feed_dict={x: xx_, y: yy_ }) 
                    sess.run(x_noise_clip) 
                    acc= sess.run(accuracy,feed_dict = {x: xx_, y: yy_ })  
                    adv_acc2 += acc
                if (minibatch1 % 50 == 0) or (minibatch1 == (int(mnist.train.num_examples/(batch_size)) - 1)):
                    print(adv_acc2/batch_size)                     
                adv_acc1[minibatch1] = adv_acc2/batch_size              
            train_acc1 = np.amax(adv_acc1)
            print('Adversary Training Acc  ', train_acc1)              

                
        stop = timeit.default_timer()
        print('Total Training Time: ', stop - start)  
        if (epochs > 1):
            fig = plt.figure(figsize=(15,7))
            plt.plot(train_acc, 'b', label='Training acc')
            plt.plot(valid_acc,'r' , label='Validation acc')
            plt.ylim(0, 1.1)
            plt.title("Unscented Variational Inference on MNIST Data")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend(loc='lower right')            
            plt.savefig(os.path.join(saved_result_path, "UVI_CNN_on_MNIST_Data.png"))
            plt.close(fig)
        print('Training is Completed')
        if (epochs==1):
            print('Training Accuracy', train_acc)
            print('Validation Accuracy', valid_acc)
        else:
            print('Training Accuracy',np.mean(train_acc))
            print('Validation Accuracy',np.mean(valid_acc))
       
        print('---------------------')
        # Saving the objects:        
        f = open(os.path.join(saved_result_path, "training_validation_acc.pkl"), 'wb') 
        pickle.dump([train_acc, valid_acc], f)                                                               
        f.close()           
        
        save_path = saver.save(sess, os.path.join(saved_result_path, "model"))    
    else:  
        sess = tf.Session()       
        saver.restore(sess, tf.train.latest_checkpoint(saved_result_path)) 
        
    if Random_noise:
        saved_result_path = './UVI_CNN_with_sigma_{}_noise_{}/epochs_{}/Gaussian_noise_{}/'.format(init_std, noise_limit, epochs, gaussain_noise_std)#
   
    nm_test_ima = 1
    batch_size = 10000
    test_acc = np.zeros(nm_test_ima)
    uncert_ = np.zeros([nm_test_ima, batch_size, num_labels, num_labels])
    cls_true = np.zeros([nm_test_ima, batch_size])
    cls_pred = np.zeros([nm_test_ima, batch_size])
    ref_test_xx1 =  np.zeros([nm_test_ima, batch_size, 28*28])
    #test_data = np.zeros([nm_test_ima, batch_size, 28, 28, 1])
    #accu = np.zeros([nm_test_ima, batch_size])
    for l in range(nm_test_ima):
        print(l+1 ,'/', nm_test_ima)
        #update_progress(l/nm_test_ima)  
        ref_test_xx1[l,:,:], test_labels = mnist.test.next_batch(batch_size)
        np.save(os.path.join(saved_result_path, "ref.npy") , ref_test_xx1)
        test_data = ref_test_xx1[l,:,:].reshape(-1,28,28,1)                   
        cls_true[l,:] =  np.argmax(test_labels, axis = 1)   
        #cls_pred = np.zeros_like(cls_true)       
        accu1 = 0       
        for i in range(batch_size):
            update_progress(i/batch_size)
            if Random_noise:
                test_data[i,:,:,:]  = random_noise((test_data[i,:,:,:] ),mode='gaussian', var= gaussain_noise_var)            
            test_xx_ = np.expand_dims(test_data[i,:,:,:],axis=0)           
            test_yy_ = np.expand_dims(test_labels[i,:], axis=0)            
            cls_pred[l,i], accu, uncert_[l, i, :, :]  = sess.run( [y_pred_cls, accuracy, output_sigma], feed_dict={x: test_xx_, y: test_yy_})           
            accu1 += accu           
        test_acc[l] = accu1/batch_size
        #test_acc[l] = np.mean(accu[l,:])
        #print('Test Acc', test_acc[l])            
     
    f2 = open(os.path.join(saved_result_path, "snr_uncert_info.pkl"), 'wb')     
    #pickle.dump([uncert_ , test_data , cls_true, cls_pred], f2)
    pickle.dump([uncert_ , cls_true, cls_pred], f2)
    f2.close()
    
    fig  = plt.figure(figsize=(15,7))
    plt.plot(test_acc, 'r', label='Test acc')
    plt.ylim(0, 1.2)
   # plt.plot(test_err, 'b', label='Test error')
    plt.title("Unscented VI CNN on MNIST Data Test Acc")
    plt.xlabel("Test Image Number")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right')        
    plt.savefig(os.path.join(saved_result_path, "UVI_CNN_on_MNIST_Data_test.png" ))  
    plt.close(fig)  
    
    print('Test Accuracy', np.mean(test_acc))
    print('Maximum Test Accuracy', np.amax(test_acc))   
    f1 = open(os.path.join(saved_result_path, "test_acc.pkl" ), 'wb')
    pickle.dump(test_acc, f1)
    f1.close()    
    ################################
    var = np.zeros([nm_test_ima,batch_size])
    for i in range(nm_test_ima):
        for j in range(batch_size):
            var[i,j] = uncert_[i,j, np.int(cls_pred[i,j]), np.int(cls_pred[i,j])]#cls_pred.shape=(30, 10000)
     
    print('Output Variance', np.mean(var))
    print('STD of Output Variance', np.std(var, dtype=np.float64))
    ################################
    if Adversarial_noise:
        ref = np.load(os.path.join(saved_result_path, "ref.npy"))# ref.shape=(30, 10000, 28, 28)
        ref = ref.reshape(nm_test_ima,batch_size,28,28) 
        
        adver_example = sess.run(x_noise)
        snr_signal = np.zeros([nm_test_ima, batch_size])
        for i in range(nm_test_ima):
            for j in range(batch_size):
                snr_signal[i,j] = 10*np.log10( np.sum(np.square(ref[i,j,:,:]))/(np.sum( np.square(ref[i,j,:,:] - (ref[i,j,:,:]+np.squeeze(adver_example) )) )))
        
        print('SNR', np.mean(snr_signal))   
    
    if Random_noise: 
        ref = np.load(os.path.join(saved_result_path, "ref.npy"))# ref.shape=(30, 10000, 28, 28)
        ref = ref.reshape(nm_test_ima,batch_size,28,28) 
           
        snr_signal = np.zeros([nm_test_ima, batch_size])
        for i in range(nm_test_ima):
            for j in range(batch_size):
                snr_signal[i,j] = 10*np.log10( np.sum(np.square(ref[i,j,:,:]))/(np.sum( np.square(ref[i,j,:,:] - (random_noise((ref[i,j,:,:] ),mode='gaussian', var= gaussain_noise_var)  )) )))
        
        print('SNR', np.mean(snr_signal))        
      
    ################################
    test_file = open('./test_images_mnist.pkl', 'rb')
    img_test, test_label =   pickle.load(test_file) 
    test_file.close()
    img_test = img_test.reshape(-1,28,28,1) 
    cls_true =  np.argmax(test_label, axis = 1)
    cls_pred = np.zeros_like(cls_true) 
    uncert = np.zeros([9, num_labels, num_labels]) 
    mean_val = np.zeros([9, num_labels])
    class_score1 = np.zeros([9, num_labels])
    
    
    sigma_f1 = np.zeros([9, num_labels, num_labels]) 
    for j in range(9):
        if Random_noise:
            img_test[j,:,:,:] = random_noise(img_test[j,:,:,:], mode='gaussian', var = gaussain_noise_var) 
        test_xx_ = np.expand_dims(img_test[j,:,:,:],axis=0)
        test_yy_ = np.expand_dims(test_label[j,:], axis=0) 
        cls_pred[j],mean_val[j,:], class_score1[j,:], uncert[j,:,:], sigma_f1[j,:,:] = sess.run([y_pred_cls, prediction, class_score, output_sigma, sigma_f], feed_dict={x: test_xx_, y: test_yy_})
    images = img_test
    
    f2 = open(os.path.join(saved_result_path, "test_uncert_info.pkl" ), 'wb')    
    pickle.dump([uncert, mean_val, cls_pred, class_score1, sigma_f1], f2)
    f2.close()      
    
    con_m, conv_s, fc_m, fc_s, b_m, b_s  = sess.run([conv1_weight_M, conv1_weight_sigma, fc1_weight_mu, fc1_weight_sigma, fc1_bias_mu, fc1_bias_sigma])
    file11 = open(os.path.join(saved_result_path, "UVI_CNN_weights_features.pkl" ), 'wb')     
    pickle.dump([ con_m, conv_s, fc_m, fc_s, b_m, b_s], file11)
    file11.close()
    
    if Adversarial_noise:
        adver_example = sess.run(x_noise)        
        file1 = open(os.path.join(saved_result_path, "UVI_CNN_x_noise.pkl" ), 'wb')        
        pickle.dump( adver_example , file1)
        file1.close() 
        
            
    if Adversarial_noise:  
        noise = sess.run(x_noise)
        noise = noise.reshape(28,28,1)
        noise1 = np.squeeze(noise)
        print("Noise:")
        print("- Min:", noise1.min())
        print("- Max:", noise1.max())        
        #plt.axis('off')         
        #plt.imsave(os.path.join(saved_result_path, "UVI_on_MNIST_noise.png"), noise1,  cmap='seismic', vmin=-1.0, vmax=1.0) 
        
        
        plot_images(images=images[0:9,:,:,:], sigma_std=init_std, epoch=epochs, cls_true=cls_true[0:9],path = saved_result_path, cls_pred=cls_pred[0:9],  noise=noise)
    else:
        plot_images(images=images[0:9,:,:,:], sigma_std=init_std, epoch=epochs, cls_true=cls_true[0:9],path = saved_result_path, cls_pred=cls_pred[0:9]) 
          
    textfile = open(os.path.join(saved_result_path, "Related_info.txt"),'w')        
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
        textfile.write('\n Random Noise std: '+ str(gaussain_noise_std))
        textfile.write("\n SNR: "+ str(np.mean(snr_signal)))
    if Adversarial_noise:
        textfile.write('\n Noise limit: '+ str(noise_limit)) 
        textfile.write("\n- Min : "+ str(noise1.min())) 
        textfile.write("\n- Max : "+ str(noise1.max()))       
        textfile.write("\n---------------------------------")
        textfile.write("\n SNR: "+ str(np.mean(snr_signal)))            
    textfile.write("\n---------------------------------")
    textfile.write("\n Output Variance: "+ str(np.mean(var))) 
    textfile.write("\n STD of the Output Variance: "+ str(np.std(var, dtype=np.float64)))
    textfile.write('\n Initial std of sigma of the weights : ' +str(init_sigma_std))
    textfile.write('\n Initial log(1+exp(sigma)) : ' +str(init_std)) 
    textfile.write('\n Rate of Convergence : ' + str(epochs)+ ' epochs')    
    textfile.close()
    sess.close()          
if __name__ == '__main__':
    main_function()    
