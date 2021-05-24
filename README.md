# Variational-Density-Propagation-CNN
Model conﬁdence or uncertainty is critical in autonomous systems as they directly tie to the safety and trustworthiness of the system. 
The quantiﬁcation of uncertainty in the output decisions of deep neural networks (DNNs) is a challenging problem. 
The Bayesian framework enables the estimation of the predictive uncertainty by introducing probability distributions over the (unknown) network weights; 
however, the propagation of these high-dimensional distributions through multiple layers and non-linear transformations is mathematically intractable. 
In this work, we propose an extended variational inference (eVI) framework for convolutional neural network (CNN) based on tensor Normal distributions (TNDs) deﬁned over convolutional kernels. 
Our proposed eVI framework propagates the ﬁrst two moments (mean and covariance) of these TNDs through all layers of the CNN. 
We employ ﬁrst-order Taylor series linearization to approximate the mean and covariances passing through the non-linear activations. 
The uncertainty in the output decision is given by the propagated covariance of the predictive distribution. 
Furthermore, we show, through extensive simulations on the MNIST and datasets, that the CNN becomes more robust to Gaussian noise and adversarial attacks.

The code is scaled to TF 2 version in 
https://github.com/dimahdera/Variational-Density-Propagation-CNN-MNIST-Tensorflow-2.x.git
https://github.com/dimahdera/PremiUm-CNN-CIFAR10-Tensorflow-2.x.git

# Citation

Dimah Dera, Nidhal C. Bouaynaya, Ghulam Rasool, Roman Shterenberg and Hassan M Fathallah Shaykh, “PremiUm-CNN: Propagating Uncertainty towards Robust Convolutional Neural Networks,” IEEE Transactions on Signal Processing, 2021 (under review).
