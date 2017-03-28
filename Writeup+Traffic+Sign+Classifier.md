
**Write Up**

##Data Set Summary & Exploration##

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

X_train shape: (83245, 32, 32, 1)
y_train shape: (83245,)
X_test shape: (12630, 32, 32, 3)
y_test shape: (12630,)

**Question 1
Describe the techniques used to preprocess the data.**

For Data Reduction, I converted the images from RGB to grayscale - that way we don't have to preprocess the image with all three color channels seperately. 

Also I one-hot encoded the labels to make computing the cross entropy for the loss function possible. 

<img src="Real-world-images/Unknown.png">








**Question 2
Describe how you set up the training, validation and testing data for your model.**

I built the validation set choosing random data, then applied the following methods: skew, perspective transforms and rotation.

**Question 3
What does your final architecture look like?**

mu = 0 sigma = 0.1 Number of convolutional layers - 3 Number of fully connected layers - 2 

**Layer 1:**
Convolution of 5x5 kernel, 1 stride, 24 feature maps.  Activation: ReLU Pooling 1: 2x2 kernel and 2 stride.

**Layer 2:**
Convolution of 3x3 kernel, 1 stride and 32 feature maps Activation: ReLU Pooling 2: 2X2 kernel and 2 stride

**Layer 3:** 
Fully connected layer with 512 units. ReLU activation

**Layer 4:** Fully connnected with 256 units Activation: ReLU  
Output Layer: Fully connected with 43 units for logits output. Activation: Softmax

**Question 4. How did you train your model?**

Adam optimizer with 0.001 Batch size. 128 Epochs and 15 Hyperparameters. Mean = 0; Standard Deviation of 0.1

**Question 5 What approach did you take in coming with a solution to this problem?**

I used the Adam Optimizer because it performed well being adaptive.  With lower epochs, was able to obtain high accuracy. Used dropout of 0.5 for training and 1.0 for validation to prevent overfitting.

**Question 7**
Is your model able to perform equally well on captured pictures when compared to testing on the dataset? The simplest way to do this check the accuracy of the predictions. For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate.

In the test set, the accuracy was 95%, but in the captured images, the accuracy was only 20% (1 out of 5)



