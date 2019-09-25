# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./show_images/0-Visual_Classed.png "Visualization"
[image2]: ./show_images/1.0-Visual_Random_20.png "Random"
[image3]: ./show_images/1.1-Extra-Data.png "Extra"
[image4]: ./show_images/1.2-Extra-Data.png "Extra"
[image5]: ./show_images/1.3.0-All-Type.png "All"
[image6]: ./show_images/1.3.1-All-Gray.png "Gray"

[image4.0]: ./show_images/4.0-Train-Valid-Acc.png "Acc"
[image4.1]: ./show_images/4.1-Train-Valid-Show.png "Acc"
[image4.2]: ./show_images/4.2-Test-Acc.png "Acc"
[image4.3]: ./show_images/4.3-Orig-Valid.png "Orig Valid"
[image4.4]: ./show_images/4.4-Gray-Only.png "Gray Only"


[image5.1]: ./show_images/5.1-Test-New.png "Test"
[image5.2]: ./show_images/5.2-Test-New.png "Test"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/kindht/CarND-Project3-Traffic-Sign-Recognition/blob/master/Traffic_Sign_Classifier-Final.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to read CSV file to get info about ClassID and Sign Names,  and used numpy to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many samples(images) for each class of signs in training dataset(blue bars) and validation dataset(orange bars).(Note: in this writeup, , 'class', 'label', 'type', they will mean the same thing, i.e. type of the sign)

The following picture shows that the training data set is very unbalanced, many classes of signs have very small number of samples, for ex. only about 200 samples , whereas some other  classes of signs have nearly 2000 samples.

![alt text][image1]

I also checked images in traing data set, notcied that many are very dim and some are even totally dark, which we can hardly recognize.

Here are some examples:

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

### 1) Generate Extra Data
I decided to **generate additional data** because the training data set is very unbalanced, without extra data, the model can not perform well when trying to recognize new test images downloaded from web.

To add more data to the the data set, I used the following techniques:  

First , define a target **num_samples**, then go through each class of sign images, if the number of samples for that type is less than the target number,  I calculate a number for repeats, then generate extra samples by repeating the original ones.  

num_samples should not be too high to avoid the model running too slowly.

I once tried to randomly adjust the contrast and brightness with the extra data generated, but the performance of the model didn't get improved,  even may go worse, so I just leave the data be original ones.

Up to 21958 extra images are generated ,  so that the training set totally has 56757 images (remember that original data set has 34799 images)

![alt text][image3]

The difference between the original data set and the augmented data set is the following 

Now the training data is **more balanced**, although some classes still have more images than the others, overall looks better, which can help improve the performace of the model

![alt text][image4]

### 2) Grayscaling 
I decided to convert the images to grayscale because most of original images can hardly be recognized. The grayscaling can **dramatically improve** the appearances of the images.

Here are examples of traffic sign images before and after grayscaling:

**Before grayscaling**, the following shows all classes of signs, noticing those with labels of '3' '8' '14' '38' etc. are very dark, we can hardly know what they are

![alt text][image5]

**After grayscaling**,  we can almost recognize all, especially those are very dark before , ex. those labeled as  '3' '8' '14' '38' etc.
![alt text][image6]

### 3) Normalization
As a last step, I normalized the image data because normalization is crucial for neural network training. I first tried (pixel - 128)/128, and then (pixel/255 - 0.5), finally I used per image standardization.

Although the per image standardization technique takes more time than simpler scaling menthods, 5~10 seconds of calculation for the whole data set is acceptable , considering that can clearly help to improve the recognition performance. 

The image data is zero-centered by subtracting the mean , then divided by its standard deviation to get equal variance. In this way, the neural network may converge faster with gradient decent


```python
def preprocess_image(image_data):
    """
    Normalize the image data with mean & std 
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """  
    scale_mean = np.mean(image_data)
    scale_std = np.std(image_data)
    return (image_data - scale_mean) / scale_std
```


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution1 5x5      | 1x1 stride, VALID padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling 2x2	    | 2x2 stride,  outputs 14x14x12 				|
| Convolution2 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling 2x2	    | 2x2 stride,  outputs 5x5x16				    |
| Fully connected		| Input units 400(5x5x16), Output units 120     |
| Fully connected		| Input units 120, Output units 84              |
| Fully connected		| Input units 84, Output units 43 (classes)     |

The output of the model is logits coming out of last fully connected layer


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following hyperparameters:  
    - learning rate: 0.001  
    - EPOCHS: 15  
    - BATCH_SIZE: 128

I used **AdamOptimizer** considering it can automatically adjusting learning rate while training.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were follwing with 15 epochs
* training set accuracy:   best is 0.999 (stable around 0.99)
* validation set accuracy: best is 0.951 (stable around 0.93~0.94) 
* test set accuracy: 0.929

![alt text][image4.0]   
![alt text][image4.1]   
![alt text][image4.2]   


If a well known architecture was chosen:  
* **What architecture was chosen?**  
I chose the classical LeNet architecture as a starting point

* **Why did you believe it would be relevant to the traffic sign application?**  
LeNet is known as it is the first Convolutional Neural Network designed to recognize handwritten numbers. Convolutional Nerural Networks are designed to recognize visual patterns directly from pixel images with minimal preprocessing , they can recognize patterns with extreme variablity and with robustness to distortions and simple geometric transformations"(refer to yann.lecun.com).  Since LeNet5 worked well with handwritten numbers recognition,  it is simple but powerful , thus a good choice for our traffic sign recogniztion. 

* **How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?**  
By implementating the LeNet architecture, even without any preprocessing except the simple normalization  , the model could already achieve validation accuracy up to about 0.89

#### 1) Basic LeNet with Only simple normalization
 ![alt text][image4.3] 

#### 2) With grayscaling images, the validation accuracy went up quickly
 ![alt text][image4.4] 

#### 3) **With extra data generated and a small adjustments to network archiecture, the final results got better**
Just with extra data generated, the validation accuracy even went down a bit,  I think that is because the network architecture is too simple with respect to the larger data set.   

So I decided to adjust the network architecture by increasing the number of filters for the 2 convolutional layers, ex. LeNet originally has output 28x28x6 for convolution layer 1, I changed it to 28x28x12.   similarly to 2nd convolutional layer

In that way, I made the network a bit deeper, but not too much, I made these changes very cautiouly because the more complicated the network , the eaier it may result in overfitting,I tried to keep the model as simple as possilbe, so that it is not necessaily to do dropout,since the data set is large enough, it can hardly overfit.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:


![alt text][image5.1] 


- The first 3 images have trees and leaves in the background
- The 2nd image(30km/h) has a watermark over the sign.  
- The Roadwork and Children crossing signs, both have relatively complicated patterns in the images, i.e. human don't show as regular geometric shapes or numbers. 
- The 5th image (General caution) , there is another sign plate below the caution sign


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Speed limit (30km/h)  | Speed limit (30km/h)  					    |
| Road Work				| Road Work										|
| Children crossing	    | Children crossing					 			|
| General caution		| General caution      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This test result was even out of my expectation!  This compares favorably to the accuracy on the test set of 92.9%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 34th cell of the Ipython notebook.

For all the 5 test images, the model are quite sure for its predictions (probability of 100% for 4 sign images, and 99.99% for stop sign image). The top five soft max probabilities for predictions were as following


![alt text][image5.2] 


Probably because the images I chose are all with high resolutions, the complicated patterns or noise background didn't bother the network at all. This further proved how powerful a CNN is even a simple one like LeNet

Also recall that before I generated extra data , the accuracy of preditions for other 5 new images was just 60% or 80%, this proved that more data for training is also very critical.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)


