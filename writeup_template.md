# Traffic Sign Recognition using CNN
---
### Data Set Summary & Exploration

![alt text](https://github.com/srikanthmalla/Traffic_Sign_Classifier/blob/master/images.png)

#### 1. Provide a basic summary of the data set.
I numpy library to calculate summary statistics of the traffic
signs data set and plotted using matplotlib to see the data distribution (which is not uniformly distributed for all classes):

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32, 32, 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Histogram of data distribution over different classes:

![alt text](https://github.com/srikanthmalla/Traffic_Sign_Classifier/blob/master/dataset_histogram.png)

Rotated with scipy library ndimage to create more data if the data set is less mean of the data which is 809,
then the data distribution becomes:

New Histogram of data distribution over different classes

![alt text](https://github.com/srikanthmalla/Traffic_Sign_Classifier/blob/master/histogram_augmented_dataset.png)

### Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the data for the signs is not mainly color dependent but dependent on the shapes of the signs.

Here is an example of a traffic sign image before and after grayscaling.

![alt text](https://github.com/srikanthmalla/Traffic_Sign_Classifier/blob/master/gray.png)

As a last step, I normalized the image data because normalized inputs are more reasonable than the unnormalized for the model to be more efficient.

I decided to generate additional data because the data is not uniformly distributed for all the classes 

To add more data to the the data set, I used the following techniques because without losing the sign just rotating gives more data. Cropping also gives more data if we know how much cropping will not leads to loss of the data (signs in the image in our case)

Here is an example of an original image and an augmented image:

![alt text](https://github.com/srikanthmalla/Traffic_Sign_Classifier/blob/master/rotated.png)

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|  outputs |
|:---------------------:|:-----------------------|:---------------------:| 
| Input         	        	| grayscale image   					|        32x32x1      |    	 
| Convolution1     5x5x1x6  | 1x1x1 stride, VALID padding |outputs 28x28x6 	| 
| RELU1				              |			
| Max pooling	      	2x2x1 | 2x2x1 stride, VALID padding |outputs 14x14x6 			|
| Convolution2 	  5x5x6x16  | 1x1x1 stride, VALID padding |outputs 10x10x16		|
| RELU2			               	| 
| Max pooling	      	2x2x1 | 2x2x1 stride, VALID padding |outputs 5x5x16 |
| flatten | | 400| 
| Fully connected 	400x120 |                             |outputs 120|
| RELU3				              |			
| Dropout1                  | 0.7 | 
| Fully connected 	 120x84 |                             |outputs 84|
| RELU4			               	|
| Dropout2                  | 0.7| 
| Fully connected			84x43 |                             |outputs 43|
| Softmax               		|   |  outputs 43|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Hyper parameters:
Epochs=50
Batch size=150
Optimizer= Adam
learning rate=0.001
when I changed learning rate from 0.005 to 0.001 for 100 epochs I got validation accuracy from 94% to 95%. Also, my test accuracy increased from 91% to 94%. Also in the visualization of the layers, I found few feature blocks remained dark all the time when the learning rate is 0.005

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9
* validation set accuracy of 94.9
* test set accuracy of 93.58

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? 
  LeNet architechture was chosen at first without any modifications at first, as it was described in the class and very simple architecture.

* What were some problems with the initial architecture?
  It overfits training error 100% and testing goes to around 80% for 100 epochs with learning rate 0.001
  
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  Dropout layers were added with keep probability of 0.7, then I get validation accuracy of 94% with just 50 epochs
  
* Which parameters were tuned? How were they adjusted and why?
  when I changed learning rate from 0.005 to 0.001 for 100 epochs I got validation accuracy from 94% to 95%. Also, my test accuracy increased from 91% to 94%.
 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
 As observed dropout layers helps in removing overfitting.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text](https://github.com/srikanthmalla/Traffic_Sign_Classifier/blob/master/add_pics/13.jpg)
![alt text](https://github.com/srikanthmalla/Traffic_Sign_Classifier/blob/master/add_pics/14.jpg)
![alt text](https://github.com/srikanthmalla/Traffic_Sign_Classifier/blob/master/add_pics/25.jpg)
![alt text](https://github.com/srikanthmalla/Traffic_Sign_Classifier/blob/master/add_pics/3.jpg)
![alt text](https://github.com/srikanthmalla/Traffic_Sign_Classifier/blob/master/add_pics/34.jpg)

First two images are distinct from other signs because of the shape of plate. So they are easy. But other signs are on round plates, which are difficult.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield     		| Yield 									| 
| General caution     			| Road work									|
| Keep right					| Turn left ahead											|
| No passing for vechiles over 3.5 metric tons     		| Speed limit (60km/h)				 				|
| Stop			| Stop      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|



### (Optional) Visualizing the Neural Network 
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text](https://github.com/srikanthmalla/Traffic_Sign_Classifier/blob/master/visualize.png)

In the visualization of the convolution layers, I found few feature blocks remained dark all the time when the learning rate is 0.005. When I decreased learning rate to 0.001, the performance increased and the feature layers are trained and visible.


