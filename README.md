## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
Download the dataset to run the network from https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip

and rename the extracted zip file to dataset in the cloned repo

Basic LeNet, traffic sign classifier is build:
Gives validation accuracy around 95.4% and test accuracy 94% 
with learning rate = 0.001 for 100 epochs

## Architecture

My model consisted of the following layers based on LeNet architecture.
<pre>
| Input         	        	| grayscale image   					|        32x32x1          	 
| Convolution1     5x5x1x6  | 1x1x1 stride, VALID padding |outputs 28x28x6 	 
| RELU1				              |			
| Average pooling	      	2x2x1 | 2x2x1 stride, VALID padding |outputs 14x14x6 			
| Convolution2 	  5x5x6x16  | 1x1x1 stride, VALID padding |outputs 10x10x16			
| RELU2			               	|
| Average pooling	      	2x2x1 | 2x2x1 stride, VALID padding |outputs 5x5x16
| flatten 400
|
| Fully connected 	400x120 |                             |outputs 120
| RELU3				              |			
| Dropout1                  | 0.7
| Fully connected 	 120x84 |                             |outputs 84
| RELU4			               	|			
| Dropout2                  | 0.7
| Fully connected			84x43 |                             |outputs 43
| Softmax               		|
</pre>

## Visualizing the Neural Network

Vizualizing the parameters of the first convolution layer for yield sign looks like this
![alt text](https://github.com/srikanthmalla/Traffic_Sign_Classifier/blob/master/visualize.png)
