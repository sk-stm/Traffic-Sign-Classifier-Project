#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas and numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is between 27x27 and 230 x 201
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is an example showing how the data looks like befor:

![alt text][UnprocessedImage.png]

and after the preprocessing:

![alt text][PreprocessedImage.png]

Also here is a histogram showing how the classes are distributed in the data set.
![alt text][Histogram.png]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to adjust the brightness level of the images because the seem to differ a lot in the data set.

Here is an example of a traffic sign image before and after brightness ajustment.

![alt text][PreprocessedImage.png]

As a last step, I normalized the image data because the net should learn on normalized data to avoid lerning a bias and also to set the 
activation function into a good working point from begin with.

I decided not to generate additional data because the result after the first adjustments were more then the required result.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	       		| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x16 	|
| RELU
| Input         		| 30x30x16 feature | 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 		|
| Input         		| 14x14x16 feature | 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 12x12x32 	|
| RELU
| Input         		| 12x12x32 feature | 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU	
| Max pooling	      	| 2x2 stride,  outputs 5x5x32			|
| Fully connected	| input = 800 output = 400			|
| RELU	
| DROPOUT		| keep_rate = 0.5
| Fully connected	| input = 400 output = 150			|
| RELU	
| DROPOUT		| keep_rate = 0.5
| Fully connected	| input 150 output = 43				|
| Softmax
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an adam Optimizer because it's a good default choice to go to since it also includes momentum and other learning
step adjustments compared to SGD.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.987
* validation set accuracy of 0.967
* test set accuracy of 0.944

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

The first architecture I tried was the LeNet that was used for hand written digit classification.

* What were some problems with the initial architecture?

It was to small to completely adjust to the data. So a larger model needed to be used.

* How was the architecture adjusted and why was it adjusted? Which parameters were tuned? How were they adjusted and why? What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

At first the architecture worked pretty fine. I altered the net massively using the ResNet achritecture (https://arxiv.org/abs/1512.03385), added residual connections and also skipconnections (https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) and I was planing to exeriment with batchnormalization and batch renormalization to achieve a high accuracy on the validation set.

Unfortunately I had to train the net on my personal computer. As you might imagine this tool unsustainable much time and I wasn't able to train the net in reasonable time at all. So I decided to shrink the net down again and don't take a sledgehammer to crack a nut. I returned to the LeNet architecture.

Since this architexture was unterfitting, I had to add more parameter to the net. First I replaced the 5x5 convolution by 2 3x3 convolutions, which overspan the same spatial size during the convolution but have less parameters.(5x5 = 25, 2x3x3 = 18). Thus the net trains faster and has the same translational invariance. That allowed me to enlage the net and still train it in reasonable time.

As a second step I made the net deeper. Using 16 channels in the first 2 convolutions and 32 channels in the latter two. That allowes the net to track more semantic information from the fictures. 

The next step was to gradually expand the net. I started with adding another fully conencted layer. The reason was that another fully conencted layer helps the net to turn the information from the conolutions in reasonable predictions. In hinsight it may have also worked to just enlarge the existing fc layer and don't add another one. 

When I trained the net, I got a very high training accuracy, but still a small validation accuracy. Clearly the net was overfitting. So I added 2 dropout layer after the first and second fully connected layer to reduce the overfitting. The dropout could also have been added at other places, in the end this choice would have not made much of a difference.

Running this net gives a validation accuracy of over 0.96 and was trainable for my computer on CPU in about an hour, which is a good performace / speed tradeoff for me. 

If a well known architecture was chosen:
* What architecture was chosen?

I chose ResNet then due to time issues I chose LeNet.

* Why did you believe it would be relevant to the traffic sign application?

In general the residual connections allow a better gradient flow from earlier layers to later layer and vice versa so ResNet is preferred over LeNet. Other options may have been possible but ResNet was my first try and since I couldn't train it in reasonable time I swithed back to LeNet, which is a goot performance / speed trade off for me.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
Since the accuracy of training, validation and test set are fairly close rather high values, the model is working well in classification and generalisation manner.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Picture 1][AdditionalData/pic01.png] ![Picture 2][AdditionalData/pic02.png] ![Picture 3][AdditionalData/pic03.png] 
![Picture 4][AdditionalData/pic04.png] ![Picture 5][AdditionalData/pic05.png]

The first image might be difficult to classify because the sign is croped, and the signs in the data set are fully visible all the time.
The second image might be difficult to classify because there are not many examples in the data set for that class and it can be easyly confused with the 80km/h class due to similar shape and numbers. Also this image is distorded which might not fit any other image of this class in the data set.
The third image might be difficult to classify because there are not many examples in the data set for that class and it can be easyly confused with the 20km/h class due to similar shape and numbers.
The fourth image might be difficult to classify because it might appear in a differen angle then the picture in the data set and I don't use any training set enlargin techniques.
The fifth image might be difficult to classify because it's a rather old sign for a pedestrian crossing and might not be in this form in the data set at all.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image		        |     Prediction	       			| 
|:---------------------:|:---------------------------------------------:| 
| Keep right   			| No Entry   				| 
| Speed limit 20km/h    	| Speed limit 20km/h 			|
| End of Speed limit 80km/h	| End of speed limit 80km/h		|
| Turn left ahead      		| End of all speed and passing limits	|
| Pedestrians			| Speed limit 50km/h      		|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This low accuracy is due to the difficult nature of the pictures.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is very sure that this is a No Entry sign (probability of 0.98), but the image containes a keep right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| .98        		| No Entry					| 
| .02   		| Wild Animals 					|
| .00			| Biycles crossing				|
| .00	      		| Double curve					|
| .00	 		| Road work      				|

For the second image, the model is very sure that this is a 20km/h speed limit sign (probability of 0.96), and the image does contain a 20km/h speed limit sign. The top five soft max probabilities were

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| .96        		| No Entry					| 
| .01   		| End of speed limit (80km/h)			|
| .00			| End of all speed and passing limits		|
| .00	      		| Speed limit (30km/h)				|
| .00	 		| No entry      				|


For the thrid image, the model is very sure that this is a End of speed limit (80km/h) sign (probability of 1), and the image does contain a End of speed limit (80km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| 1        		| End of speed limit (80km/h)			| 
| .00   		| End of all speed and passing limits		|
| .00			| Speed limit (80km/h)				|
| .00	      		| Speed limit (30km/h)				|
| .00	 		| End of no passing by vehicles over 3.5 metric tons|

For the fourth image, the model is not sure that this is a End of all speed and passing limits sign (probability of 0.21), and the image does contain a Turn left ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| .21        		| End of all speed and passing limits		| 
| .20   		| Right-of-way at the next intersection		|
| .10			| End of no passing				|
| .08	      		| Priority road prob				|
| .06	 		| Turn left ahead				|

For the fifth image, the model is farily sure that this is a Speed limit (50km/h) sign (probability of 0.21), and the image does contain an old pedestrians sign. The top five soft max probabilities were

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| .68        		| Speed limit (50km/h)				| 
| .20   		| Speed limit (30km/h)				|
| .02			| Wild animals crossing				|
| .02	      		| Roundabout mandatory				|
| .01	 		| Speed limit (20km/h)				|



