# Advanced-Programming-FINAL 
Final project NN to detect a facial expression ZAYNULLINA AZIZA
https://youtu.be/rS7kx3_dx3s
GITHUB LINK:
https://github.com/pypxt/Advanced-Programming-FINAL 

Introduction.
Problem 
Facial emotion recognition is one of the most popular implementations of neural network. It classifies emotions of person based on his facial expression. In my opinion, this model has several interesting practical applications, such as:
-	Interaction of AI and Human. Computer can detect the emotion of human, i.e. understand the psychology of him, and better respond and suit needs of human.
-	Cyber Security. Basically, this model can suit face recognition too. It is pretty popular to have skud with face recognition nowadays. For example: in government buildings, e-gov, universities and etc. 
-	It also can detect people that behave suspicious or nervous. So, it, potentially, can help to recognize the criminals.
Literature review with links (another solutions)

Convolutional neural networks (CNNs) are commonly used for face emotion recognition tasks. A typical CNN architecture for this task consists of several convolutional layers, followed by pooling layers and fully connected layers. The input to the model is an image of a face, and the output is a probability distribution over the possible emotions, such as happy, sad, angry, or neutral.

1. Real-time Facial Emotion Detection using deep learning 
https://github.com/atulapra/Emotion-detection
The model is trained on the FER-2013. This implementation by default detects emotions on all faces in the webcam feed. With a simple 4-layer CNN, the test accuracy reached 63.2% in 50 epochs.

2. Facial Emotion Recognition on FER2013 Dataset Using a Convolutional Neural Network
https://github.com/gitshanks/fer2013
They used two methods: Using the built model and Build from scratch. This Model - 66.369% accuracy.

Current work 
Train a Neural Network to detect the emotion by images with different facial expressions. 
2. Data and Methods
■ Information about the data (probably analysis of the data with some visualisations)
I used the data from Kaggle – FER-2013. 
https://www.kaggle.com/datasets/msambare/fer2013 
The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image.
There are seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The training set consists of 28,709 examples and the public test set consists of 3,589 examples.
I import it to Google Colab via Google drive. 
 

There was need to unzip the file with dataset because it was configures in .gz. extension. 
 
■ Description of the ML models you used with some theory
The code was written from scratch. The model used – CNN (Convolutional Neural Network) using TensorFlow and Keras. TensorFlow model was created by ‘basemodel’ and multiple layers.
 
Then compile the model. I had to provide optimizer with learning rate of 0.0001. 
There were an error, whenever I tried to train up the model, because the shapes did not match. The reason is    loss = 'sparse_categorical_crossentropy'. 
![photo_2023-02-20_20-51-10](https://user-images.githubusercontent.com/97113511/220138778-0fd4e6cd-7e65-4fdf-84c8-b5abbaebbfe4.jpg)



■ Results with tables, pictures and interesting numbers
I checked on 50 epochs and got pretty good accuracy, even though validation accuracy is still pretty low:
 
And the result is pretty accurate model that predicts the emotions based on images:  
  



![photo_2023-02-20_20-51-09](https://user-images.githubusercontent.com/97113511/220138633-4157741a-1b49-46a7-84c1-e9f15c083de3.jpg)







Resources:
https://youtu.be/UHdrxHPRBng
https://ieeexplore.ieee.org/document/9641053
https://www.sciencedirect.com/science/article/pii/S1877050920318019


