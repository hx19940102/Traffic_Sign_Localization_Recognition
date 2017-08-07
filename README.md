# Traffic_Sign_Localization_Recognition

This project mainly contains two different parts:
1. Locate/Detect the traffic signs in the image.
2. Recognize/Classify the traffic signs detected by part 1.

For part one, I mainly use lbp cascade classifier for detect traffic signs in an image. It is similar to face detection, you only need to train the cascade classifier for detecting traffic signs. The information about cascade classifier training and detection could be found from opencv document:http://docs.opencv.org/3.0-beta/doc/user_guide/ug_traincascade.html

The cascade classifier could return multiple detection results from one image.

For part two, I take use of the Lenet model and change the structure a little bit, input image size now should be 64*64 instead of 32*32 and I add one more CONV layer with Max_Pool layer. I collect data for 5 different types of traffic signs:
![alt text](http://url/to/img.png)
![alt text](http://url/to/img.png)
![alt text](http://url/to/img.png)
![alt text](http://url/to/img.png)
![alt text](http://url/to/img.png)

I also augment the data with rotations and gaussian blur.
Training of the network is fast and easy, the accuracy on the training dataset could reach more than 98%.

I combine both part 1 and part 2 together and test with the camera input. The result seems great:
![alt text](http://url/to/img.png)
![alt text](http://url/to/img.png)
![alt text](http://url/to/img.png)
