
# A prototype for cnn_forward_backward for classification and object localization in C++

Notes:
1) the methodology is referred to Andrew NG., Deep Learning Specilization, Coursera, which provided the backbone and framework of incomplete Python code.
2) the major contribution of this prototype convers:
  a) it complete the full procedure of CNN for classification and object detection, including the additonal layers of flatten-> fully-connected layers -> classification -> location detection, and data in and results out with the main function; 
  b) it wrotes in C++ and with parallism option for CPUs. It is designed in object oriented with three classes and one main function: 
     (1) class: math operation; 
     (2) class: forward computation;
     (3) Class: backward computation;
     (4) main: read data in, train the model and output prediction results.

![image](https://user-images.githubusercontent.com/78186650/212702824-d621ca39-5b29-4632-8095-28e5c6fbf079.png)

![image](https://user-images.githubusercontent.com/78186650/212701507-6e3d0643-3343-4e0b-b2a6-7903875ad8d2.png)
![image](https://user-images.githubusercontent.com/78186650/212701742-e1c33b72-215c-4dc5-8b3e-05a809efb1b1.png)

![image](https://user-images.githubusercontent.com/78186650/212701814-abc098c9-66fb-4d1c-b3b8-616ecbb6caff.png)



