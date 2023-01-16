
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

Forward computation flowchart
![image](https://user-images.githubusercontent.com/78186650/211235835-e9a197f7-f367-45c7-b364-ce29ae79dbe2.png)

![image](https://user-images.githubusercontent.com/78186650/212700656-2442f76c-f826-4fc6-acfe-349b1ac6d15c.png)
![image](https://user-images.githubusercontent.com/78186650/212700984-d8a66097-252f-4d67-9e73-89aab12b9ccf.png)
![image](https://user-images.githubusercontent.com/78186650/212701050-6f70e930-6125-4751-9f50-7efeaee4dd64.png)
