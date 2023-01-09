
# A prototype for cnn_forward_backward for classification and object localization in C++
Notes:
1) the methodology is referred to Andrew NG., Deep Learning Specilization, Coursera, which provided the backbone and framework of incomplete Python code.
2) the major contribution of this prototype convers:
  a) it complete the full procedure of CNN for classification and object detection, including the additonal layers of flatten-> fully-connected layers -> classification -> location detection, and data in and results out with the main function; 
  b) it wrotes in C++ and organized in object oriented and with parallism option for CPUs 

Forward computation flowchart
![image](https://user-images.githubusercontent.com/78186650/211235835-e9a197f7-f367-45c7-b364-ce29ae79dbe2.png)
