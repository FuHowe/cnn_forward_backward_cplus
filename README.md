
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

Forward computation steps of CNN
Inputs X
Hidden layers of CNN blocks:
1st CNN block: -> $$Z_1=W_1 X$ -> $A_1=σ(Z_1)$ -> Pooling (max or average)
2nd CNN block: -> $Z_2=W_2 A_1$->  $A_2=σ(Z_2)$ -> Pooling (max or average)
ith CNN block: -> $Z_i=W_i A_(i-1)$->  $A_i=σ(Z_i)$ -> Pooling (max or average)
Nth/ last CNN block: -> $Z_N=W_N A_(N-1)$-> $A_N=σ(Z_N)$ -> Pooling (max or average)
Final layers:
Flatten the last activation (tensor to 1-D vector)
Fully connect layer: $Z_(N+1)=A_N W$
Final activation for classification, e.g. softmax for multi-class one label problem:
$a_i=e^(z_i )/(∑_(k=1)^(n_c)▒e^(z_k))$											(1)
Where $n_c$ is number of classes.
For numerical stability purpose, equation (1) could be modified as follows:
$a_i=e^(z_i-max⁡(z))/(∑_(k=1)^(n_c)▒e^(z_k-max⁡(z)))$										(2)
The loss function after the final activation using cross entropy can be express as below:
$L=-∑_(k=1)^(n_c)▒(y_k loga_k)$										(3)
Where a_k is the activation of the kth class, and y_k is the true value of that class (0 or 1).
For numerical stability of logarithmic function, equation (3) can be modified as below:
$L=-∑_(k=1)^(n_c)▒(y_k log((a)_k+ϵ))$									(4)
Where ϵ is a very small value such as $ϵ=1e^(-8)$.

