
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
1st CNN block: -> Z_1=W_1 X -> A_1=σ(Z_1) -> Pooling (max or average)
2nd CNN block: -> Z_2=W_2 A_1->  A_2=σ(Z_2) -> Pooling (max or average)
ith CNN block: -> Z_i=W_i A_(i-1)->  A_i=σ(Z_i) -> Pooling (max or average)
Nth/ last CNN block: -> Z_N=W_N A_(N-1)-> A_N=σ(Z_N) -> Pooling (max or average)
Final layers:
Flatten the last activation (tensor to 1-D vector)
Fully connect layer: Z_(N+1)=A_N W
Final activation for classification, e.g. softmax for multi-class one label problem:
a_i=e^(z_i )/(∑_(k=1)^(n_c)▒e^(z_k ) )											(1)
Where n_c is number of classes.
For numerical stability purpose, equation (1) could be modified as follows:
a_i=e^(z_i-max⁡(z))/(∑_(k=1)^(n_c)▒e^(z_k-max⁡(z)) )										(2)
The loss function after the final activation using cross entropy can be express as below:
L=-∑_(k=1)^(n_c)▒〖y_k loga_k 〗										(3)
Where a_k is the activation of the kth class, and y_k is the true value of that class (0 or 1).
For numerical stability of logarithmic function, equation (3) can be modified as below:
L=-∑_(k=1)^(n_c)▒〖y_k log〖(a〗_k+ϵ)〗									(4)
Where ϵ is a very small value such as ϵ=1e^(-8).
Backward computation steps of CNN
Now, let’s start the backward computation step by step.
For each layer, we need to calculate the derivative of the loss L with respect to (wrt) the model parameters of weights W and bias b so that we could updater the model parameters using stochastic gradient decent (SGD) or other optimization methods. For example, when we use SGD, the model parameters are updated as follows at each iteration step:
W_j=W_(j-1)-lr (dL_j)/(dW_(j-1) )										(5)
b_j=b_(j-1)-lr (dL_j)/(db_(j-1) )										(6)
Where j is iteration step, and lr is learning rate.
We will start from the final loss function, and then backward to compute the derivatives using chain rule. We first compute the derivative of L wrt Z at the last layer (Z=W.A+b), as follows:
dL/(dz_i )=(-d(∑_(k=1)^(n_c)▒〖y_k loga_k 〗))/(dz_i )=-∑_(k=1)^(n_c)▒(y_k d(loga_k ))/(d(a_k))  d(a_k )/(d(z_i))=-∑_(k=1)^(n_c)▒y_k/a_k   d(a_k )/(d(z_i))				(7)
Calculate the derivative of L wrt weight at the Nth(last) layer, W^N				(8)
dL/(dW^N )=dL/(dZ^N )  (dZ^N)/(dW^N )=(A^N-Y)  d(A^(N-1) W^N+b^N )/(dW^N )=(A^N-Y) A^(N-1)					(9)
Calculate the derivative of L wrt weight at the (N-1)th layer:
dL/(dW^(N-1) )=dL/(dZ^N )  (dZ^N)/(dA^(N-1) )  (dA^(N-1))/(dZ^(N-1) )  (dZ^(N-1))/(dW^(N-1) )									(10)
Where,
(dZ^N)/(dA^(N-1) )=W^N											(11)
(dZ^(N-1))/(dW^(N-1) )=A^(N-1) 											(12)
Let’s define derivative of activation function wrt Z as:
(dA^(N-1))/(dZ^(N-1) )=σ'(Z^(N-1))										(13)
Where σ is the activation function. For sigmoidal activation:
σ^' (Z^(N-1) )=d(1/(1+e^(-Z^(N-1) ) ))/(dZ^(N-1) )=(-e^(-Z^(N-1) ))/(1+e^(-Z^(N-1) ) )^2  								(14)
For ReLu activation:
σ^' (Z^(N-1) )={█(1 if Z^(N-1)>0@0 esle              )┤									(15)
Substitute equation 11, 12, 13 into equation 10, we obtain the derivation of L wrt W^(N-1) as:
dL/(dW^(N-1) )=(A^N-Y)W^N σ^' (Z^(N-1) ) A^(N-1)								(16)
Apply the chain rule going back to the (N-2)th layer, we can obtain the derivative of L wrt W^(N-1) as:
dL/(dW^(N-2) )=dL/(dZ^N )  (dZ^N)/(dA^(N-1) )  (dA^(N-1))/(dZ^(N-1) )  (dZ^(N-1))/(dA^(N-2) )  (dA^(N-2))/(dZ^(N-2) )  (dZ^(N-2))/(dW^(N-2) )							
=(A^N-Y)W^N W^(N-1) σ^' (Z^(N-1) ) σ^' (Z^(N-2) ) A^(N-2)							(17)
Similarly, we can derivate the derivative of L wrt W^(N-i) for i=1,2,…N-1 as follows:
dL/(dW^(N-i) )=(A^N-Y)W^N W^(N-1)…W^(N-i) σ^' (Z^(N-1) ) σ^' (Z^(N-2) )…σ^' (Z^(N-i) ) A^(N-i) 			(18)
To compute the derivative of L wrt bias b, we also use chain rule and start from the last layer. The derivative of L wrt b at Nth layer is:
dL/(db^N )=dL/(dZ^N )  (dZ^N)/(db^N )=(A^N-Y) 									(20)
The derivative of L wrt b at (N-1)th layer is obtained by using the chain rule as:
dL/(db^(N-1) )=dL/(dZ^N )  (dZ^N)/(dA^(N-1) )  (dA^(N-1))/(dZ^(N-1) )  (dZ^(N-1))/(db^(N-1) )=(A^N-Y) W^(N-1) σ^' (Z^(N-1) ) 					(21)
The derivative of L wrt b at (N-2)th layer is obtained by using the chain rule as:
dL/(db^(N-2) )=dL/(dZ^N )  (dZ^N)/(dA^(N-1) )  (dA^(N-1))/(dZ^(N-1) )  (dZ^(N-1))/(dA^(N-2) )  (dA^(N-2))/(dZ^(N-2) )  (dZ^(N-2))/(db^(N-2) )=(A^N-Y) W^(N-1) W^(N-2) σ^' (Z^(N-1) ) σ^' (Z^(N-2) ) 		(22)
The derivative of L wrt b at ith layer for i=1,2,…N-1 is obtained by using the chain rule as follows:
dL/(db^(N-i) )=(A^N-Y) W^(N-1) W^(N-2)…W^(N-i) σ^' (Z^(N-1) ) σ^' (Z^(N-2) )…σ^' (Z^(N-i) ) 				(23)
