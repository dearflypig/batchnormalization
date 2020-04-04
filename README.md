batchnormalization<br>
MyBN是用自定义的BN层构建的神经网络，代码中给出了对一个回归任务的学习和测试<br>
bn_pytorch是用pytorch提供的BN层构建的神经网络<br>

# 一、相关理论介绍<br>

## 1、当采用较深层数的网络进行训练时：<br>
* 后层网络需要不停调整来适应输入数据分布的变化，导致网络学习速度的降低；<br>
* 当在神经网络中采用饱和激活函数时（例如sigmoid、tanh激活函数），网络的训练过程容易陷入梯度饱和区，梯度会变得很小甚至接近于0，参数的更新速度会减慢；<br>

## 2、Batch Normalization
* 在对输入数据进行激活函数的处理之前，对每个特征进行normalization，让每个特征都有均值为0，方差为1的分布，让数据位于激活函数的敏感区域；<br>
<br>
* 计算方法：<br>
![image](https://github.com/dearflypig/batchnormalization/blob/master/picture/0.png)<br>
<br>
# 二、实现功能：<br>
>编写BN层的代码，并利用一个简单的数据进行检测，确保自己编写的BN层能实现对数据的标准化处理：<br>
## 1、使用pytorch提供的BatchNorm1d模块<br>
![image](https://github.com/dearflypig/batchnormalization/blob/master/picture/1.png)<br>
![image](https://github.com/dearflypig/batchnormalization/blob/master/picture/2.png)<br>
## 2、使用自己编写的BN层
![image](https://github.com/dearflypig/batchnormalization/blob/master/picture/3.png)<br>
![image](https://github.com/dearflypig/batchnormalization/blob/master/picture/4.png)<br>
![image](https://github.com/dearflypig/batchnormalization/blob/master/picture/5.png)<br>
* 两者输出是相同的，说明自己编写的BN层可以实现对数据的标准化<br>
<br>
# 三、体会BN层的效果

## 数据：<br>
制作伪数据来模拟一个回归的任务，看在加BN层前后神经网络模型的预测能力，采用的数据为y=a\*x^2+b，对y数据加上噪声点来更加真实的展示；<br>
<br>
## 网络模型：<br>
构造3个四层的神经网络，分别是不加BN层的网络、使用pytorch提供的BatchNorm1d模块的网络，使用自己编写的MyBN层的网络，对训练数据进行学习，并在测试数据上观察拟合的曲线。<br>
<br>
## 其他：<br>
激活函数：tanh<br>
Epoch：10<br>
<br>
## （一）当输入数据x的范围在（-1,1）区间，对比三种模型的效果：<br>
* 数据：<br>
![image](https://github.com/dearflypig/batchnormalization/blob/master/picture/6.png)<br>
<br>
1、不使用BN层，左图展示的是每个epoch的误差，右图是测试数据及模型拟合的曲线：<br>
![image](https://github.com/dearflypig/batchnormalization/blob/master/picture/7.png)<br>
<br>
2、使用pytorch提供的BatchNorm1d，左图展示的是每个epoch的误差，右图是测试数据及模型拟合的曲线：<br>
![image](https://github.com/dearflypig/batchnormalization/blob/master/picture/8.png)<br>
<br>
3、使用自己编写的MyBN层，左图展示的是每个epoch的误差，右图是测试数据及模型拟合的曲线：<br>
![image](https://github.com/dearflypig/batchnormalization/blob/master/picture/9.png)<br>
<br>
### 分析：<br>
当数据分布在（-1,1）的区间时，由于处在激活函数的敏感区，所以增加BN层的效果不明显<br>
<br>
## （二）输入数据x的范围在（-5,5）时，对比三种模型的效果：<br>
* 数据：<br>
![image](https://github.com/dearflypig/batchnormalization/blob/master/picture/10.png)<br>
1、不使用BN层，左图为loss，右图为拟合曲线<br>
![image](https://github.com/dearflypig/batchnormalization/blob/master/picture/11.png)<br>
2、使用pytorch提供的BatchNorm1d，左图为loss，右图为拟合曲线：<br>
![image](https://github.com/dearflypig/batchnormalization/blob/master/picture/12.png)<br>
3、使用自己编写的MyBN层，左图为loss，右图为拟合曲线:<br>
![image](https://github.com/dearflypig/batchnormalization/blob/master/picture/13.png)<br>
### 分析：<br>
当输入的数据在（-5,5）时，大量数据位于激活函数的饱和区，此时如果无BN作用，该神经网络将几乎没有学习能力。而加上pytorch提供的BatchNorm1d模块和自定义的MyBN层都有不错的效果<br>
<br>

# 四、自定义BN层时遇到的问题<br>
1、我初始的想法是，对于BN层的forward和backward都由自己编写，利用dL/dy以及正向过程存储的变量求出dL/dγ、dL/dβ、dL/dx，但是由于要利用后层的梯度，并且对前层参数的更新有影响，而在pytorch搭建的网络里，其他层的反向传播更新参数的过程都是自动完成的，所以一开始在bn层参数的更新上遇到了麻烦；但后来查资料，可以把自定义BN层的参数放入网络的param_groups中，由pytorch来对我需要更新的参数进行更新，所以只需要编写forward部分即可；<br>

2、由于在训练模型时是基于minibatch的，而测试的时候数据无需分批，所以要区分这两个过程，参考了网上的资料，用两个参数记录全局的mean和var。

