# ML CW

## Task1

> TODO?

## Task2

I used a dataset of X image for all models involved in task 2 as this took an average of X to fit and test each model, and was of a sufficient enough size to produce results that are aligned with expectations. The tests can be run with `task2/run.py` and graphed with `task2/graph.py`.

### Principle Component Analysis

<center>
    <img src="task2/figs/acc_v_pca.png" alt="acc_v_pca" style="width:49%;" />
    <img src="task2/figs/f1_pc_v_dimensionality.png" alt="f1_pc_v_dimensionality" style="width:49%;" />
</center>

The results indicate that there is a reduction in accuracy with a reduction in dimensionality after around 60% of the original dimensionality. We can therefore deduce that an approximately 40% of the data in each image is redundant. The accuracy score resulting from test data is lower than that resulting from the cross validation. This is to be expected as the model was not trained on any of the test data, but was (mostly) trained on the validation data in previous iterations. We can assume that the rise in accuracy resulting from the 10% decrease in dimensionality at 50% is due to noisy data, and therefore still fits the expected trend. The F1 scores are considerably more noisy than the accuracy because each category has been trained on one tenth of the total data. We can therefore expect that trends uncovered when looking at the overall accuracy would be considerably more distinguished than those produced from examining each class separately. For best results, we would apply a reduction in dimensionality of approximately 60% of the original dimensionality as we would increase the training time considerably whilst losing little relevant information.

### Polynomial Kernel SVM:

<img src="task2/figs/poly_c_v_acc.png" alt="poly_c_v_acc" style="width:49%;" />

The polynomial kernel was most accurate on the data with a C value of around 0 for kernels of degree two, three, and four. When using a C value smaller than 10, the decision boundary is not flexible enough to allow for the correct classification of as many images and the data is under-fit. The accuracy decreases after a C value of zero as the margin is not soft enough and the decision boundary is overfitted to the training data. The higher degree kernel overfits more than lower degree because it is capable of drawing a more detailed decision boundary boundary.

<center>
    <img src="task2/figs/poly_degree_v_acc.png" alt="f1_pc_v_dimensionality" style="width:49%;" />
    <img src="task2/figs/poly_c_v_f1.png" alt="acc_v_pca" style="width:49%;" />
</center>

The graph above shows accuracy results of a changing polynomial kernel degree. The curve peaks at a polynomial value of 2, after which the accuracy begins to drop. At a degree of 3, the SVM is slightly overfitting as the test accuracy drops while the CV accuracy remains the same. However, after this the CV accuracy also falls. This is evidence to suggest that the kernel is not overfitting... From degree 1 to degree 2, the kernel is making appropriate generalisations of the data.

> TODO: revisit

- Test score lower than CV
- Kernel is not overfitting as cv and test data decrease equally so just less effective in general
- Degree 2 is making appropriate generalisations

> TODO: Remove degree 0



### RBF Kernel SVM

<center>
    <img src="task2/figs/rbf_c_v_acc.png" alt="" style="width:49%;" />
    <img src="task2/figs/rbf_c_v_f1.png" alt="" style="width:49%;" />
</center>

> TODO: Add f1 to graph and talk about difference between f1 and accuracy scores.

- C value below 0.01 causes decision plane to not flexible enough
- Accuracy peaks at value of C=10 where the margin is soft but not too soft
- Data is such that the kernel does not begin to overfit significantly for any value tested
- Most likely the maxim accuracy that can be achieved with this set of parameters and only changing the value of C

The accuracy of the RBF Kernel begins drastically increasing in accuracy around a C value of 0.1, at this point, the boundaries being drawn by the kernel function begin separating the different classes. This continues until a C value of approximately 1.0, where the accuracy reaches its maximum. It is likely the case that this is the maximum level of accuracy that can be achieved with this set of parameters for the RBF kernel, and that if we were to run tests with a larger value of C, the test accuracy would eventually begin to drop.

- Kernel function is not capable of translating the data into a linearly separable transformation at C<0.01. 
- Then starts to be able to draw a margin that classifies significantly more
- Levels off at C=100

The graph above shows that for a value of C < 0.01, the F1 scores for each class are unresponsively low. This is to be expected as the RBF kernels function will have a very soft margin and is resultantly misclassifying too many points in the training data. As the C is increased, the kernel will have a greater bias toward a decision boundary that correctly classifies more of the training data. After a C value of 1, there is a slight decrease in accuracy, this is weak evidence for overfitting as if this were the case, we would expect to see a bigger decrease in the accuracy score from the test data.



<center>
    <img src="task2/figs/rbf_gamma_v_acc.png" alt="" style="width:49%;" />
    <img src="task2/figs/rbf_gamma_v_f1.png" alt="" style="width:49%;" />
</center>

As gamma in essence changes the size of the sphere of influence of each data point, we would expect to see some peak value with the accuracy falling to 0 with a large enough and small enough gamma. We can see that at a gamma value of X, the RBF kernel weights the surrounding labelled data points with an appropriate scale in order to assess the category of the input to the highest degree of accuracy.

## Task3

### Full Connected Network

I decided to start the design of my model with an FCN and build up from there as a CNN must finish with at least one fully connected layer. I created a class `FCN()` which takes as a parameter a number of hidden layers the model should use. The class creates the number of hidden layers of size 3072 (3 * 32 * 32), and one output layer. As each of the hidden layers has 3072 channels, therefore there is no information loss as this is the size of the data fed into the network (pixels x RGB). The results are as follows. Initial research uncovered a model with only two linear layers [1]. The model consisted of numerous convolutional and max pooling layers, however the inout data was of a much higher dimensionality. Therefore I decided to use a range that both started lower and finished higher than the model used previously.

| Hidden Layers | Accuracy | F1 Score | Runtime |
| :-----------: | :------: | :------: | :-----: |
|       1       |  0.3675  |  0.3827  | 0:8:46  |
|       2       |  0.3989  |  0.3991  | 0:18:36 |
|       3       |  0.3945  |  0.3971  | 0:27:31 |

The table above shows that the accuracy of model increase with from 1 layer to 2 and consequently the complexity of the decision boundary increases with an extra layer and can categorise images more accurately. Adding a third layer to the graph then decreases the accuracy. This is likely as a result of the graph running into the vanishing gradient problem whereby the weights of the earlier layers of the model are updated less as the gradient diminishes with the propagation through each layer.

As I will be adding convolutional layers and pooling kernels I do not require the fully connected layers of my graph to be able to produce the most optimal decision boundary by themselves. Therefore I believe 2 fully connected layers for my model should be enough to define detailed enough decision boundaries.

### Adding Convolution Layers

Most models online use convolutional layers in order to decrease training time and allow for greater parameter tuning. I decided to try my model with two, three and four layers of convolutions, all connected to two linear layers of fully connected nodes. After each convolutional layer, I apply a maxpooling layer with a size of 4 and a stride of 4.

> TODO: Reword

| Model ID | Convolutional Layers | Accuracy | Average F1 Score | Runtime |
| :------: | :------------------: | :------: | :--------------: | ------- |
|  Conv1   |          1           |  0.3481  |      0.361       | 0:15:39 |
|  Conv2   |          2           |  0.4702  |      0.450       | 0:49:37 |
|  Conv3   |          3           |  0.4534  |      0.490       | 1:36:39 |

Based off the results from this test, I decided that my model should have 2 convolution layers as the results showed no improvement with a third layer. Furthermore, I will be using convolution layers in my model which will allow for additional feature extraction.

### Trialing Different Pooling Layers

I decided to trial an average pooling layer as well as max pooling. Average pooling layers are typically less effective as MaxPool favours the most significant features detected by the convolution layer's kernel rather than the average.

| Method  | Accuracy | F1 Score | Runtime |
| ------- | -------- | -------- | ------- |
| MaxPool | 0.4702   | 0.4502   | 0:49:37 |
| AvgPool | 0.3469   | 0.3293   | 0:43:40 |

### Activation Functions

I decided to trial the following four function because two of them (Sigmoid and Tanh) normalise the values output by the convolutional layers, and two of them (ReLU and softMax) do not. Therefore I should be able to see which approach my model is better suited to. In addition, I decided to test the ReLU6 method as Krizhevsky managed to achieve higher results than with standard ReLU [2].

| Activation Function | Accuracy | F1    | Runtime |
| ------------------- | -------- | ----- | ------- |
| None                | 0.4702   | 0.471 | 0:49:37 |
| ReLU                | 0.5863   | 0.583 | 0:44:24 |
| Sigmoid             | 0.1924   | 0.092 | 0:42:38 |
| SoftMax             | 0.1000   | 0.018 | 0:40:41 |
| Tanh                | 0.1000   | 0.018 | 0:40:51 |
| ReLU6               | 0.617    | 0.617 | 0:46:2  |

<img src="task3/figs/Activation_Comparison_acc.png" alt="Activation_Comparison_acc " style="zoom:50%;" />

We can see that ReLU6 is performing the best just as in Krizhevsky work. This is likely because it offers some normalisation in the form of an upper and lower bound of output, but does not affect a large proportion of values. I will therefore continue to build the model with this choice of activation function.

### Channels

| Channel format (conv1, conv2) | Accuracy Score | F1 Score | Runtime |
| ----------------------------- | -------------- | -------- | ------- |
| (8,24)                        | 0.615          | 0.612    | 0:44:24 |
| (5, 15)                       | 0.595          | 0.593    | 0:24:18 |
| (6, 18)                       | 0.619          | 0.607    | 0:29:14 |
| (10,20)                       | 0.63           | 0.633    | 0:33:20 |
| (14,42)                       | 0.628          | 0.629    | 1:31:22 |

Next I have decided to trial varying the numbers of output channels of each convolutional layer. The most optimal number of layers trailed appears to be 10 for the first layer and 20 for the next. This is therefore the level where enough features are being extracted in each convolutional layer.

### Epochs

<img src="task3/figs/epoch_acc.png" alt="epoch_acc " style="width:49%;" />

The graph above shows the accuracy of the model on test data at each epoch. From epochs one to five, the model is fitting the data to an appropriate mount. After five epochs, the model begins to over fit the data. The test data remained the same throughout the training of this model and therefore was never trained on. Typically test data is not used to adjust parameters of the model and is instead used to evaluate the model at the complete end of development. However, I am not planning to productionised my model and am instead experimenting with different model architectures on the dataset. I am therefore allocating no data for a final test set to evaluate the model at the end in favour of more data to train the model on during development. In this way, I am choosing to have a higher accuracy of the model at the cost of a lower accuracy of the evaluation of my model.

### Learning Rate



### Conclusion

> TODO:
>
> SVM would be better suited in X environment because..
>
> CNN would be better suited in X environment because..

### Works Cited

1. Haj-Hassan, Hawraa, et al. "Classifications of multispectral colorectal cancer tissues using convolution neural network." *Journal of pathology informatics* 8 (2017).
2. Alex Krizhevsky. "Convolutional Deep Belief Networks on CIFAR-10".