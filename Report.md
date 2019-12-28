# ML CW

## Task1

> TODO?

## Task2

I used a dataset of X image for all models involved in task 2 as this took an average of X to fit and test each model, and was of a sufficient enough size to produce results that are aligned with expectations. The tests can be run with `task2/run.py` and graphed with `task2/graph.py`.

### Principle Component Analysis

![f1_pc_v_dimensionality](task2/figs/acc_v_pca.png)

![f1_pc_v_dimensionality](task2/figs/f1_pc_v_dimensionality.png)

Above are the results of the overall accuracy of a Linear SVM when run against data that has been reducing in dimensionality at varying levels by using PCA. The SVM was run with a C value of 1. The results show a decrease in accuracy with a decrease in dimensionality. This is to be expected as PCA removes information from the data. What's more, PCA attempts to remove the least important information when reducing dimensionality, so we can expect the rate of change of accuracy to also be decreasing. The accuracy score produced by testing on test data is also lower than the score produced by cross validation. This is because, the model is trained on the same data used to calculate the validation scores and will therefore be better at classifying the images. If we were to productionise this model we would use PCA to convert the data to a dimensionality of approximately 60% of the original dimensionality as this is the point where we lose minimal/no accuracy. The F1 scores for each class are a considerably more noisy as they have each been trained on effectively one tenth of the size of data. 

### Polynomial Kernel SVM:

![poly_c_v_acc](task2/figs/poly_c_v_acc.png)

From the chart above we can see that the more optimal combination of degree and C value tested is 2 and 0. When degree=2, the decision boundary is not flexible enough to allow for the classification of as many images as when the degree is 3. When the degree is 4, the decision surface is overfitting to the values in the test data.

![poly_c_v_f1](task2/figs/poly_c_v_f1.png)

![ploy_degree_v_acc](task2/figs/poly_degree_v_acc.png)

The graph above shows accuracy results of a changing polynomial kernel degree. The curve peaks at a polynomial value of 2, after which the accuracy begins to drop. At a degree of 3, the SVM is slightly overfitting as the test accuracy drops while the CV accuracy remains the same. From degree 0 (linear) to degree 2, the kernel is making appropriate generalisations of the data.

### RBF Kernel SVM

![rbf_c_v_acc](task2/figs/rbf_c_v_acc.png)

The accuracy of the RBF Kernel begins drastically increasing in accuracy around a C value of 0.1, at this point, the boundaries being drawn by the kernel function begin separating the different classes. This continues until a C value of approximately 1.0, where the accuracy reaches its maximum. It is likely the case that this is the maximum level of accuracy that can be achieved with this RBF kernel, and that if we were to run tests with a larger value of C, the test accuracy would begin to drop.

![rbf_c_v_f1](task2/figs/rbf_c_v_f1.png)

The graph above shows that for a value of C < 0.05, the F1 scores for each class are unresponsively low. This is to be expected as the RBF kernels function will have a large bias toward drawing the decision boundary through the center-most point of the data, maximising the margin. As the C is increased, the kernel will have a greater bias toward attempting to correctly classify as many data points as possible, increasing accuracy.

## Task3

### Full Connected Network

I decided to start at the conceptually simplest level; an FCN. I picked the following parameters as they are considered to be typical:

- Learning rate: 0.001
- Momentum: 0.9
- Epochs: 2

I created a class `FCN()` which takes as parameter a number n of hidden layers. The class creates the number of hidden layers of size 3072 (3 * 32 * 32), and one output layer. As each of the hidden layers has 3072 channels, therefore there is no information loss at each layer.

![FCN_ttrain_v_hl](task3/figs/FCN_acc_v_hl.png)

| Hidden Layers | Accuracy |
| ------------- | -------- |
| 1             | 0.3675   |
| 2             | 0.3989   |
| 3             | 0.3945   |

The graph above shows that the accuracy of model increase with each layer added until plateauing at around X hidden layers. This is an example of the vanishing gradient problem whereby the optimal decision surface for the problem requires fewer hidden layers than the model has. Therefore it would require significantly more data in order to train the model to the same level of accuracy as a model with fewer layers.

> TODO: does or doesn't this happen?

As I will be adding convolutional layers and pooling kernels I do not require the fully connected layers of my graph to be able to produce the most optimal decision boundary by themselves. Therefore I believe 2 fully connected layers for my model should be enough to define detailed enough decision boundaries.

### Adding Convolution Layers

Most models online use convolutional layers in order to decrease training time and allow for greater parameter tuning. I decided to try my model with two, three and four layers of convolutions, all connected to two linear layers of fully connected nodes. After each convolutional layer, I apply a maxpooling layer with a size of 4 and a stride of 4.

> TODO: Reword

| Model ID | Number of convolutional layers | Accuracy | Average F1 Score |
| -------- | ------------------------------ | -------- | ---------------- |
| Conv1    | 1                              | 0.3481   |                  |
| Conv2    | 2                              | 0.4702   |                  |
| Conv3    | 3                              | 0.4534   |                  |

Based off the results from this test, I decided that my model should have X convolution layers.

### Trialing Different Pooling Layers

I have decided to try out different pooling layers before tuning the parameters of the pooling layer.

| Method  | Accuracy |
| ------- | -------- |
| MaxPool | 0.4702   |
| AvgPool | 0.3469   |

This is often the case as MaxPool favours the most significant features detected by the convolution layer rather than the average case.

### Activation Functions

I decided on these 4 functions to test as Sigmoid and Tanh both normalise values output of the convolutional layers, and ReLU and softMax do not. Therefore I should be able to see which type is better for my model.

| Activation Function | Accuracy | F1    |
| ------------------- | -------- | ----- |
| None                | 0.4702   | 0.471 |
| ReLU                | 0.5863   | 0.583 |
| Sigmoid             | 0.1924   | 0.092 |
| SoftMax             | 0.1000   | 0.018 |
| Tanh                | 0.1000   | 0.018 |

![Activation_Comparison_acc](task3/figs/Activation_Comparison_acc.png)

We can see that ReLU is performing the best so we will use this. The model its current state must therefore benefit from retaining more memory with each layer as opposed to normalisation.

### Channels

Next I have decided to trial varying the numbers of output channels of each convolutional layer.

### Learning Rate

### Epochs



### Conclusion

> TODO:
>
> SVM would be better suited in X environment because..
>
> CNN would be better suited in X environment because..
