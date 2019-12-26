# ML CW

## Task1

> TODO

## Task2

I used a dataset of X image for all models involved in task 2 as this took an average of X to fit and test each model, and was of a sufficient enough size to produce results that are aligned with expectations. The tests can be run with `task2/run.py` and graphed with `task2/graph.py`.

### Principle Component Analysis

A | B


![f1_pc_v_dimensionality](task2/figs/acc_v_pca.png)

![f1_pc_v_dimensionality](task2/figs/f1_pc_v_dimensionality.png)

Above are the results of the overall accuracy of a Linear SVM when run against data that has been reducing in dimensionality at varying levels by using PCA. The SVM was run with a C value of 1. The results show a decrease in accuracy with a decrease in dimensionality. This is to be expected as PCA removes information from the data. What's more, PCA attempts to remove the least important information when reducing dimensionality, so we can expect the rate of change of accuracy to also be decreasing. The accuracy score produced by testing on test data is also lower than the score produced by cross validation. This is because, the model is trained on the same data used to calculate the validation scores. If we were to productionise this model we would use PCA to convert the data to a dimensionality of 60% of the original as this is the point where we lose minimal/no accuracy.

The F1 scores for each class are a considerably more noisy as they have each been trained on effectively one tenth of the size of data. 

### Polynomial Kernel SVM:

![poly_c_v_acc](task2/figs/poly_c_v_acc.png)

From the chart above we can see that the more optimal combination of degree and C value tested is 3 and X. When degree=2, the decision boundary is not flexible enough to allow for the classification of as many images as when the degree is 3. When the degree is 4, the decision surface is overfitting to the values in the test data.

![poly_c_v_f1](task2/figs/poly_c_v_f1.png)

![ploy_degree_v_acc](task2/figs/poly_degree_v_acc.png)

The graph above shows accuracy results of a changing polynomial kernel degree. The curve peaks at a polynomial value of 2, after which the SVM is likely overfitting to the training data. From degree 0 (linear) to degree 2, the kernel is making appropriate generalisations of the data.

### RBF Kernel SVM

![rbf_c_v_acc](task2/figs/rbf_c_v_acc.png)

The accuracy of the RBF Kernel begins drastically increasing in accuracy around a C value of 0.1, at this point, the boundaries being drawn by the kernel function begin separating the different classes. This continues until a C value of approximately 1.0, where the accuracy reaches its maximum. It is likely the case that this is the maximum level of accuracy that can be achieved with this RBF kernel, and that if we were to run tests with a larger value of C, the test accuracy would begin to drop.

![rbf_c_v_f1](task2/figs/rbf_c_v_f1.png)

The graph above shows that for a value of C < 0.05, the F1 scores for each class are unresponsively low. This is to be expected as the RBF kernels function will have a large bias toward drawing the decision boundary through the center-most point of the data, maximising the margin. As the C is increased, the kernel will have a greater bias toward attempting to correctly classify as many data points as possible, increasing accuracy.

## Task3

### Full Connected Network

I created a class `FCN()` which takes as parameter a number n of hidden layers. The class creates the number of hidden layers, output layer. Each of the hidden layers has 3072 channels, therefore there is no information lass at each layer.

![FCN_ttrain_v_hl](task3/figs/FCN_acc_v_hl.png)

The graph above shows that the accuracy of model increase with each layer added until plateauing at around X hidden layers. This is an example of X whereby the earlier layers have trained on the output less.

> TODO: fix whatever it is im talking about here

### Convolutions

Most models online use convolutional layers in order to decrease training time and allow for greater parameter tuning. I decided to try my model with two, three and four layers of convolutions, all connected to two linear layers of fully connected nodes.

> TODO: Reword

| Model ID | Number of convolutional layers | Accuracy | Average F1 Score |
| -------- | ------------------------------ | -------- | ---------------- |
| Conv1    | 1                              |          |                  |
| Conv2    | 2                              |          |                  |
| Conv3    | 3                              |          |                  |

### Pooling





### Conclusion

> TODO:
>
> SVM would be better suited in X environment because..
>
> CNN would be better suited in X environment because..
