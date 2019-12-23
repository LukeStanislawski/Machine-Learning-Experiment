# ML CW

## Task2

I used a dataset of X image for all models involved in task 2 as this took an average of X to fit and test each model, and was of a sufficient enough size to produce results that are aligned with expectations. The tests can be run with `task2/run.py` and graphed with `task2/graph.py`.

### Principle Component Analysis

<img src="task2/figs/acc_v_pca.png" alt="acc_v_pca" style="zoom:100%;" />

Above are the results of the overall accuracy of a Linear SVM when run against data that has been reducing in dimensionality at varying levels by using PCA. The SVM was run with a C value of 1. The results show a decrease in accuracy with a decrease in dimensionality. This is to be expected as PCA removes information from the data. What's more, PCA attempts to remove the least important information when reducing dimensionality, so we can expect the rate of change of accuracy to also be decreasing. The accuracy score produced by testing on test data is also lower than the score produced by cross validation. This is because, the model is trained on the same data used to calculate the validation scores. If we were to productionise this model we would use PCA to convert the data to a dimensionality of 60% of the original as this is the point where we lose minimal/no accuracy.

![f1_pc_v_dimensionality](task2/figs/f1_pc_v_dimensionality.png)

The F1 scores for each class are a considerably more noisy as they have each been trained on effectively one tenth of the size of data. 

### Polynomial Kernel SVM: 

![ploy_degree_v_acc](task2/figs/poly_degree_v_acc.png)

The graph above shows accuracy results of a changing polynomial kernel degree. The curve peaks at a polynomial value of 2, after which the SVM is likely overfitting to the training data. From degree 0 to 2, the kernel is making appropriate generalisations of the data



## Task3

> TODO