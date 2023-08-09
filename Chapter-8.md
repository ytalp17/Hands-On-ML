#  Dimensionality Reduction Exercises

1- What are the main motivations for reducing a dataset’s dimensionality? What are the main drawbacks?

The main motivation for reducing a dataset's dimensionality is to optimize the training performance of the estimator by 
reducing the dimensions that have slight information about the data at hand. Consequently, dimensionality reduction would
lead to a faster training process. Moreover, being able to visualize important features is another motivation for dimension reduction.

On the other hand, some dimensionality algorithms can be computationally expensive and add further complexity to your ML pipeline. 
One of the most important drawbacks is the fact that transformed features are hard to interpret. Also, in some cases, they could negatively
affect the training performance of the predictor.

2- What is the curse of dimensionality?

As the dimensionality of the data space increases the probability of an observation being an outlier increases. Moreover, 
the average distance between randomly selected two points increases exponentially (Sparse). Given these, learning (generalizing) from the data gets 
exponentially harder. Thus, you would need more and more data in order to generalize well. Consequently, the probability of overfitting increases 
at higher dimensions.

3- Once a dataset’s dimensionality has been reduced, is it possible to reverse the operation? If so, how? If not, why?

If the main objective is to reduce dimensionality, then it is not possible to reverse the operation. It's mainly because of the fact that 
the dimension which was discarded also carries information (possibly slightly) about the data at hand. You cant have to eat your cake and have it too...

4- Can PCA be used to reduce the dimensionality of a highly nonlinear dataset?

In theory, yes you can, but probably would end up losing a lot of information (variance) from the initial data. On the other hand, variations of PCA
such as Kernel PCA which utilizes infinite dimension mapping can be used for the nonlinear dataset.

5- Suppose you perform PCA on a 1,000-dimensional dataset, setting the explained variance ratio to 95%. How many dimensions will the resulting dataset have?

It is hard to say. number of dimensions the resulting dataset will have strictly depends on the structure of the initial data.

6- In what cases would you use vanilla PCA, Incremental PCA, Randomized PCA, or Kernel PCA?

Vanilla PCA: For the linear medium datasets.
Incremental PCA: For the linear large datasets, and also for online tasks.
Randomized PCA: Same as Vanilla PCA, if you have a time restriction.
Kernel PCA: For the non-linear datasets.

7- How can you evaluate the performance of a dimensionality reduction algorithm on your dataset?

Since it is an unsupervised learning algorithm it is hard to evaluate the performance of them. However, if you build up a pipeline you can evaluate its perfomance
on the validation set together with your predictor algorithm, and of course, you need to keep the hyper-parameter of the predictor algorithm constant.

Another way to evaluate the performance of them is to calculate reconstruction error. Intuitively, the one with the least reconstruction error, hence the least 
information loss would be a better performant.

8- Does it make any sense to chain two different dimensionality reduction algorithms?

It does if you want to reduce the total time of training. You man chose to use less complex algorithms (such as PCA) to get rid of a large number of useless dimensions,
and then use more complex algorithms to keep really important dimensions.

9- Apply PCA + RF Classifier on Mnist Dataset

```Python
import numpy as np
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

#Fetch the data
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
X = mnist["data"]
y = mnist["target"].astype(np.uint8)

#train/test split
X_train = X[:60000]
y_train = y[:60000]
X_test = X[60000:]
y_test = y[60000:]

#Model Training
#start time
start = time.time()

RF_classifer = RandomForestClassifier(n_estimators = 100, random_state= 42)
RF_classifer.fit(X_train, y_train)
                 
#end time               47.79407095909119  
end = time.time()
print("training time:", end - start)

#Evalaute on test dataset
test_pred = RF_classifer.predict(X_test)
acc_score = accuracy_score(test_pred, y_test)
print("\n accuracy:",acc_score)       #0.9705


#RF on reduced dataset
#let's start with arbitraryly large number of components and find the one that contains %95 of variance
n=500
pca = PCA(n_components= n)
pca.fit(X_train)

#n = 154
n = np.argmax(np.where(np.cumsum(pca.explained_variance_ratio_)<=0.951))

#dim reduction
pca = PCA(n_components= n)
tr_X_train = pca.fit_transform(X_train)

#Model Training
#start time
start = time.time()

RF_classifer = RandomForestClassifier(n_estimators = 100, random_state= 42)
RF_classifer.fit(tr_X_train, y_train)
                 
#end time               128.53290271759033 
end = time.time()
print("training time:", end - start)

#Evalaute on test dataset
#do not forget to transform first
tr_X_test = pca.transform(X_test)
test_pred = RF_classifer.predict(tr_X_test)
acc_score = accuracy_score(test_pred, y_test)
print("\n accuracy:", acc_score)    #0.94

```
