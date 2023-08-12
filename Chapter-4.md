#Exercises

1. Which Linear Regression training algorithm can you use if you have a training set with millions of features?
   
   All types of Gradient Descent algorithms if you have sufficient RAM, otherwise, mini-bach GD or Stochastic GD.

2. Suppose the features in your training set have very different scales. Which algorithms might suffer from this, and how? What can you do about it?
   
  All types of Gradient descent algorithms might suffer. The speed of the learning process might be affected negatively due to the domination of the
  parameters of the higher-scale features. Standardizing all the numerical features prior to the learning process would fix the problem.

3. Can Gradient Descent get stuck in a local minimum when training a Logistic Regression model?
   
   No, it is a convex optimization problem.

4. Do all Gradient Descent algorithms lead to the same model, provided you let them run long enough?
   
   It depends on a couple of variables. The first and foremost is the shape of the cost function. If the cost function is non-convex, it is very less likely.
   Nevertheless, even if the cost function is convex, there is a high probability that your SGD algorithm would get stuck in local minima. However, with the help
   of little luck, if the learning schedule/learning rate is set well enough, all GD algorithms could lead to the same model. Shortly, In theory, yes; but it
   is very unlikely.

5. Suppose you use Batch Gradient Descent and you plot the validation error at every epoch. If you notice that the validation error consistently goes up, 
  what is likely going on? How can you fix this?
  It is very likely that your model has overfitted your training set. You can either, add data (increase quantity), increase the quality of your data, and/or chose 
  a less complex model and/or implement a regularizer to your model.

6. Is it a good idea to stop Mini-batch Gradient Descent immediately when the validation error goes up?
    
   It probably would not be a good thing. It is good to keep in mind that Mini-batch Gradient Descent has a stochastic component. Hence, by chance a 'mini-batch'
   of slightly different info from the information of the validation set could be learned by the algorithm which would result in a temporary performance decrease on
   the validation set.

7. Which Gradient Descent algorithm (among those we discussed) will reach the vicinity of the optimal solution the fastest? Which will actually converge?
   How can you make the others converge as well?

   While SGD would reach the vicinity of the optimal solution the fastest, while the BGD reach with the least steps. BGD will converge. On the other hand,
   if the learning schedule of SGD and MBGD were set cleverly they would both converge as well.

8. Suppose you are using Polynomial Regression. You plot the learning curves and you notice that there is a large gap between the training error and
   the validation error. What is happening? What are three ways to solve this?

   Your model has overfitted! Increase the quantity and/or quality of your training data. Use a less complex polynomial regression model with fewer polynomial degrees,
   increase C hyper-parameter which is the inverse of alpha (regularizer hyper-parameter).

9. Suppose you are using Ridge Regression and you notice that the training error and the validation error are almost equal and fairly high. Would you say that
   the model suffers from high bias or high variance? Should you increase the regularization hyperparameter α or reduce it?

   The model suffers from high bias (it underfits to the training data). You better decrease the regularization hyper-parameter.

10. Why would you want to use:

a. Ridge Regression instead of plain Linear Regression (i.e., without any regularization)?

b. Lasso instead of Ridge Regression?

c. Elastic Net instead of Lasso?


Aa. To make your model less complex. (increase variability/ reduce the bias of your model)

Ab. To get rid of the least important features. (to perform feature selection as you train your model) . However, keep in mind that Lasso is always prone to problems.

Ac. If your data matrix is not singular (#features > #data points) and/or multicollinearity exist between your features. Also, the sweet spot!

11. Suppose you want to classify pictures as outdoor/indoor and daytime/nighttime. Should you implement two Logistic Regression classifiers or
    one Softmax Regression classifier?
    
    Class labels in the mentioned problem are not mutually exclusive. Hence, Softmax Regression can not handle the problem. (Do not confuse 
    multilabel and multioutput problems!) Hence, two Logistic regressions for each class pair would be more suitable!

12. Implement Batch Gradient Descent with early stopping for Softmax Regression (without using Scikit-Learn).


```python
rfrom sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

iris = datasets.load_iris()



#Softmax Regression with Batch Gradient Descent
np.random.seed(1743)
#prepare data
X = iris["data"][:, (2, 3)]  # petal length, petal width
#two classes
y = iris["target"]

#preprocess and add bias column all 1s
one_hot = OneHotEncoder().fit_transform(y.reshape(-1,1))
y_one_hot = one_hot.toarray()
X_bias = np.c_[np.ones((len(X),1)),X]

#split dataset
rand_indices = np.random.permutation(len(X))

train_l = int(len(X)*0.6)

X_train = X_bias[rand_indices[:train_l],:]
X_val = X_bias[rand_indices[train_l:], :]

y_train = y_one_hot[rand_indices[:train_l], :]
y_val = y_one_hot[rand_indices[train_l:],:]

#BGD
##def softmax fnc
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)



epsilon = 1e-7
eta = 0.3
n_iterations = 10000
m = len(X_train)

#randomly initialize theta
theta = np.random.randn(X_train.shape[1], y_train.shape[1])

best_loss = np.infty
Epoch = None

val_loss_vector = []
train_loss_vector = []



#training with early stop
for i in range(n_iterations):
    logit = X_train.dot(theta)
    y_prob = softmax(logit)
    
    train_loss = -np.mean(np.sum(y_train * np.log(y_prob + epsilon), axis=1))
    train_loss_vector.append(train_loss)
        
    error = y_prob - y_train
    gradients = 1/m * X_train.T.dot(error)
    theta = theta - eta * gradients
    
    
    #validaiton
    logit = X_val.dot(theta)
    y_prob = softmax(logit)
    val_loss = -np.mean(np.sum(y_val * np.log(y_prob + epsilon), axis=1))

    val_loss_vector.append(val_loss)
    
    if i % 500 == 0:
        print(i, val_loss)
    if val_loss < best_loss:
        best_loss = val_loss
    else:
        print(i - 1, best_loss)
        print(i, val_loss, "early stopping!")
        break


print("final model parameters:\n", theta)
print("best_epoch:", i)


#pred_classes = np.argmax(y_prob, axis=1)
```
    

    
   
   
   
   
