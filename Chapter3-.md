#Exercises

1. Which Linear Regression training algorithm can you use if you have a training set with millions of features?
   All type of Gradient Descent algortihms if you have a sufficient RAM, otherwise, mini-bach GD or Stochastic GD.

2. Suppose the features in your training set have very different scales. Which algorithms might suffer from this, and how? What can you do about it?
  All type of Gradient descent algortihms might suffer. Speed of learning process might be affected negatively due to domination of the
  parameters of the higher scale features. Standardizing all the numerical features prior to the learning process would fix the problem.

3. Can Gradient Descent get stuck in a local minimum when training a Logistic Regression model?
   No, it is a convex optimization problem.

4. Do all Gradient Descent algorithms lead to the same model, provided you let them run long enough?
   It dependends on couple of variables. The first and foremost is the shape of the cost function. If the cost function is non-convex, is is very less likely.
   Nevertheless, even if the cost function is convex, there i a high probability that your SGD algorithm would stuck in a local-minima. However, with the help
   of little luck, if the learning schedule/learning rate are set good enough, all GD algorithms could lead to the same model. Shortly, In theary yes; but it
   is very unlikely.

5. Suppose you use Batch Gradient Descent and you plot the validation error at every epoch. If you notice that the validation error consistently goes up, 
  what is likely going on? How can you fix this?
  It is very likely that your model has overfitted to your training set. You can either, add data (increase quantity), increase quality of your data and/or chose 
  a less complex model and/or implement a regularizer to your model.

6. Is it a good idea to stop Mini-batch Gradient Descent immediately when the validation error goes up?
   It probably would not a good thing. It is good to keep in mind that Mini-batch Gradiend Descent have a stochastic component. Hence, by chance a 'mini-batch'
   of slightly different info from the information of validation set could be learned by algorithm which would result in temporary performance decrease on
   the validation set.

7. Which Gradient Descent algorithm (among those we discussed) will reach the vicinity of the optimal solution the fastest? Which will actually converge?
   How can you make the others converge as well?
   While SGD would reach the vicinity of the optimal solution the fastest, while the BGD reach with the least steps. BGD will converge. On the other hand,
   if the learning schedule of SGD and MBGD set cleverly they would both converge as well.

8. Suppose you are using Polynomial Regression. You plot the learning curves and you notice that there is a large gap between the training error and
   the validation error. What is happening? What are three ways to solve this?
   Your model has overfitted! Increase quantity and/or quality of your training data. Use less complex polynomial regression model with less polynomial degrees,
   increase C hyper-parameter which is the inverse of alpha (regularizer hyper-parameter).

9. Suppose you are using Ridge Regression and you notice that the training error and the validation error are almost equal and fairly high. Would you say that
   the model suffers from high bias or high variance? Should you increase the regularization hyperparameter α or reduce it?
   The model suffers from high bias (it underfits to the training data). You better decrease the regularization hyper-parameter.

10.Why would you want to use:

a. Ridge Regression instead of plain Linear Regression (i.e., without any regularization)?
b. Lasso instead of Ridge Regression?
c. Elastic Net instead of Lasso?

Aa. To make your model less complex. (increase variability/ reduce bias of your model)
Ab. To get rid of the least important features. (to perform feature selection as you train your model) . However, keep in mind that Lasso always prone to problems.
Ac. If your data matrix is not singular (#features > #data points) and/or multicollinearity exist between your features. Also, the sweetspot!

11. Suppose you want to classify pictures as outdoor/indoor and daytime/nighttime. Should you implement two Logistic Regression classifiers or
    one Softmax Regression classifier?
    Class labels in the mentioned problem are not mutually exclussive. Hence, Softmax Regression can not handle with the poblem. (Do not confuse between
    multilabel and multioutput problems!) Hence, two Logistic regression for each class pairs would be more suitable!

12. Implement Batch Gradient Descent with early stopping for Softmax Regression (without using Scikit-Learn).
    

    
   
   
   
   
