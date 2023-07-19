Chapter 1. The Machine Learning Landscape

1. How would you define Machine Learning?
  A computer program (machine) learns if its performance on some specific task increases by Experience.

2. Can you name four types of problems where it shines?
  - Problems for which existing solutions require a lot of fine-tuning or long lists of rules.
  - Complex problems for which using a traditional approach yields no good solution.
  - Fluctuating environments: a Machine Learning system can adapt to new data.
  - Getting insights about complex problems and large amounts of data.

3. What is a labeled training set?
   A training set whose instances have desired solutions. (either class or target depending on the problem at hand (classification/regression))

4. What are the two most common supervised tasks?
  Classification and Regression tasks/problems.

5. Can you name four(4) common unsupervised tasks?
  - Clustering.
  - Dimensionality reduction/ Visualisation.
  - Anomaly/ Novelty detection.
  - Association rule learning.

6. What type of Machine Learning algorithm would you use to allow a robot to walk in various unknown terrains?
  Reinforcement Learning.

7. What type of algorithm would you use to segment your customers into multiple groups?
  Clustering algortihm.
  
8. Would you frame the problem of spam detection as a supervised learning problem or an unsupervised learning problem?
  Although spam detection problem can be conventionally framed as a supervised learning problem, -I believe- it can also be framed as an unsupervised learning problem 
  if we don't have a labeled training set. It may also be framed as a semi-supervised learning problem if we have few labeled instances in the given training set.

9. What is an online learning system?
  If a system is trained incrementally by feeding it data instances sequentially, either individually or in small groups it is called an online learning system or  incremental learning system.

10. What is out-of-core learning?
    Out-of-core learning is a version of incremental learning that is opted for scenarios where the main memory of your machine is not enough to store all training data at once. Hence, the algorithm loads part of the data, runs a training step on that data, and repeats the process until it has run on all of the data.
A thing worth remembering about this type of learning is it takes place offline.

11. What type of learning algorithm relies on a similarity measure to make predictions?
  Instance-based learning algorithm.

12. What is the difference between a model parameter and a learning algorithm’s hyperparameter?
  While the former belongs to the model - as its name implies- and is subject to change throughout the training process by the algorithm. The latter one does not belong to the model, hence it is needed to be set manually.

13. What do model-based learning algorithms search for? What is the most common strategy they use to succeed? How do they make predictions?
  They aim to come up with a model, a well enough representation of the dataset to make inferences about the unseen data (prediction). This is accomplished with the help of the minimization of a cost function (or maximization of a gain function) so that the model is representative of all the data points in the data set. They make prediction using the model that the algorithm suggested.

14. Can you name four of the main challenges in Machine Learning?
  The main challenges in ML are due to either data or algorithms. Hence, the four main challenges are directly related to them.
  - Nonrepresentative training data / Poor quality training data/ Irrelevant features (Quality)
  - Insufficient Quantity (Quantity)
  - Overfitting the training data
  - Underfitting the training data

15. If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions?
    The algorithm overfits the training data. This can be due to either having a complex algorithm with respect to your data and/or the selection of a small regularizer hyper-parameter. On the other hand, it can be due to having not a sufficient quantity of data and/or nonrepresentative/ poor quality of training data.

16. What is a test set, and why would you want to use it?
    A test set is data that is not introduced to the machine throughout the training process so that it can be optimally used to evaluate your model performance in real-life (production) settings.

17. What is the purpose of a validation set?
    The purpose of a validation set is to have an initial idea of your model(s) performance on "unseen" data. It prevents the data leakage phenomenon to occur on a test set. It is also used to compare the performance of candidate models and hyper-parameter tuning.

18. What is the train-dev set, when do you need it, and how do you use it?
   

19. What can go wrong if you tune hyperparameters using the test set?
    A phenomenon called data leakage occurs, your model may learn to optimize its parameters according to details in your test set. As a result, bad performance in real-life (production) settings.


  


  

   
