#Exercises (Ensemble Learning and Random Forests)

1- If you have trained five different models on the exact same training data, and they all achieve 95% precision, 
   is there any chance that you can combine these models to get better results? If so, how? If not, why?

It is possible to get better results by combining these models into a voting ensemble model. Given that they are diversly different, it is likely that 
not all the individual models classify the instances into exactly the same class. Hence aggregating their answers by voting is very likely to get 
better results. (Wisdom of the crowd)

2- What is the difference between hard and soft voting classifiers?

While soft-voting classifiers reach their final answer by averaging the estimated class probabilities of the corresponding classes, hard-voting classifiers 
does it by aggregating counts of votes of classes and picking up the majority class.

3- Is it possible to speed up the training of a bagging ensemble by distributing it across multiple servers? What about pasting ensembles, boosting ensembles, 
   Random Forests, or stacking ensembles?

It is possible to speed up the training of a bagging ensemble, pasting ensembles, and Random Forests by distributing it across multiple servers since sampling 
or training does not depend on other predictors in the ensemble, while it is not possible to speed up boosting the ensemble since the weights of 
the instances are changed according to predecessor model's predictions. In other words, they are sequential.

4- What is the benefit of out-of-bag evaluation?

They can be utilized as validation sets in order to get an idea about the model's generalizability. Consequently, you would have more instances for training since you
would not need an extra validation set which is always lovely!

5- What makes Extra-Trees more random than regular Random Forests? How can this extra randomness help? Are Extra-Trees 
   slower or faster than regular Random Forests?

Extra-Trees do not seek the best threshold to split in each node but use a randomly selected threshold. This characteristic of extra-trees plays the role of 
the regularizer. Hence, they can be used for the data sets where Random Forests overfit. Since they do not search for the best threshold, they are much faster than
Random Forests.

6- If your AdaBoost ensemble underfits the training data, which hyperparameters should you tweak and how?

You can increase the number of predictors and/or increase the learning rate of predictors in the AdaBoost ensemble. You can also choose a more complex base model
or/and decrease the regularizers hyper-parameter of the base model.

7- If your Gradient Boosting ensemble overfits the training set, should you increase or decrease the learning rate?

You better decrease the learning rate!

