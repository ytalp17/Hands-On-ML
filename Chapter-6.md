#Decision Three Exercises (Chapter 6)

1- What is the approximate depth of a Decision Tree trained (without restrictions) on a training set with one million instances?

log2(10^6) = 20.

2- Is a node’s Gini impurity generally lower or greater than its parent’s? Is it generally lower/greater, or always lower/greater?

Gini impurity of a node is generally lower than its parent's.

3- If a Decision Tree is overfitting the training set, is it a good idea to try decreasing max_depth?

If a Decision Tree is overfitting the training set, it is a good idea to decrease mac_depth parameter. Doing that would be same as increasing regularizing hyper-parameter.

4 - If a Decision Tree is underfitting the training set, is it a good idea to try scaling the input features?

Standard scaling methods such as mean centering or standardizing would not make any difference on the performace of a decision tree algorithm since it takes care of
features in a isolation.

5 - If it takes one hour to train a Decision Tree on a training set containing 1 million instances, roughly how much time will it take to train another Decision Tree 
on a training set containing 10 million instances?

11,7h.

6 - If your training set contains 100,000 instances, will setting speed up training?

For the default settings of a decision tree on **large datasets**, setting this to true may slow down the training process. 
When using either a smaller dataset or a restricted depth, this may speed up the training.

7 - Train and fine-tune a Decision Tree for the moons dataset

```Python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

#generate data
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
#train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Model traning
dtc = DecisionTreeClassifier(random_state=42)
Params = {"criterion": ["gini", "log_loss"], "max_leaf_nodes":[3, 4, 5, 6, 7]}

dtc_cv = GridSearchCV(estimator=dtc, param_grid = Params, cv = 5, verbose = 0)
dtc_cv.fit(X_train, y_train)

#make pred
y_test_pred = dtc_cv.predict(X_test)
acc_score = accuracy_score(y_test_pred, y_test)
print("Accuracy score on test set:", acc_score, "\n")

##Plot decision boundary and the X_test data
#def plot size
plt.rcParams.update({'figure.figsize':(10,6), 'figure.dpi':100})
fig, ax = plt.subplots(1,2)

#disp training data on the lefts
disp = DecisionBoundaryDisplay.from_estimator(estimator = dtc_cv, 
                                              X = X_train, 
                                              response_method = "predict",
                                              xlabel = "feature1", ylabel = "feature2",
                                              alpha = 0.5, 
                                              cmap = plt.cm.coolwarm,
                                              ax = ax[0]
                                             )


disp.ax_.scatter(X_train[:, 0], X_train[:, 1], 
                 c=y_train, edgecolor="k",
                 cmap=plt.cm.coolwarm)
disp.ax_.set_title("Training Data")

#disp test data on the right
disp2 = DecisionBoundaryDisplay.from_estimator(estimator = dtc_cv, 
                                              X = X_test, 
                                              response_method = "predict",
                                              xlabel = "feature1", ylabel = "feature2",
                                              alpha = 0.5, 
                                              cmap = plt.cm.coolwarm,
                                              ax = ax[1],
                                            title = 'aa'
                                            
                                             )


disp2.ax_.scatter(X_test[:, 0], X_test[:, 1], 
                 c=y_test, edgecolor="k",
                 cmap=plt.cm.coolwarm)

disp2.ax_.set_title("Test Data")

plt.show()
```

8 - Grow a forest

```Python
from sklearn.model_selection import ShuffleSplit
from scipy.stats import mode

n_trees= 1000
n_instances=100
splitter= ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances, random_state=42)

tree_preds = []
for  train_index, test_index in splitter.split(X_train):
    
    #model training
    dtc_cv.fit(X_train[train_index,:], y_train[train_index])
    y_test_pred = dtc_cv.predict(X_test)
    tree_preds.append(y_test_pred)
    acc_score = accuracy_score(y_test_pred, y_test)
    print("DTC Accuracy score on test set:", acc_score, "\n")
    
tree_preds = np.vstack(tree_preds)  
y_pred_majority_votes = mode(tree_preds, axis=0, keepdims=True)
acc_score = accuracy_score(y_test, y_pred_majority_votes[0].reshape([-1]))
print("Random Forest Accuracy score on test set:", acc_score, "\n")

```

