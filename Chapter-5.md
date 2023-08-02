#Exercises

1 - What is the fundamental idea behind Support Vector Machines?

  For the linearly separable problems:
  The fundamental idea behind SVMs is to create a decision boundary with a maximum margin while minimizing margin violations (Soft Margin Classifier) 
  or even without margin violations at all (Hard Margin Classifier). (Margin violation = Misclassification)
  
  For the linearly non-separable problems:
  Utilizing kernel trick while mapping inputs into new feature spaces.

2 - What is a support vector?

Support vectors are points that lie on the margin of the learned decision boundary, in the case of classification we can say that 
they are the observations that are the closest to the decision boundary from both sides. The decision boundary only supported by these vectors. In other words, 
the rest of the observations do not play any role in the construction of the decision boundary. Consequently, new predictions are made by only utilizing these
observations.

3 - Why is it important to scale the inputs when using SVMs?

Standardizing (mean centering + scaling) of the inputs is important when using SVMs because if we have inputs with different scales, the feature with 
the higher scale(s) would dominate the decision function learning. In other words, the importance of the features with higher scales would be increased inorganically
in the decision process...

4 - Can an SVM classifier output a confidence score when it classifies an instance? What about probability?

The distance between the observation and the decision boundary could be interpreted as a confidence scare. Hence, yes. However, it is not possible for an SVM to
output prediction probability.

5 - Should you use the primal or the dual form of the SVM problem to train a model on a training set with millions of instances and hundreds of features?

The dual form of the SVM problem would be computationally more expensive than the primal form. Hence, the primal form should be used especially in the case of large/big
data sets.

6 - Say you’ve trained an SVM classifier with an RBF kernel, but it seems to underfit the training set. Should you increase or decrease γ (gamma)? What about C?

You better increase both the gamma and the C hyper-parameters.

7 - How should you set the QP parameters (H, f, A, and b) to solve the soft margin linear SVM classifier problem using an off-the-shelf QP solver?


8 - Train a LinearSVC on a linearly separable dataset. Then train an SVC and an SGDClassifier on the same dataset. See if you can get them to produce roughly
the same model.

```Python
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

#Data prep
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]

setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]


#model training
m = X_train.shape[0] 
C = 0.1

linsvc_pipeline = Pipeline([
    ('ss', StandardScaler()),
    ('lin_svc', LinearSVC(loss = "hinge",C = C, random_state = 42)), #'ovr' default
])

svc_pipeline = Pipeline([
    ('ss', StandardScaler()),
    ('lin_svc', SVC(kernel = 'linear', C = C, random_state = 42)), #'ovr' default
])

sgd_pipeline = Pipeline([
    ('ss', StandardScaler()),
    ('SGDClassifier', SGDClassifier(loss='hinge', alpha=1/(m*C), random_state = 42)), #'ovr' default
])

svc_pipeline.fit(X_train, y_train)
linsvc_pipeline.fit(X, y)
sgd_pipeline.fit(X_train, y_train)


print('LinSVC Parameters:\n')
print('w1:', linsvc_pipeline[1].coef_[0][0], ', w2:', linearsvc_pipeline[1].coef_[0][1])
print('b:',linsvc_pipeline[1].intercept_[0])
print('-----------------------\n')
print('SVC Parameters:\n')
print('w1:', svc_pipeline[1].coef_[0][0], ', w2:', svc_pipeline[1].coef_[0][1])
print('b:',svc_pipeline[1].intercept_[0])
print('-----------------------\n')
print('SGDClassifier Parameters:\n')
print('w1:', sgd_pipeline[1].coef_[0][0], ', w2:', sgd_pipeline[1].coef_[0][1])
print('b:', sgd_pipeline[1].intercept_[0])
```



