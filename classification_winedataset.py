import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import keras
import seaborn as sns
# %matplotlib inline

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

wine_data = load_wine()

W = wine_data.data
F = wine_data.feature_names
W

F

wine_df = pd.DataFrame(W,columns=F)

wine_data.target

wine_df['winetype'] = wine_data.target

wine_df.head()

wine_df.info()

wine_df[150:]

wine_df.describe()

wine_df.hist(figsize=(10,10))
plt.show()

X = wine_df.drop('winetype', axis=1)

y = wine_df.winetype

X

"""# Training & Test Split"""

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.20, random_state=10)

X_train = StandardScaler().fit_transform(X_train)

X_test = StandardScaler().fit_transform(X_test)

enc = preprocessing.LabelEncoder()
enc.fit(y_train)
Y_train = enc.transform(y_train)
Y_test = enc.transform(y_test)

"""# Neural Network """

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow
tensorflow.compat.v1.disable_eager_execution()

def create_ann_model(lr):
    model = Sequential()
    model.add(Dense(8, input_dim=13, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
    return model

ann_model = create_ann_model(0.001)

ann_model.fit(X_train, Y_train, epochs=200, batch_size=10, verbose=1)

ann_pred = np.round(ann_model.predict(X_test))

"""#Performance metrics - Neural Network"""

print("Results for neural network model")
print(accuracy_score(Y_test, ann_pred))
print(classification_report(Y_test, ann_pred))

"""#Hyper Parameters - Neural Network"""

learn_rates = [0.1, 0.01, 0.001, 0.0001]
epochs      = [100, 200, 300, 400]

training_samples = [20, 30, 40, 50, 60, 80]
test_percent     = [0.8, 0.7, 0.6, 0.5, 0.4, 0.2]

scores = []
for lr in learn_rates:
    ann_model = create_ann_model(lr)
    ann_model.fit(X_train, Y_train, epochs=200, batch_size=10)
    ann_pred = np.round(ann_model.predict(X_test)).astype(int)
    scores.append(accuracy_score(Y_test, ann_pred))

scores_tr = []
for lr in learn_rates:
    ann_model = create_ann_model(lr)
    ann_model.fit(X_train, Y_train, epochs=200, batch_size=10)
    ann_pred = np.round(ann_model.predict(X_train)).astype(int)
    scores_tr.append(accuracy_score(Y_train, ann_pred))

plt.plot(scores, label = 'test', marker='o')
plt.plot(scores_tr, label = 'train', marker='o')
plt.xticks(np.arange(len(learn_rates)), learn_rates, rotation=45)
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy SCore')
plt.title(label="Wine type detection graph", fontsize=12) 
plt.legend()
plt.show()

scores_trainingsamples_test = []
for t in test_percent:
    ann_model = create_ann_model(0.001)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=t, random_state=10)
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)
    enc = preprocessing.LabelEncoder()
    enc.fit(y_train)
    Y_train = enc.transform(y_train)
    Y_test = enc.transform(y_test)
    ann_model.fit(X_train, Y_train, epochs=200, batch_size=10)
    ann_pred = np.round(ann_model.predict(X_test)).astype(int)
    scores_trainingsamples_test.append(accuracy_score(Y_test, ann_pred))

scores_trainingsamples_train = []
for t in test_percent:
    ann_model = create_ann_model(0.001)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=t, random_state=10)
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)
    enc = preprocessing.LabelEncoder()
    enc.fit(y_train)
    Y_train = enc.transform(y_train)
    Y_test = enc.transform(y_test)
    ann_model.fit(X_train, Y_train, epochs=200, batch_size=10)
    ann_pred = np.round(ann_model.predict(X_train)).astype(int)
    scores_trainingsamples_train.append(accuracy_score(Y_train, ann_pred))

plt.plot(scores_trainingsamples_test, label = 'test', marker='o')
plt.plot(scores_trainingsamples_train, label = 'train', marker='o')
plt.xticks(np.arange(len(training_samples)), training_samples)
plt.xlabel('Training data % - Neural Network')
plt.ylabel('Accuracy SCore')
plt.title(label="Wine type detection graph", fontsize=12) 
plt.legend()
plt.show()

"""# KNN Classifier"""

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.20, random_state=10)
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)
enc = preprocessing.LabelEncoder()
enc.fit(y_train)
Y_train = enc.transform(y_train)
Y_test = enc.transform(y_test)

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, Y_train)
print("Results for K Nearest Neighbors")
knn.score(X_test, Y_test)

"""# Hyper parameters - KNN"""

neighbors = [3, 4, 5, 6, 7, 9, 10, 12, 15]
knn_scores = []
for n in neighbors:
    knn = KNeighborsClassifier(n_neighbors = n)
    knn.fit(X_train, Y_train)
    knn_scores.append(knn.score(X_test, Y_test))

knn_scores_train = []
for n in neighbors:
    knn = KNeighborsClassifier(n_neighbors = n)
    knn.fit(X_train, Y_train)
    knn_scores_train.append(knn.score(X_train, Y_train))

plt.plot(knn_scores, label = 'test', marker = 'o')
plt.plot(knn_scores_train, label = 'train', marker='o')
plt.xticks(np.arange(len(neighbors)), neighbors, rotation=45)
plt.xlabel('Nearest Neighbors')
plt.ylabel('Accuracy Score')
plt.title(label="Wine type detection graph", fontsize=12) 
plt.legend()
plt.show()

scores_trainingsamples_train_knn = []
for t in test_percent:
    knn = KNeighborsClassifier(n_neighbors = 5)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=t, random_state=10)
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)
    enc = preprocessing.LabelEncoder()
    enc.fit(y_train)
    Y_train = enc.transform(y_train)
    Y_test = enc.transform(y_test) 
    knn.fit(X_train, Y_train)
    scores_trainingsamples_train_knn.append(knn.score(X_train, Y_train))

scores_trainingsamples_test_knn = []
for t in test_percent:
    knn = KNeighborsClassifier(n_neighbors = 5)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=t, random_state=10)
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)
    enc = preprocessing.LabelEncoder()
    enc.fit(y_train)
    Y_train = enc.transform(y_train)
    Y_test = enc.transform(y_test)
    knn.fit(X_train, Y_train)
    scores_trainingsamples_test_knn.append(knn.score(X_test, Y_test))

plt.plot(scores_trainingsamples_test_knn, label = 'test', marker = 'o')
plt.plot(scores_trainingsamples_train_knn, label = 'train', marker='o')
plt.xticks(np.arange(len(training_samples)), training_samples)
plt.xlabel('Training data % - KNN')
plt.ylabel('Accuracy Score')
plt.title(label="Wine type detection graph", fontsize=12) 
plt.legend()
plt.show()

"""# Decision Tree Classifier"""

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.20, random_state=10)
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)
enc = preprocessing.LabelEncoder()
enc.fit(y_train)
Y_train = enc.transform(y_train)
Y_test = enc.transform(y_test)

dtree_clf = DecisionTreeClassifier(criterion='gini', min_samples_split=20, max_depth=5, random_state=10) 
dtree_clf.fit(X_train,Y_train)

Y_pred = dtree_clf.predict(X_test)

print("Results for decision tree model")
print(accuracy_score(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

"""#Hyper parameters - Decision Tree"""

tree_depth_max = [3, 4, 5, 7, 9, 10, 15, 17, 20]
sample_split_min = [10, 20, 30]

scores_depth_test = []
for d in tree_depth_max:
    dtree_model = DecisionTreeClassifier(criterion='gini', min_samples_split=20, max_depth=d, random_state=10) 
    dtree_model.fit(X_train,Y_train)
    Y_pred = dtree_model.predict(X_test)
    scores_depth_test.append(accuracy_score(Y_test, Y_pred))

scores_depth_train = []
for d in tree_depth_max:
    dtree_model = DecisionTreeClassifier(criterion='gini', min_samples_split=20, max_depth=d, random_state=10) 
    dtree_model.fit(X_train,Y_train)
    Y_pred = dtree_model.predict(X_train)
    scores_depth_train.append(accuracy_score(Y_train, Y_pred))

plt.plot(scores_depth_test, label = 'test', marker='o')
plt.plot(scores_depth_train, label = 'train', marker='o')
plt.xticks(np.arange(len(tree_depth_max)), tree_depth_max, rotation=45)
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy Score')
plt.title(label="Wine type detection graph", fontsize=12) 
plt.legend()
plt.show()

scores_sample_split_test = []
for s in sample_split_min:
    dtree_model = DecisionTreeClassifier(criterion='gini', min_samples_split=s, max_depth=5, random_state=10) 
    dtree_model.fit(X_train,Y_train)
    Y_pred = dtree_model.predict(X_test)
    scores_sample_split_test.append(accuracy_score(Y_test, Y_pred))

scores_sample_split_train = []
for s in sample_split_min:
    dtree_model = DecisionTreeClassifier(criterion='gini', min_samples_split=s, max_depth=5, random_state=10) 
    dtree_model.fit(X_train,Y_train)
    Y_pred = dtree_model.predict(X_train)
    scores_sample_split_train.append(accuracy_score(Y_train, Y_pred))

plt.plot(scores_sample_split_test, label = 'test', marker='o')
plt.plot(scores_sample_split_train, label = 'train', marker='o')
plt.xticks(np.arange(len(sample_split_min)), sample_split_min, rotation=45)
plt.xlabel('Min samples split')
plt.ylabel('Accuracy Score')
plt.title(label="Wine type detection graph", fontsize=12) 
plt.legend()
plt.show()

scores_trainingsamples_train_dtree = []
for t in test_percent:
    dtree_clf = DecisionTreeClassifier(criterion='gini', min_samples_split=20, max_depth=5, random_state=10) 
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=t, random_state=10)
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)
    enc = preprocessing.LabelEncoder()
    enc.fit(y_train)
    Y_train = enc.transform(y_train)
    Y_test = enc.transform(y_test) 
    dtree_clf.fit(X_train,Y_train)
    Y_pred = dtree_clf.predict(X_train)
    scores_trainingsamples_train_dtree.append(accuracy_score(Y_train, Y_pred))

scores_trainingsamples_test_dtree = []
for t in test_percent:
    dtree_clf = DecisionTreeClassifier(criterion='gini', min_samples_split=20, max_depth=5, random_state=10) 
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=t, random_state=10)
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)
    enc = preprocessing.LabelEncoder()
    enc.fit(y_train)
    Y_train = enc.transform(y_train)
    Y_test = enc.transform(y_test) 
    dtree_clf.fit(X_train,Y_train)
    Y_pred = dtree_clf.predict(X_test)
    scores_trainingsamples_test_dtree.append(accuracy_score(Y_test, Y_pred))

plt.plot(scores_trainingsamples_test_dtree, label = 'test', marker = 'o')
plt.plot(scores_trainingsamples_train_dtree, label = 'train', marker='o')
plt.xticks(np.arange(len(training_samples)), training_samples)
plt.xlabel('Training data % - Decision Tree')
plt.ylabel('Accuracy Score')
plt.title(label="Wine type detection graph", fontsize=12) 
plt.legend()
plt.show()

"""# SVM Classifier"""

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.20, random_state=10)
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)
enc = preprocessing.LabelEncoder()
enc.fit(y_train)
Y_train = enc.transform(y_train)
Y_test = enc.transform(y_test)

svc_clf = SVC(kernel='rbf', C=10, random_state=10)

svc_clf.fit(X_train, Y_train)

Y_pred = svc_clf.predict(X_test)

print("Results for SVM model")
print(accuracy_score(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

"""# Hyper parameters - SVM"""

C = [0.01, 0.1, 1, 10]
kernel = ['rbf', 'linear']

scores_C_test = []
for c in C:
    svc_model = SVC(C = c, random_state=10) 
    svc_model.fit(X_train,Y_train)
    Y_pred = svc_model.predict(X_test)
    scores_C_test.append(accuracy_score(Y_test, Y_pred))

scores_C_train = []
for c in C:
    svc_model = SVC(C = c, random_state=10) 
    svc_model.fit(X_train,Y_train)
    Y_pred = svc_model.predict(X_train)
    scores_C_train.append(accuracy_score(Y_train, Y_pred))

plt.plot(scores_C_test, label = 'test', marker='o')
plt.plot(scores_C_train, label = 'train', marker='o')
plt.xticks(np.arange(len(C)), C)
plt.xlabel('C')
plt.ylabel('Accuracy Score')
plt.title(label="Wine type detection graph", fontsize=12) 
plt.legend()
plt.show()

scores_kernel_test = []
for k in kernel:
    svc_model = SVC(C = 10, kernel=k, random_state=10) 
    svc_model.fit(X_train,Y_train)
    Y_pred = svc_model.predict(X_test)
    scores_kernel_test.append(accuracy_score(Y_test, Y_pred))

scores_kernel_train = []
for k in kernel:
    svc_model = SVC(C = 10, kernel=k, random_state=10) 
    svc_model.fit(X_train,Y_train)
    Y_pred = svc_model.predict(X_train)
    scores_kernel_train.append(accuracy_score(Y_train, Y_pred))

plt.plot(scores_kernel_test, label = 'test', marker='o')
plt.plot(scores_kernel_train, label = 'train', marker='o')
plt.xticks(np.arange(len(kernel)), kernel)
plt.xlabel('kernel')
plt.ylabel('Accuracy Score')
plt.title(label="Wine type detection graph", fontsize=12) 
plt.legend()
plt.show()

"""# Boosting"""

from sklearn.ensemble import AdaBoostClassifier

boost_clf = AdaBoostClassifier(n_estimators=10, learning_rate=0.5)
boost_clf.fit(X_train, y_train)
Y_pred = boost_clf.predict(X_test)
print("Results for Adaboost model")
print(accuracy_score(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))