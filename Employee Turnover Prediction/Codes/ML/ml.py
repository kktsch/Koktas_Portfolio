import pandas as pd
import numpy as np

from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import recall_score, precision_score, accuracy_score


import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def display_scores(scores):
  print("Mean:", scores.mean())
  print("Standard deviation:", scores.std())


data = pd.read_csv("../data/employee_data.csv", index_col="EmployeeID")

data = data.drop(['EmployeeCount', 'StandardHours'], axis=1)
data = data.dropna().reset_index(drop=True)

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(data, data["Attrition"]):
 train_set = data.loc[train_index]
 test_set = data.loc[test_index]

# Prepare
data = train_set.copy()
data['Attrition'] = data['Attrition'].replace(to_replace=['No', 'Yes'], value=[0, 1])
data = train_set.drop("Attrition", axis=1)
data_labels = train_set["Attrition"].copy()

data["JobInvolvement-Performance"] = data["JobInvolvement"]*data["PerformanceRating"]
data = data.drop(['JobInvolvement','PerformanceRating'], axis=1)
data["Satisfaction"] = data["JobSatisfaction"]*data["EnvironmentSatisfaction"]*data["WorkLifeBalance"]
data = data.drop(['JobSatisfaction','EnvironmentSatisfaction','WorkLifeBalance'], axis=1)


# Transform
data_num = data.drop(['BusinessTravel', 'Department', 'EducationField', 'Gender', 'DistanceFromHome', 'JobLevel',
       'JobRole', 'MaritalStatus', 'Over18', 'Education','StockOptionLevel','JobInvolvement-Performance','Satisfaction'], axis=1)
data_num_tr = StandardScaler().fit_transform(data_num)

cat_attribs = data[['BusinessTravel', 'Department', 'EducationField', 'Gender',
       'JobRole', 'MaritalStatus', 'Over18']]
data_cat_1hot = pd.get_dummies(cat_attribs, prefix_sep="_", drop_first=True)

label_attribs = data[['DistanceFromHome', 'Education','JobLevel','StockOptionLevel',
                 'JobInvolvement-Performance','Satisfaction']]
data_cat_label = label_attribs.apply(LabelEncoder().fit_transform)

data_prepared = np.concatenate((data_num_tr,data_cat_1hot,data_cat_label),axis=1)
data_labels = LabelEncoder().fit_transform(data_labels)


# Test algorithms

## Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(data_prepared, data_labels)

scores_accuracy = cross_val_score(log_reg, data_prepared, data_labels,
                         scoring="accuracy", cv=5)
scores_precision = cross_val_score(log_reg, data_prepared, data_labels,
                         scoring="precision", cv=5)
scores_recall = cross_val_score(log_reg, data_prepared, data_labels,
                         scoring="recall", cv=5)

print("Logisctic Regression")
print("Accuracy:")
display_scores(scores_accuracy)
print("Precision:")
display_scores(scores_precision)
print("Recall:")
display_scores(scores_recall)

## Decision Tree
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(data_prepared, data_labels)

scores_accuracy = cross_val_score(tree_clf, data_prepared, data_labels,
                         scoring="accuracy", cv=5)
scores_precision = cross_val_score(tree_clf, data_prepared, data_labels,
                         scoring="precision", cv=5)
scores_recall = cross_val_score(tree_clf, data_prepared, data_labels,
                         scoring="recall", cv=5)

print("Decision Tree")
print("Accuracy:")
display_scores(scores_accuracy)
print("Precision:")
display_scores(scores_precision)
print("Recall:")
display_scores(scores_recall)

## Random Forest
rnd_clf = RandomForestClassifier(random_state=42)
rnd_clf.fit(data_prepared, data_labels)

scores_accuracy = cross_val_score(rnd_clf, data_prepared, data_labels,
                         scoring="accuracy", cv=5)
scores_precision = cross_val_score(rnd_clf, data_prepared, data_labels,
                         scoring="precision", cv=5)
scores_recall = cross_val_score(rnd_clf, data_prepared, data_labels,
                         scoring="recall", cv=5)

print("Random Forest")
print("Accuracy:")
display_scores(scores_accuracy)
print("Precision:")
display_scores(scores_precision)
print("Recall:")
display_scores(scores_recall)

## Grid Search for Decision Tree
param_grid = {
    'criterion': ['gini','entropy'],
    'max_depth': [1,3,5,8,10,None],
    }

tree_clf = DecisionTreeClassifier(random_state=42)

grid_search = GridSearchCV(tree_clf, param_grid, cv=5,
                           scoring='recall',
                           return_train_score=True)

grid_search.fit(data_prepared, data_labels)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
  print(mean_score, params)
print("-------------------------------------")
print('Best model:', grid_search.best_params_)

## Decision Tree with kNN
neigh = NearestNeighbors()
neigh.fit(data_prepared)
y_knn = neigh.kneighbors(data_prepared, return_distance=False)
data_prepared_knn = np.concatenate((data_prepared,y_knn),axis=1)

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(data_prepared, data_labels)

scores_accuracy = cross_val_score(tree_clf, data_prepared_knn, data_labels,
                         scoring="accuracy", cv=5)
scores_precision = cross_val_score(tree_clf, data_prepared_knn, data_labels,
                         scoring="precision", cv=5)
scores_recall = cross_val_score(tree_clf, data_prepared_knn, data_labels,
                         scoring="recall", cv=5)

print("kNN-Decision Tree")
print("Accuracy:")
display_scores(scores_accuracy)
print("Precision:")
display_scores(scores_precision)
print("Recall:")
display_scores(scores_recall)

# Train the whole data with the decision tree
final_model = DecisionTreeClassifier().fit(data_prepared)

# Test
y_pred = final_model.predict(X_test_prepared)

print("Test Results:")
print("Recall: ", recall_score(y_test_labels, y_pred))
print("Precision: ", precision_score(y_test_labels, y_pred))
print("Accuracy: ", accuracy_score(y_test_labels, y_pred))

## Confidence interval for the test scores
confidence = 0.95

print("Recall:")
idx = y_test_labels==1
errors = 1 - abs(y_pred[idx] - y_test_labels[idx])
print(stats.t.interval(confidence, len(errors) - 1, loc=errors.mean(), scale=stats.sem(errors)))

print("Precision:")
idx = y_pred==1
errors = 1 - abs(y_pred[idx] - y_test_labels[idx])
print(stats.t.interval(confidence, len(errors) - 1, loc=errors.mean(), scale=stats.sem(errors)))

print("Accuracy:")
errors = 1 - abs(y_pred - y_test_labels)
print(stats.t.interval(confidence, len(errors) - 1, loc=errors.mean(), scale=stats.sem(errors)))

# Save model
import joblib
joblib.dump(final_model, "10-11-2022-AttritionML-FinalModel.pkl")
