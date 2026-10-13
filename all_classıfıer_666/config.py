from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

DEFAULT_MODELS = [
    ('Logistic Regression', LogisticRegression()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('SVM', SVC()),
    ('KNN', KNeighborsClassifier()),
    ('Naive Bayes', GaussianNB()),
    ('Gradient Boosting', GradientBoostingClassifier()),
    ('AdaBoost', AdaBoostClassifier()),
    ('SGD', SGDClassifier()),
    ('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),
    ('Quadratic Discriminant Analysis', QuadraticDiscriminantAnalysis()),
    ('MLP Neural Network', MLPClassifier()),
    ('LightGBM', LGBMClassifier(verbose=-1)),
    ('XGBoost', XGBClassifier()),
  ]

DEFAULT_PARAMETERS = {
    'Logistic Regression': {'C': [0.1, 1.0, 10.0]},
    'Decision Tree': {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
    'Random Forest': {'n_estimators': [50, 100], 'max_depth': [None, 10]},
    'SVM': {'C': [0.1, 1.0, 10.0],'kernel':['linear','rbf']},
    'KNN':{'n_neighbors': [3,5,7],'weights': ['uniform', 'distance']}
}

