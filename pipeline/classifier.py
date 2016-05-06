import pandas as pd
import pylab as pl
from pylab import figure, axes, pie, title, show
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler

class Model:
    '''Builds a Model class'''
    def __init__(self, model, x_train, y_train, x_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.runs = self.clfs_run()

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "Model {}".format(self.model)

    def clf_run(self):
        runs = {}
        clfs, params = self.set_params()

        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            print models_to_run[index]
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    print clf
                    y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                    #threshold = np.sort(y_pred_probs)[::-1][int(.05*len(y_pred_probs))]
                    #print threshold
                    print precision_at_k(y_test,y_pred_probs,.05)
                    #plot_precision_recall_n(y_test,y_pred_probs,clf)
                except IndexError, e:
                    print 'Error:',e
                    continue

    def set_params(self):
        '''Params Borrowed from: https://github.com/rayidghani/magicloops/blob/master/magicloops.py'''

        clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
            'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
            'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
            'LR': LogisticRegression(penalty='l1', C=1e5),
            'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
            'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
            'NB': GaussianNB(),
            'DT': DecisionTreeClassifier(),
            'SGD': SGDClassifier(loss="hinge", penalty="l2"),
            'KNN': KNeighborsClassifier(n_neighbors=3)
                }

        grid = {
        'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
        'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
        'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
        'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
        'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
        'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
        'NB' : {},
        'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
        'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
        'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
               }
        return  clfs, grid


    # def logistic(x_train, y_train, x_new):
    #
    #     #import the class
    #     from sklearn.linear_model import LogisticRegression
    #
    #     #instantiate
    #     logreg = LogisticRegression()
    #
    #     # fit
    #     logreg.fit(x_train, y_train)
    #
    #     #predict
    #     return logreg.predict(x_new)












# def nearest_neighbor(train_df, test_df, target_col, n_neighbors):
#     #import model
#     from sklearn.neighbors import KNeighborsClassifier
#     #instantiate model
#     knn = KNeighborsClassifier(n_neighbors = n_neighbors)
#     #fit data
#     y = df_train[df_train.columns[target_col]].as_matrix()
#     del df_train[df_train.columns[target_col]]
#     x = df_train.as_matrix()
#     knn.fit(x,y)

#     #predict
#     del df_test[df_test.columns[target_col]]
#     x_new = df_test.as_matrix()
#     return knn.predict(x_new)



    # df.dropna()

    # df_train = df.iloc[0: int(0.8 *len(df)),]
    # df_test = df.iloc[int(0.8 *len(df)):int(len(df)),]

    # target = df_train[df_train.columns[target_col]].as_matrix()
    # del df_train[df_train.columns[target_col]]
    # data = df_train.as_matrix()

    # knn = neighbors.KNeighborsClassifier()
    # knn.fit(data, target)
    # return
