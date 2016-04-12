# from sklearn import neighbors
import pandas as pd
import pylab as pl


def logistic(df_train,df_test,target_col):
    columns = df_train.columns
    #import the class
    from sklearn.linear_model import LogisticRegression

    #instantiate
    logreg = LogisticRegression()

    # fit 
    x_train = df_train.copy()
    y_train = x_train[columns[target_col]]
    del x_train[columns[target_col]]
    logreg.fit(x_train, y_train)

    #predict
    x_new = df_test.copy()
    del x_new[columns[target_col]]
    return logreg.predict(x_new)












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





