import pandas as pd
import numpy as np
import lime.lime_tabular
import subprocess

def check(clf,X_train, y_train,X_test,sensitive=None,K=1):
    """
    clf: fitted scikit-learn classifier
        according to LIME's doc, clf should have (probability=True) enabled
    X_train: Dataframe
        training data used to build LIME explainer
    y_train: Dataframe or array
        labels of training data
    X_test: Dataframe
        testing data for fairness analysis
    sensitive: 1-D array
        a list of indices of sensitive attribute in X_train
    K: int
       For each instance in X_test's explanation, if any sensitive feature is among the top-K features, send an alert
    """

    n_fea = X_train.shape[1]
    explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train, training_labels=y_train,
                                                       feature_names=X_train.columns,
                                                       discretizer='entropy', feature_selection='lasso_path',
                                                       mode='classification')
    counter = 0
    rank = []
    rankval = []
    for i in range(X_test.shape[0]):
        ins = explainer.explain_instance(data_row=pd.to_numeric(X_test.values[i]),
                                         predict_fn=clf.predict_proba, num_features=n_fea, num_samples=5000)
        ind = ins.local_exp[1]
        fair = True
        for j in range(K):
            if ind[j][0] in sensitive:
                counter += 1
                print("Unfair!", X_test.columns[ind[j][0]], ind[j][0])
                print('   ', [each[0] for each in ind])
                fair = False
                break
        if fair:
            print('Fair!')
            print('   ', [each[0] for each in ind])
        temp = [each[0] for each in ind]
        temp2 = [each[1] for each in ind]
        rank.append(temp)
        rankval.append(temp2)
    return counter, rank, rankval


def LIMEBAG(clf,X_train, y_train,X_test,sensitive=None,K=1):
    """
    clf: fitted scikit-learn classifier
        according to LIME's doc, clf should have (probability=True) enabled
    X_train: Dataframe
        training data used to build LIME explainer
    y_train: Dataframe or array
        labels of training data
    X_test: Dataframe
        testing data for fairness analysis
    sensitive: 1-D array
        a list of indices of sensitive attribute in X_train
    K: int
       For each instance in X_test's explanation, if any sensitive feature is among the top-K features, send an alert
    """

    cnt, cache, cache2 = check(clf,X_train, y_train,X_test,sensitive=sensitive,K=K)
    size = len(cache)
    ranks= [[] for n in range(14)]
    rankvals = [[] for n in range(14)]
    for i in range(size):
        for j in range(len(cache[i])):
            col = cache[i][j]
            ranks[col].append(j)
            rankvals[col].append(np.abs(cache2[i][j]))
    print("Number of unfair instances:", cnt,"out of all",X_test.shape[0],"instances")
    cols = X_test.columns

    f = open("lime_rank" + ".txt", "w")
    for j in range(14):
        f.write(cols[j] + '\n')
        for each in ranks[j]:
            f.write("%f" % each + ' ')
        f.write('\n')
    f.close()

    f = open("lime_val" + ".txt", "w")
    for j in range(14):
        f.write(cols[j] + '\n')
        for each in rankvals[j]:
            f.write("%f" % each + ' ')
        f.write('\n')
    f.close()
    return
