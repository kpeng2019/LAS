import pandas as pd
import numpy as np
import lime.lime_tabular
import subprocess



class LIMEBAG():
    def __init__(self,clf, X_train, y_train,X_test,sensitive=None,K=1):
        self.clf=clf
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.sensitive = sensitive
        self.K=K

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

    def _check(self):
        n_fea = self.X_train.shape[1]
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=self.X_train, training_labels=self.y_train,
                                                           feature_names=self.X_train.columns,
                                                           discretizer='entropy', feature_selection='lasso_path',
                                                           mode='classification')
        counter = 0
        rank = []
        rankval = []
        for i in range(self.X_test.shape[0]):
            ins = explainer.explain_instance(data_row=pd.to_numeric(self.X_test.values[i]),
                                             predict_fn=self.clf.predict_proba, num_features=n_fea, num_samples=5000)
            ind = ins.local_exp[1]
            fair = True
            for j in range(self.K):
                if ind[j][0] in self.sensitive:
                    counter += 1
                    print("Unfair!", self.X_test.columns[ind[j][0]], ind[j][0])
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
    def explain(self):
        cnt, cache, cache2 = self._check()
        size = len(cache)
        col_len = self.X_train.shape[1]
        ranks= [[] for n in range(col_len)]
        rankvals = [[] for n in range(col_len)]
        for i in range(size):
            for j in range(len(cache[i])):
                col = cache[i][j]
                ranks[col].append(j)
                rankvals[col].append(np.abs(cache2[i][j]))
        print("Number of unfair instances:", cnt,"out of all",self.X_test.shape[0],"instances")
        cols = self.X_test.columns

        f = open("lime_rank" + ".txt", "w")
        for j in range(col_len):
            f.write(cols[j] + '\n')
            for each in ranks[j]:
                f.write("%f" % each + ' ')
            f.write('\n')
        f.close()

        f = open("lime_val" + ".txt", "w")
        for j in range(col_len):
            f.write(cols[j] + '\n')
            for each in rankvals[j]:
                f.write("%f" % each + ' ')
            f.write('\n')
        f.close()

