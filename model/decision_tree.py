from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd

from data.actitracker_data import ActitrackerData as data


class DecisionTreeModel:
    classifier = DecisionTreeClassifier(criterion="gini")
    accuracy = None

    def train_model(self, test_split=0.4):
        # split training and test data
        feature_train, feature_test, lable_train, lable_test = train_test_split(data.get_features(), data.get_lables()
                                                                                , test_size=test_split)

        self.classifier.fit(feature_train, lable_train)
        self.accuracy = self.get_accuracy(feature_test, lable_test)

    def get_accuracy(self, feature_test, lable_test):
        pred_lables = self.classifier.predict(feature_test)
        return accuracy_score(lable_test, pred_lables)

    def get_n_fold_validation_score(self, fold=10):
        features = data.get_features()
        lables = data.get_lables()
        length = len(features)
        jump = length/fold
        index = 0
        k = 0
        scores = list()
        while k < fold:
            feature_test = features.iloc[index:(index + jump), :]
            lable_test = lables.iloc[index: (index + jump), :]
            feature_train_1, feature_train_2 = features.iloc[0: index-1, :] if index != 0 else pd.DataFrame(), features.iloc[index+jump+1: length-1]
            feature_train = pd.concat([feature_train_1, feature_train_2])
            lable_train_1, lable_train_2 = lables.iloc[0: index-1, :] if index != 0 else pd.DataFrame(), lables.iloc[index+jump+1: length-1]
            lable_train = pd.concat([lable_train_1, lable_train_2])
            index += jump
            k += 1
            classifier = DecisionTreeClassifier(criterion="gini")
            classifier.fit(feature_train, lable_train)
            scores.append(accuracy_score(lable_test, classifier.predict(feature_test)))
        return sum(scores)/float(len(scores))