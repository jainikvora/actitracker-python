from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

from data.actitracker_data import ActitrackerData as data


class RandomForestModel:
    classifier = RandomForestClassifier(criterion="gini", n_estimators=10)
    accuracy = None

    def train_model(self, test_split=0.4):
        # split training and test data
        feature_train, feature_test, lable_train, lable_test = train_test_split(data.get_features(), data.get_lables()
                                                                                , test_size=test_split)
        self.classifier.fit(feature_train, lable_train['lable'].values)
        self.accuracy = self.get_accuracy(feature_test, lable_test.values)

    def get_accuracy(self, feature_test, lable_test):
        pred_lables = self.classifier.predict(feature_test)
        return accuracy_score(lable_test, pred_lables)