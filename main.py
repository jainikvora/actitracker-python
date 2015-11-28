from model.decision_tree import DecisionTreeModel
from model.naive_bayes import GaussianNaiveBayesModel
from model.svm import SupportVectorMachineModel
from model.random_forest import RandomForestModel
from model.extreme_tree import ExtremeTreeModel
from model.gradient_boosted_tree import GradientBoostedModel
from model.knn import kNNModel
from model.adaboost import AdaBoostModel

if __name__ == "__main__":
    # dt = DecisionTreeModel()
    # dt.train_model()
    # print "Decision Tree: " + str(dt.accuracy)
    #
    # nb = GaussianNaiveBayesModel()
    # nb.train_model()
    # print "Gaussian Naive Bayes: " + str(nb.accuracy)
    #
    # svm = SupportVectorMachineModel()
    # svm.train_model()
    # print "Support Vector Machine: " + str(svm.accuracy)

    rf = RandomForestModel()
    rf.train_model()
    print "Random Forest: " + str(rf.accuracy)

    # et = ExtremeTreeModel()
    # et.train_model()
    # print "Extreme Tree: " + str(et.accuracy)
    #
    # gb = GradientBoostedModel()
    # gb.train_model()
    # print "Gradient Boosted Tree: " + str(gb.accuracy)
    #
    # knn =kNNModel()
    # knn.train_model()
    # print "k Neartest Neighbour: " + str(knn.accuracy)
    #
    # adaboost =AdaBoostModel()
    # adaboost.train_model()
    # print "AdaBoost: " + str(adaboost.accuracy)