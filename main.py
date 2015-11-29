from model.decision_tree import DecisionTreeModel
from model.naive_bayes import GaussianNaiveBayesModel
from model.svm import SupportVectorMachineModel
from model.random_forest import RandomForestModel
from model.extreme_tree import ExtremeTreeModel
from model.gradient_boosted_tree import GradientBoostedModel
from model.knn import kNNModel
from model.adaboost import AdaBoostModel

if __name__ == "__main__":
    svm = SupportVectorMachineModel()
    svm.train_model()
    print "Support Vector Machine 10 fold validation: ", svm.get_n_fold_validation_score()
    print "Support Vector Machine: " + str(svm.accuracy)

    knn =kNNModel()
    knn.train_model()
    print "k Neartest Neighbour 10 fold validation: ", knn.get_n_fold_validation_score()
    print "k Neartest Neighbour: " + str(knn.accuracy)

    nb = GaussianNaiveBayesModel()
    nb.train_model()
    print "Gaussian Naive Bayes 10 fold validation: ", nb.get_n_fold_validation_score()
    print "Gaussian Naive Bayes: " + str(nb.accuracy)

    dt = DecisionTreeModel()
    dt.train_model()
    print "Decision Tree Machine 10 fold validation: ", dt.get_n_fold_validation_score()
    print "Decision Tree: " + str(dt.accuracy)

    rf = RandomForestModel()
    rf.train_model()
    print "Random Forest 10 fold validation: ", rf.get_n_fold_validation_score()
    print "Random Forest: " + str(rf.accuracy)

    et = ExtremeTreeModel()
    et.train_model()
    print "Extreme Tree 10 fold validation: ", et.get_n_fold_validation_score()
    print "Extreme Tree: " + str(et.accuracy)

    gb = GradientBoostedModel()
    gb.train_model()
    # print gb.accuracy
    print "Gradient Boosted Tree 10 fold validation: ", gb.get_n_fold_validation_score()
    print "Gradient Boosted Tree: " + str(gb.accuracy)

    adaboost =AdaBoostModel()
    adaboost.train_model()
    print "AdaBoost 10 fold validation: ", adaboost.get_n_fold_validation_score()
    print "AdaBoost: " + str(adaboost.accuracy)