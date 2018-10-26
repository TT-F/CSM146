"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that always predicts the majority class.
        
        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None
    
    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self
    
    def predict(self, X) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        n,d = X.shape
        y = [self.prediction_] * n 
        return y


class RandomClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.
        
        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None
    
    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        majority_vals = Counter(y).most_common(2)
        total = majority_vals[0][1] + majority_vals[1][1]
        self.probabilities_ = {0: majority_vals[0][1] / total, 1: majority_vals[1][1] / total}

        ### ========== TODO : END ========== ###
        
        return self
    
    def predict(self, X, seed=1234) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)
        
        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        #print(self.probabilities_.values())
        y = np.random.choice(list(self.probabilities_.keys()), X.shape[0], p=list(self.probabilities_.values()))

        ### ========== TODO : END ========== ###
        
        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)  
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
 
    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))
    
    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2, train_size = 0.8) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """
    
    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    
    train_error = 0
    test_error = 0
    for i in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i, train_size=train_size)
        train_clf = clf  # create MajorityVote classifier, which includes all model parameters
        train_clf.fit(X_train, y_train)  # fit training data using the classifier
        train_y_pred = train_clf.predict(X_train)  # take the classifier and run it on the training data
        train_error += 1 - metrics.accuracy_score(y_train, train_y_pred, normalize=True)
        test_y_pred = train_clf.predict(X_test)
        test_error += 1 - metrics.accuracy_score(y_test, test_y_pred, normalize=True)

    train_error /= ntrials
    test_error /= ntrials
        
    ### ========== TODO : END ========== ###
    
    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features
    
    
    
    #========================================
    # part a: plot histograms of each feature
    # print('Plotting...')
    # for i in range(d) :
    #     plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)

       
    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    
    
    
    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    ran_clf = RandomClassifier()
    ran_clf.fit(X, y)
    ran_y_pred = ran_clf.predict(X)
    ran_train_error = 1 - metrics.accuracy_score(y, ran_y_pred, normalize=True)
    print('\t-- random classifier training error: %.3f' % ran_train_error)
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain 
    print('Classifying using Decision Tree...')
    dec_tree_clf = DecisionTreeClassifier(criterion="entropy")
    dec_tree_clf.fit(X, y)
    dec_tree_y_pred = dec_tree_clf.predict(X)
    dec_tree_train_error = 1-  metrics.accuracy_score(y, dec_tree_y_pred, normalize=True)
    print('\t-- Decision Tree classifier training error: %.3f' % dec_tree_train_error)
    
    ### ========== TODO : END ========== ###

    

    # note: uncomment out the following lines to output the Decision Tree graph
    '''
    # save the classifier -- requires GraphViz and pydot
    from io import StringIO
    import pydot
    from sklearn import tree
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf")
    '''



    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors 
    print('Classifying using k-Nearest Neighbors...')
    k_3_clf = KNeighborsClassifier(n_neighbors=3)
    k_3_clf.fit(X, y)
    k_3_y_pred = k_3_clf.predict(X)
    k_3_train_error = 1 - metrics.accuracy_score(y, k_3_y_pred, normalize=True)
    print('\t-- K-3 classifier training error: %.3f' % k_3_train_error)
    k_5_clf = KNeighborsClassifier(n_neighbors=5)
    k_5_clf.fit(X, y)
    k_5_y_pred = k_5_clf.predict(X)
    k_5_train_error = 1 - metrics.accuracy_score(y, k_5_y_pred, normalize=True)
    print('\t-- K-5 classifier training error: %.3f' % k_5_train_error)
    k_7_clf = KNeighborsClassifier(n_neighbors=7)
    k_7_clf.fit(X, y)
    k_7_y_pred = k_7_clf.predict(X)
    k_7_train_error = 1 - metrics.accuracy_score(y, k_7_y_pred, normalize=True)
    print('\t-- K-7 classifier training error: %.3f' % k_7_train_error)
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    # Majority
    m_train_error, m_test_error = error(MajorityVoteClassifier(), X, y)
    print('\t-- Majority cross-validation training error: %.3f and test error: %.3f' % (m_train_error, m_test_error))
    # Random
    r_train_error, r_test_error = error(RandomClassifier(), X, y)
    print('\t-- Random cross-validation training error: %.3f and test error: %.3f' % (r_train_error, r_test_error))
    # Decision Tree
    d_train_error, d_test_error = error(DecisionTreeClassifier(criterion="entropy"), X, y)
    print('\t-- Decision Tree cross-validation training error: %.3f and test error: %.3f' % (d_train_error, d_test_error))
    # K-5
    k_train_error, k_test_error = error(KNeighborsClassifier(n_neighbors=5), X, y)
    print('\t-- K-5 cross-validation training error: %.3f and test error: %.3f' %( k_train_error, k_test_error))

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    x_axis =[]
    y_axis =[]
    for k in range(1, 50,2):
        k_unknown_clf = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(k_unknown_clf,X,y ,cv= 10)
        print('\t-- %d-NN 10-fold cross validation error: %.3f' % (k, 1 - score.mean()))
        x_axis.append(k)
        y_axis.append(1 - score.mean())
        plt.plot(x_axis, y_axis, '-')
        plt.axis('auto')
        plt.xlabel('K')
        plt.ylabel('10-Fold Cross Validation Average Error')
        plt.title('K-NN Classifier 10-Fold Cross Validation\nFor Titanic Data')
        plt.savefig("Problem4.2-f.pdf")
        plt.clf()
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')

    x_axis = []
    train_errpr_axis = []
    test_error_axis = []

    for tree_depth in range (1, 21):
        x_axis.append(tree_depth)
        d_unknown_clf = DecisionTreeClassifier(criterion="entropy", max_depth=tree_depth)
        d_train_error, d_test_error = error(d_unknown_clf, X, y)
        print('\t-- Depth: %d. Decision Tree 80/20 cross validation training error: %.3f \t testing error: %.3f' % (
        tree_depth, d_train_error, d_test_error))
        train_errpr_axis.append(d_train_error)
        test_error_axis.append(d_test_error)
        plt.plot(x_axis, train_errpr_axis, '-', label='Training Error')
        plt.plot(x_axis, test_error_axis, '-', label='Test Error')
        plt.axis('auto')
        plt.xlabel('Max Depth')
        plt.ylabel('Average Error')
        plt.legend(loc='lower left')
        plt.title('Max-Depth Decision Tree Classifier 80/20 Cross Validation\nFor Titanic Data')
        plt.savefig("Problem4.2-g.pdf")
        plt.clf()

    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')
    x_training_data=[]
    dt_train_error=[]
    dt_test_error=[]
    kn_train_error=[]
    kn_test_error=[]
    X_90_train, X_10_test, y_90_train, y_10_test = train_test_split(X, y, test_size = 0.1, train_size=0.9)
    vs_Decision_Tree = DecisionTreeClassifier(criterion='entropy', max_depth=6)
    vs_K_Neighbors = KNeighborsClassifier(n_neighbors=7)
    for train_percent in range (1,11):
        x_training_data.append(train_percent*0.1)
        if train_percent == 10:
            train_percent = 9.9
        vs_dt_train_error = 0
        vs_dt_test_error = 0
        vs_kn_train_error = 0
        vs_kn_test_error = 0
        for i in range(10):
            X_var_train, _, y_var_train, _ = train_test_split(X_90_train, y_90_train, test_size = (1- 0.1*train_percent), train_size=0.1*train_percent)
            vs_Decision_Tree.fit(X_var_train,y_var_train)
            vs_train_y_pred = vs_Decision_Tree.predict(X_var_train)
            vs_test_y_pred = vs_Decision_Tree.predict(X_10_test)
            vs_dt_train_error += 1 - metrics.accuracy_score(y_var_train, vs_train_y_pred, normalize=True)
            vs_dt_test_error += 1 - metrics.accuracy_score(y_10_test, vs_test_y_pred, normalize=True)
            vs_K_Neighbors.fit(X_var_train,y_var_train)
            vs_kn_train_y_pred = vs_K_Neighbors.predict(X_var_train)
            vs_kn_test_y_pred = vs_K_Neighbors.predict(X_10_test)
            vs_kn_train_error += 1 - metrics.accuracy_score(y_var_train, vs_kn_train_y_pred, normalize=True)
            vs_kn_test_error += 1 - metrics.accuracy_score(y_10_test, vs_kn_test_y_pred, normalize=True)
        vs_dt_train_error /= 10
        vs_dt_test_error /= 10
        vs_kn_train_error /= 10
        vs_kn_test_error /= 10
        dt_train_error.append(vs_dt_train_error)
        dt_test_error.append(vs_dt_test_error)
        kn_test_error.append(vs_kn_test_error)
        kn_train_error.append(vs_kn_train_error)
        print('\t-- Max-Depth 6 Decision Tree %d%% learning training error: %.3f \t testing error: %.3f' % (
        train_percent * 10, vs_dt_train_error, vs_dt_test_error))
        print('\t-- 7-NN %d%% learning training error: %.3f \t testing error: %.3f' % (
        train_percent * 10, vs_kn_train_error, vs_kn_test_error))
        plt.plot(x_training_data, dt_train_error, '-', label='Decision Tree Training Error')
        plt.plot(x_training_data, dt_test_error, '-', label='Decision Tree Test Error')
        plt.plot(x_training_data, kn_train_error, '-', label='K-NN Training Error')
        plt.plot(x_training_data, kn_test_error, '-', label='K-NN Test Error')
        plt.axis('auto')
        plt.xlabel('Training Percentage')
        plt.ylabel('Average Error')
        plt.legend(loc='lower right')
        plt.title('Learning Rate of 7-NN and Max Depth 6 Decision Tree\nFor Titanic Data')
        plt.savefig("Problem4.2-h.pdf")
        plt.clf()



    ### ========== TODO : END ========== ###
    
       
    print('Done')


if __name__ == "__main__":
    main()
