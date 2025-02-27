FINGERPRINT = "Top25" # Top5, Top10, Top15, Top20, Top25

import pandas as pd
import scipy
import sklearn
import skmultilearn
from sklearn.metrics import (accuracy_score, precision_score, f1_score, recall_score, hamming_loss, zero_one_loss)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.problem_transform import ClassifierChain
print("...Complete")

### Here you are expected to load the dataset, then doing train/test split 80:20 using a fixed seed
### Consequently, the train and test tables must be connected with the rest of the workflow

print("Loading datasets...")
df = pd.read_csv("{0}_Fingerprint/{0}.csv.gz".format(FINGERPRINT), compression='gzip')

#X = df[df.columns[list(df.columns).index('bitvector0'):]] 
#y = df[df.columns[:list(df.columns).index('bitvector0')]] 
X = df.filter(like='bit', axis=1)  #select rows containing 'bit' 
y = df[df.columns.drop(list(df.filter(regex='bit|row')))] #remove columns containing 'bit' and RowID
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("...Complete")

def set_problem_transformation(clf, method):
    """
    ====================================================================================
                                  Set Problem Transformation
    ====================================================================================
    Key arguments:
        clf     =  scikit-learn classifier (e.g. RandomForestClassifier())
        method  =  scikit-multilearn problem transformation method among the ones below:
        
                   'BR' = BinaryRelevance()
                   'LP' = LabelPowerset()
                   'CC' = ClassifierChain()
    ====================================================================================
    """
    # define methods
    methods = {'BR' : BinaryRelevance(classifier=clf, require_dense=[True,True]),
               'LP' : LabelPowerset(classifier=clf, require_dense=[True,True]),
               'CC' : ClassifierChain(classifier=clf, require_dense=[True,True])}
    
    # set the problem transformation
    if method in methods.keys():
        pt_clf = methods[method]
        return pt_clf
    else:
        return str(method)+" is not contained among the possible methods (Try to use 'BR', 'LP', or 'CC')"
    
def metrics_problem_transformation(pt_clf, X_train, y_train, X_test, y_test):
    """
    ====================================================================================
                                  Test Problem Transformation
    ====================================================================================
    Key arguments:
        pt_clf   =  scikit-multilearn problem transformation classifier
        X_train  =  pandas dataframe containing the training set values
        y_train  =  pandas dataframe containing the training set labels
        X_test   =  pandas dataframe containing the test set values
        y_test   =  pandas dataframe containing the test set labels
    ====================================================================================
    """    
    # load modules
    import pandas as pd
    from sklearn.metrics import (accuracy_score, precision_score, f1_score, recall_score, hamming_loss, zero_one_loss)
    
    # train
    print("Fitting the function...")
    pt_clf.fit(X_train, y_train)
    print("...Complete")
    
    # predict and convert
    print("Predicting the entries...")
    y_pred = pt_clf.predict(X_test)
    print("...Complete\n")
    y_pred = y_pred.todense(order=None, out=None)
    y_pred = pd.DataFrame(y_pred, index=y_test.index.values, columns=y_test.columns.values)
    
    # metrics
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    hloss = sklearn.metrics.hamming_loss(y_test, y_pred)
    zero_one_loss = sklearn.metrics.zero_one_loss(y_test, y_pred)
    micro_recall = sklearn.metrics.recall_score(y_test, y_pred, average='micro')
    weighted_recall = sklearn.metrics.recall_score(y_test, y_pred, average='weighted')
    micro_precision = sklearn.metrics.precision_score(y_test, y_pred, average='micro')
    weighted_precision = sklearn.metrics.precision_score(y_test, y_pred, average='weighted')
    micro_f1 = sklearn.metrics.f1_score(y_test, y_pred, average='micro')
    weighted_f1 = sklearn.metrics.f1_score(y_test, y_pred, average='weighted')
    
    # list metrics and round them
    metrics = [accuracy, hloss, zero_one_loss, micro_recall, weighted_recall,
              micro_precision, weighted_precision, micro_f1, weighted_f1]
    rnd_metrics =  [round(x,2) for x in metrics]
    
    # return metrics
    print (pt_clf)
    print (rnd_metrics)
    print ()
    return metrics

print("Defining the classifiers...")
# define rf classifier
rf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, 
                            min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                            max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
                            min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=4, 
                            random_state=11, verbose=0, warm_start=False, class_weight=None)

# define svm classifier
svm = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, 
                C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, 
                class_weight=None, verbose=0, random_state=11, max_iter=1000)
print("...Complete")

methods_to_test = ['BR', 'CC', 'LP']
classifiers_to_test = [rf, svm]
error = []
for c in classifiers_to_test:
    for m in methods_to_test:
        try:
            pt_clf = set_problem_transformation(c, m)
            metrics = metrics_problem_transformation(pt_clf, X_train, y_train, X_test, y_test)
            # append metrics into file
            f = open("{0}_Fingerprint/{0}-metrics.csv".format(FINGERPRINT), "a+")
            fr = open("{0}_Fingerprint/{0}-metrics.csv".format(FINGERPRINT), "r")
            if(sum(1 for line in fr)==0):
                f.write("method,classifier,accuracy, hloss, zero_one_loss, micro_recall, weighted_recall, micro_precision, weighted_precision, micro_f1, weighted_f1"+"\n")
            fr.close()
            f.write(','.join([m,str(c).split('(')[0],','.join(map(str, [round(x,2) for x in metrics]))])+"\n")
            f.close()
        except Exception as e:
            error.append(e)