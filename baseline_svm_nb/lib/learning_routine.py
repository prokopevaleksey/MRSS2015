import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn import feature_selection
from sklearn.ensemble import RandomForestClassifier,  ExtraTreesClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import Isomap
from sklearn import cross_validation
from sklearn import pipeline
import libscores
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import pdb

def apply_cross_validation(X_train, Y_train, clf, clf_call, which):
    kfold = cross_validation.KFold(len(X_train), n_folds=which['n_folds'])
    
    cross_vals = []
    for train, test in kfold:       
        XX = eval('clf.' + clf_call)
        YY = Y_train[test]
        [cXX, cYY] = libscores.normalize_array(XX, YY)
        if which['metric'] == 'bac_metric':
            cur = libscores.bac_metric(cXX[np.newaxis, :], cYY[np.newaxis, :])
        else:
            cur = libscores.auc_metric(cXX[np.newaxis, :], cYY[np.newaxis, :])
        cross_vals.append(cur)
    return np.mean(cross_vals)
    
def default_prediction(X_train, Y_train, X_valid, X_test, params):            
    best_cycle = 0
    best_mean = -100
    feat_num = X_train.shape[1]
    #TODO TIME, fusion, select num of estimators
    if not params['is_sparse']:
        ##################################
        # FEATURE SELECTION
        ##################################      
        for cycle in xrange(14):
            begin = time.time() 
            n_estimators = 33  
            
            if cycle > 0:
                if cycle == 1:
                    fi = M.feature_importances_
                    fis = sorted(fi)

                th = fis[-feat_num*(100-cycle*7)/100 - 1]
                which = fi > th
            else:
                which = np.ones((feat_num,), dtype=bool) # Take all features

            clf = RandomForestClassifier(n_estimators, random_state=1, n_jobs=params['n_jobs'], min_samples_leaf = 3 , min_samples_split = 6)
            
            score_mean = apply_cross_validation(X_train[:, which], Y_train, 
                clf, 'fit(X_train[train], Y_train[train]).predict_proba(X_train[test])[:,1]',
                params)
                            
            if cycle == 0:    
                M = clf.fit(X_train, Y_train)
                
            if score_mean > best_mean:
                best_mean = score_mean
                best_cycle = cycle
        
        # Choose how many features to keep
        if best_cycle > 0:
            th = fis[-feat_num*(100-best_cycle*7 )/100 - 1]
            which = fi > th

        else:
            which = np.ones((feat_num,), dtype=bool) # Take all features
        
        ##################################
        # TRAINING CLASSIFIERS
        ################################## 
        clf = RandomForestClassifier(333, random_state=1, n_jobs=params['n_jobs'], 
              min_samples_leaf = 1 , min_samples_split = 3)
        M = clf.fit(X_train[:, which], Y_train)
        
    else:
        ##################################
        # FEATURE SELECTION
        ##################################  
        for cycle in xrange(14):
            begin = time.time() 
            n_estimators = 33  
            
            if cycle > 0:
                if cycle == 1:
                    #normalize to [0,1]
                    scale = MinMaxScaler()
                    X_train = scale.fit_transform(X_train)  
                    #take chi2 statistics of each feature 
                    fi, _ = feature_selection.chi2(X_train,Y_train) #may cause problem assigning to that, but it's used in stackoverflow solution
                    fis = sorted(fi)

                th = fis[-feat_num*(100-cycle*7)/100 - 1]
                which = fi > th
            else:
                which = np.ones((feat_num,), dtype=bool) # Take all features

            clf = BaggingClassifier(base_estimator=BernoulliNB(), n_estimators=33, n_jobs=params['n_jobs'])#shit load of stuff to tune, but fuck it for now
            
            score_mean = apply_cross_validation(X_train[:, which], Y_train, 
                clf, 'fit(X_train[train], Y_train[train]).predict_proba(X_train[test])[:,1]',
                params)
                            
            if cycle == 0:    
                M = clf.fit(X_train, Y_train)
                
            if score_mean > best_mean:
                best_mean = score_mean
                best_cycle = cycle
        
        # Choose how many features to keep
        if best_cycle > 0:
            th = fis[-feat_num*(100-best_cycle*7 )/100 - 1]
            which = fi > th

        else:
            which = np.ones((feat_num,), dtype=bool) # Take all features
        ##################################
        # TRAINING CLASSIFIERS
        ################################## 
        clf = BaggingClassifier(base_estimator=BernoulliNB(), n_estimators=300, n_jobs=params['n_jobs']) #shit load of stuff to tune, but fuck it for now
        M = clf.fit(X_train[:, which], Y_train)


    ##################################
    # CLASSIFICATION
    ################################## 
    # Make predictions
    Y_valid = M.predict_proba(X_valid[:, which])[:, 1]
    Y_test =  M.predict_proba(X_test[:, which])[:, 1]
    return Y_valid, Y_test 

def process_christine(Xtrain, ytrain, Xval, Xtest, params):
    print 'ITS A CHRISTINE TIME !!!'
    print
    
    which = np.array([False, True, False, False, False, False, False, False, False, True, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, True, False, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, False, True, True, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, True, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, False, True, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, True, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, True, True, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    
    t0 = time.time()
             
    modelrf = RandomForestClassifier(n_estimators = 2000, criterion = 'entropy', n_jobs = params['n_jobs'])
    modelrf.fit(Xtrain[:, which], ytrain)
    
    print 'RF DONE'
    print (time.time() - t0) / 60.
             
    ytestrf = modelrf.predict_proba(Xtest[:, which])[:, 1]
    yvalrf = modelrf.predict_proba(Xval[:, which])[:, 1]
             
    modelknn = KNeighborsClassifier(n_neighbors = 6, weights = 'distance', metric = 'braycurtis')
    modelknn.fit(Xtrain[:, which], ytrain)

    print 'KNN DONE'
    print (time.time() - t0) / 60.
             
    ytestknn = modelknn.predict_proba(Xtest[:, which])[:, 1]
    yvalknn = modelknn.predict_proba(Xval[:, which])[:, 1]
    
    ytestfinal = np.round(0.8 * ytestrf + 0.2 * ytestknn)
    yvalfinal = np.round(0.8 * yvalrf + 0.2 * yvalknn)
             
    return yvalfinal, ytestfinal

def process_my_christine(Xtrain, ytrain, Xval, Xtest, params):
    print 'ITS A MY CHRISTINE TIME !!!'
    
    t0 = time.time()
    
    modelrf = pipeline.Pipeline([
            ('feature_selection', feature_selection.SelectPercentile(percentile=30, score_func=feature_selection.f_classif)),
            ('classification', RandomForestClassifier(n_estimators=200, random_state=1, n_jobs=params['n_jobs']))
        ])
    
    modelrf.fit(Xtrain, ytrain)
    
    print 'RF DONE'
    print (time.time() - t0) / 60.
             
    ytestrf = modelrf.predict_proba(Xtest)[:, 1]
    yvalrf = modelrf.predict_proba(Xval)[:, 1]
    
    ytestfinal = ytestrf
    yvalfinal = yvalrf
             
    return yvalfinal, ytestfinal

def process_jasmine(X_train, Y, X_valid, X_test, params):
    which = np.array([False, False, False, False, False,  True, False, False, False,
       False,  True, False,  True, False, False, False, False, False,
       False, False,  True, False,  True, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False,  True,  True,  True, False, False,  True, False,  True,
       False, False, False, False, False, False, False, False, False,
       False,  True,  True, False,  True, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False,  True, False, False, False,
        True, False, False, False, False, False, False, False, False,
       False,  True, False, False, False, False, False, False, False,
        True, False,  True, False, False, False, False, False,  True,
        True, False,  True,  True, False, False, False, False, False,
       False, False, False, False, False,  True, False, False,  True,
       False, False, False, False,  True,  True, False,  True, False,
       False, False, False, False, False, False, False, False, False])
    
    # RF
    param = {'n_estimators': 5000,
             'criterion' : 'gini',
             'max_depth': None,
             'max_features' : 'auto',
             'min_samples_leaf' : 1,
             'n_jobs': params['n_jobs'],
    }
    
    modelrf = RandomForestClassifier(**param)
    print clf
    modelrf.fit(X_train[:,which],Y)
    
    pr_rf_train = modelrf.predict_proba(X_train[:,which])[:, 1]
    pr_rf_val = modelrf.predict_proba(X_valid[:,which])[:, 1]
    pr_rf_test = modelrf.predict_proba(X_test[:,which])[:, 1]
    
    X_train_norm = preprocessing.scale(X_train, axis=1, with_mean=True, with_std=True, copy=True)
    X_valid_norm = preprocessing.scale(X_valid, axis=1, with_mean=True, with_std=True, copy=True)
    X_test_norm = preprocessing.scale(X_test, axis=1, with_mean=True, with_std=True, copy=True)
    
    modelsvm = svm.SVC(probability=True)
    modelsvm.fit(X_train_norm[:, which], Y_train)

    pr_svm_train = modelsvm.predict_proba(X_train_norm[:,which])[:, 1]
    pr_svm_val = modelsvm.predict_proba(X_valid_norm[:,which])[:, 1]
    pr_svm_test = modelsvm.predict_proba(X_test_norm[:,which])[:, 1]
    
    # Stack both predictions
    pr_stack_train = np.vstack((pr_rf_train, pr_svm_train)).T
    pr_stack_val = np.vstack((pr_rf_val, pr_svm_val)).T
    pr_stack_test = np.vstack((pr_rf_test, pr_svm_test)).T
    
    modellr = LogisticRegression()
    
    modellr.fit(pr_stack_train, Y_train)
    
    pr_val = modellr.predict_proba(pr_stack_val)[:, 1]
    pr_test = modellr.predict_proba(pr_stack_test)[:, 1]
    
    return pr_val, pr_test


def process_madeline(X_train, Y_train, X_valid, X_test, params):
    which = np.array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False,  True, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False,  True, False, False, False, False, False, False,
       False, False,  True, False,  True, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
        True, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
        True, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False,  True,  True,  True, False, False, False,
       False, False,  True, False, False, False, False, False,  True,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False,  True, False,
       False, False, False,  True, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False])
    
    # Random forest
    modelrf = RandomForestClassifier(n_estimators=5000, n_jobs=params['n_jobs'], criterion='entropy', bootstrap=False)
    print modelrf
    modelrf.fit(X_train[:,which],Y_train)

    pr_rf_train = modelrf.predict_proba(X_train[:,which])[:, 1]
    pr_rf_val = modelrf.predict_proba(X_valid[:,which])[:, 1]
    pr_rf_test = modelrf.predict_proba(X_test[:,which])[:, 1]
    
    # Normalization
    X_train_norm = preprocessing.scale(X_train, axis=1, with_mean=True, with_std=True, copy=True)
    X_valid_norm = preprocessing.scale(X_valid, axis=1, with_mean=True, with_std=True, copy=True)
    X_test_norm = preprocessing.scale(X_test, axis=1, with_mean=True, with_std=True, copy=True)
    
    # SVM
    modelsvm = svm.SVC(probability=True)
    modelsvm.fit(X_train_norm[:, which], Y_train)

    pr_svm_train = modelsvm.predict_proba(X_train_norm[:,which])[:, 1]
    pr_svm_val = modelsvm.predict_proba(X_valid_norm[:,which])[:, 1]
    pr_svm_test = modelsvm.predict_proba(X_test_norm[:,which])[:, 1]
    
    # NB
    modelbnb = BernoulliNB()
    modelbnb.fit(X_train_norm[:, which], Y_train)
    pr_bnb_train = modelbnb.predict_proba(X_train_norm[:,which])[:, 1]
    pr_bnb_val = modelbnb.predict_proba(X_valid_norm[:,which])[:, 1]
    pr_bnb_test = modelbnb.predict_proba(X_test_norm[:,which])[:, 1]
    
    # Stack both predictions
    pr_stack_train = np.vstack((pr_rf_train, pr_svm_train, pr_bnb_train)).T
    pr_stack_val = np.vstack((pr_rf_val, pr_svm_val, pr_bnb_val)).T
    pr_stack_test = np.vstack((pr_rf_test, pr_svm_test, pr_bnb_test)).T
    
    modellr = LogisticRegression()
    
    modellr.fit(pr_stack_train, Y_train)
    
    pr_val = modellr.predict_proba(pr_stack_val)[:, 1]
    pr_test = modellr.predict_proba(pr_stack_test)[:, 1]
    
    return pr_val, pr_test
    
def process_madeline2(X_train, Y_train, X_valid, X_test, params):
    which = np.array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False,  True, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False,  True, False, False, False, False, False, False,
       False, False,  True, False,  True, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
        True, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
        True, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False,  True,  True,  True, False, False, False,
       False, False,  True, False, False, False, False, False,  True,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False,  True, False,
       False, False, False,  True, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False])
        
    clf = RandomForestClassifier(n_estimators=5000, n_jobs=params['n_jobs'], criterion='entropy', bootstrap=False)
    print clf
    #clf.fit(X_train[:,which],Y_train)
    
    #score_mean0 = apply_cross_validation(X_train[:, which], Y_train, 
        #clf, 'fit(X_train[train], Y_train[train]).predict_proba(X_train[test])[:,1]',
        #params)
    #pdb.set_trace()
    
    #pr_val = clf.predict(X_valid[:,which])
    #pr_test = clf.predict(X_test[:,which])
    
    X_train_norm = preprocessing.scale(X_train, axis=1, with_mean=True, with_std=True, copy=True)
    X_valid_norm = preprocessing.scale(X_valid, axis=1, with_mean=True, with_std=True, copy=True)
    X_test_norm = preprocessing.scale(X_test, axis=1, with_mean=True, with_std=True, copy=True)
        
    # Naive Bayes
    modelgnb = GaussianNB()
    score_mean1 = apply_cross_validation(X_train_norm[:, which], Y_train, 
        modelgnb, 'fit(X_train[train], Y_train[train]).predict_proba(X_train[test])[:,1]',
        params)
        
    modelbnb = BernoulliNB()
    score_mean2 = apply_cross_validation(X_train_norm[:, which], Y_train, 
        modelbnb, 'fit(X_train[train], Y_train[train]).predict_proba(X_train[test])[:,1]',
        params)
    
    
    
    modelsvm = svm.SVC(probability=True)
    score_mean3 = apply_cross_validation(X_train_norm[:, which], Y_train, 
        modelsvm, 'fit(X_train[train], Y_train[train]).predict_proba(X_train[test])[:,1]',
        params)
    
    print score_mean1, score_mean2, score_mean3
    pdb.set_trace()
        
    
    
    
    
    return pr_val, pr_test


def process_sylvine(Xtrain, ytrain, Xval, Xtest, params):
    print 'ITS A SYLVINE TIME'
    print
    
    t0 = time.time()
         
    which = np.array([False, False, False, False, False, False, True, False, True, True, False, False,
 False, False, True, True, False, False, False, True])

    Xnewtrain = np.array(Xtrain[:, which])
    Xnewtest = np.array(Xtest[:, which])
    Xnewval = np.array(Xval[:, which])

    t0 = time.time()

    iso = Isomap(n_neighbors = 20, n_components = 3).fit(Xnewtrain[:, :6])

    print 'ISOSTAS !!!'
    print (time.time() - t0) / 60.

    t0 = time.time()

    Xisotrain = iso.transform(Xnewtrain[:, :6])
    Xisotest = iso.transform(Xnewtest[:, :6])
    Xisoval = iso.transform(Xnewval[:, :6])

    print 'ISOSTAS RETURNED !!!'
    print (time.time() - t0) / 60.

    Xnewtrain = np.hstack((Xnewtrain, Xisotrain))
    Xnewtest = np.hstack((Xnewtest, Xisotest))
    Xnewval = np.hstack((Xnewval, Xisoval))
        
    modelrf = ExtraTreesClassifier(n_estimators = 10000, n_jobs = params['n_jobs'])
    modelrf.fit(Xnewtrain, ytrain)
    
    print (time.time() - t0) / 60.
             
    ytestrf = modelrf.predict_proba(Xnewtest)[:, 1]
    yvalrf = modelrf.predict_proba(Xnewval)[:, 1]


    ytestfinal = np.round(ytestrf)
    yvalfinal = np.round(yvalrf)
             
    return yvalfinal, ytestfinal


def process_philippine(X_train, Y_train, X_valid, X_test, params):
    which = np.array([False, False, False, False, False, False, False, False, False,
       False,  True, False,  True, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False,  True, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False,  True, False, False, False, False, False,
       False, False, False,  True,  True, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False,  True, False,  True,
       False, False, False,  True, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False,  True, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False,  True, False, False, False, False, False, False, False,
       False,  True, False, False,  True,  True, False, False,  True,
       False, False, False, False, False, False,  True, False, False,
       False, False, False, False, False, False, False,  True, False,
       False, False, False, False, False, False, False, False, False,
        True, False, False, False, False, False, False, False,  True,
       False,  True, False,  True,  True, False,  True, False, False,
       False, False, False, False, False, False, False, False,  True,
        True, False, False, False, False, False, False, False,  True,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False,  True,  True, False, False, False,
        True, False, False, False, False, False, False, False, False,
       False, False,  True, False, False, False, False, False,  True,
       False,  True, False, False,  True, False, False,  True, False,
       False, False, False,  True, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False], dtype=bool)
    
    param = {'n_estimators': 5000,
             'criterion' : 'entropy',
             'max_depth':None,
             'max_features' : 'auto',
             'min_samples_leaf' : 1,
         
             'n_jobs': params['n_jobs'],
    }
    
    clf = RandomForestClassifier(**param)
    print clf
    clf.fit(X_train[:,which],Y_train)
    
    pr_rf_train = clf.predict_proba(X_train[:,which])[:, 1]
    pr_rf_val = clf.predict_proba(X_valid[:,which])[:, 1]
    pr_rf_test = clf.predict_proba(X_test[:,which])[:, 1]
    
    pdb.set_trace()
    
    # KNN
    param = {'n_neighbors':5,
         'weights':'distance',
         'metric':'canberra',
        }
    
    clf = KNeighborsClassifier(**param)
    
    print clf
    clf.fit(X_train[:,which],Y_train)
    
    pr_knn_train = clf.predict_proba(X_train[:, which])[:, 1]
    pr_knn_val = clf.predict_proba(X_valid[:,which])[:, 1]
    pr_knn_test = clf.predict_proba(X_test[:,which])[:, 1]
    
    # Scale
    X_train_norm = preprocessing.scale(X_train, axis=1, with_mean=True, with_std=True, copy=True)
    X_valid_norm = preprocessing.scale(X_valid, axis=1, with_mean=True, with_std=True, copy=True)
    X_test_norm = preprocessing.scale(X_test, axis=1, with_mean=True, with_std=True, copy=True)
    
    # SVM
    modelsvm = svm.SVC(probability=True)
    modelsvm.fit(X_train_norm[:, which], Y_train)

    pr_svm_train = modelsvm.predict_proba(X_train_norm[:,which])[:, 1]
    pr_svm_val = modelsvm.predict_proba(X_valid_norm[:,which])[:, 1]
    pr_svm_test = modelsvm.predict_proba(X_test_norm[:,which])[:, 1]
    
    # NB
    modelbnb = BernoulliNB()
    modelbnb.fit(X_train_norm[:, which], Y_train)
    pr_bnb_train = modelbnb.predict_proba(X_train_norm[:,which])[:, 1]
    pr_bnb_val = modelbnb.predict_proba(X_valid_norm[:,which])[:, 1]
    pr_bnb_test = modelbnb.predict_proba(X_test_norm[:,which])[:, 1]
    
    # Stack predictions
    pr_stack_train = np.vstack((pr_rf_train, pr_svm_train, pr_knn_train, pr_bnb_train)).T
    pr_stack_val = np.vstack((pr_rf_val, pr_svm_val, pr_knn_val, pr_bnb_val)).T
    pr_stack_test = np.vstack((pr_rf_test, pr_svm_test, pr_knn_test, pr_bnb_test)).T
    
    modellr = LogisticRegression()
    
    modellr.fit(pr_stack_train, Y_train)
    
    pr_val = modellr.predict_proba(pr_stack_val)[:, 1]
    pr_test = modellr.predict_proba(pr_stack_test)[:, 1]
    
    return pr_val, pr_test
    
    
    #alpha = 0.1
    #mix =  (1-alpha)*pr_val_rf+alpha* pr_val_knn
    #pr_val =  mix[:,0] < mix[:,1]

    #alpha = 0.1
    #mix =  (1-alpha)*pr_test_rf+alpha* pr_test_knn
    #pr_test =  mix[:,0] < mix[:,1]
    
    #return pr_val, pr_test
