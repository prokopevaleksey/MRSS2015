import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
import libscores
import pdb

def default_prediction(X_train, y_train, X_valid, X_test):
	y_valid = np.zeros(X_valid.shape[0])
	y_test  = np.zeros(X_valid.shape[0]) 
	return y_valid, y_test 


def process_christine(Xtrain, ytrain, Xval, Xtest, global_params):
    print 'ITS A CHRISTINE TIME !!!'
    print
    
    goods = np.array([False, True, False, False, False, False, False, False, False, True, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, True, False, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, False, True, True, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, True, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, False, True, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, True, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, True, True, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    
    t0 = time.time()
             
    modelrf = RandomForestClassifier(n_estimators = 2000, criterion = 'entropy', n_jobs = global_params['n_jobs'])
    
    #
    #
    # Cross-validation example
    #scores = apply_cross_validation(X_train=Xtrain[:, goods], Y_train=ytrain, 
        #n_folds=global_params['n_folds'], clf=modelrf, 
        #clf_call='clf.fit(X_train[train], Y_train[train]).predict_proba(X_train[test])[:,1]')
    #
    #
    #
    
    modelrf.fit(Xtrain[:, goods], ytrain)    
    
    print 'RF DONE'
    print (time.time() - t0) / 60.
             
    ytestrf = modelrf.predict_proba(Xtest[:, goods])[:, 1]
    yvalrf = modelrf.predict_proba(Xval[:, goods])[:, 1]
             
    modelknn = KNeighborsClassifier(n_neighbors = 6, weights = 'distance', metric = 'braycurtis')
    modelknn.fit(Xtrain[:, goods], ytrain)

    print 'KNN DONE'
    print (time.time() - t0) / 60.
             
    ytestknn = modelknn.predict_proba(Xtest[:, goods])[:, 1]
    yvalknn = modelknn.predict_proba(Xval[:, goods])[:, 1]
    
    ytestfinal = np.round(0.85 * ytestrf + 0.15 * ytestknn)
    yvalfinal = np.round(0.85 * yvalrf + 0.15 * yvalknn)
             
    return yvalfinal, ytestfinal


def process_jasmine(X_train, Y, X_valid, X_test, global_params):
    feats_to_use = np.array([False, False, False, False, False,  True, False, False, False,
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
    
    param = {'n_estimators': 1000,
             'criterion' : 'gini',
             'max_depth':None,
             'max_features' : 'auto',
             'min_samples_leaf' : 1,
             'n_jobs':global_params['n_jobs'],
    }
    
    clf = RandomForestClassifier(**param)
    print clf
    clf.fit(X_train[:,feats_to_use],Y)
    
    pr_val = clf.predict(X_valid[:,feats_to_use])
    pr_test = clf.predict(X_test[:,feats_to_use])
    return pr_val, pr_test

def my_process_jasmine(X_train, Y, X_valid, X_test, global_params):
    goods = np.array([False, False, False, False, False,  True, False, False, False,
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
    
    print 'ITS A MY JASMINE TIME !!!'
    print
    
    t0 = time.time()
             
    modelrf = RandomForestClassifier(n_estimators = 500, n_jobs = global_params['n_jobs'], random_state=123)
    
    #
    #
    # Cross-validation example
    #scores = apply_cross_validation(X_train=Xtrain[:, goods], Y_train=ytrain, 
        #n_folds=global_params['n_folds'], clf=modelrf, 
        #clf_call='clf.fit(X_train[train], Y_train[train]).predict_proba(X_train[test])[:,1]')
    #
    #
    #
    
	# featselect = LinearSVC(C=0.1, penalty="l1", dual=False).fit(D.data['X_train'], D.data['Y_train']).transform
	
    modelrf.fit(Xtrain[:, goods], ytrain)    #featselect(D.data['X_train']), criterion = 'entropy'
    
    print 'RF DONE'
    print (time.time() - t0) / 60.
             
    ytestrf = modelrf.predict_proba(Xtest[:, goods])[:, 1]
    yvalrf = modelrf.predict_proba(Xval[:, goods])[:, 1]
    
	# features_percent = 30 # 10
    # vprint( verbose,  "[+] Eliminate features to %s percent" % features_percent)
    # feature_selector = SelectPercentile(f_classif, features_percent) # chi2
    # X_new = feature_selector.fit_transform(D.data['X_train'], D.data['Y_train'])
	
	modelgbc = GradientBoostingClassifier(n_estimators=100) #base_estimator=BernoulliNB(), 75% - best (30% feats, n_est==100)
    modelgbc.fit(Xtrain[:, goods], ytrain) #X_new
	
	#
    #
    # Cross-validation example
    scores = apply_cross_validation(X_train=Xtrain[:, goods], Y_train=ytrain, 
        n_folds=global_params['n_folds'], clf=modelrf, 
        clf_call='clf.fit(X_train[train], Y_train[train]).predict_proba(X_train[test])[:,1]')
    print(scores)
    #
    
	#
	print 'GBC DONE'
    print (time.time() - t0) / 60.
    
	ytestgbc = modelgbc.predict_proba(Xtest[:, goods])[:, 1]
	yvalgbc = modelgbc.predict_proba(Xval[:, goods])[:, 1]
    
    # modelknn = KNeighborsClassifier(n_neighbors = 6, weights = 'distance', metric = 'braycurtis')
    # modelknn.fit(Xtrain[:, goods], ytrain)

    # print 'KNN DONE'
    # print (time.time() - t0) / 60.
             
    # ytestknn = modelknn.predict_proba(Xtest[:, goods])[:, 1]
    # yvalknn = modelknn.predict_proba(Xval[:, goods])[:, 1]
    
	#
    #
    # Cross-validation example
    scores = apply_cross_validation(X_train=Xtrain[:, goods], Y_train=ytrain, 
        n_folds=global_params['n_folds'], clf=modelgbc, 
        clf_call='clf.fit(X_train[train], Y_train[train]).predict_proba(X_train[test])[:,1]')
    print(scores)
    #
    #
    
	print("FINAL")
	
	ytestfinal = ytestgbc # np.round(0.85 * ytestrf + 0.15 * ytestknn)
    yvalfinal = yvalgbc #np.round(0.85 * yvalrf + 0.15 * yvalknn)
	
    return yvalfinal, ytestfinal

def process_madeline(X_train, Y, X_valid, X_test, global_params):
    feats_to_use = np.array([False, False, False, False, False, False, False, False, False,
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
        
    clf = RandomForestClassifier(n_estimators=3000, n_jobs=global_params['n_jobs'], criterion='entropy', bootstrap=False)
    print clf
    clf.fit(X_train[:,feats_to_use],Y)
    
    pr_val = clf.predict(X_valid[:,feats_to_use])
    pr_test = clf.predict(X_test[:,feats_to_use])
    return pr_val, pr_test


def process_sylvine(Xtrain, ytrain, Xval, Xtest, global_params):
    print 'ITS A SYLVINE TIME'
    print
    
    t0 = time.time()
             
    modelrf = RandomForestClassifier(n_estimators = 5000, criterion = 'entropy', n_jobs = global_params['n_jobs'])
    modelrf.fit(Xtrain, ytrain)
    
    print 'RF DONE'
    print (time.time() - t0) / 60.
             
    ytestrf = modelrf.predict_proba(Xtest)[:, 1]
    yvalrf = modelrf.predict_proba(Xval)[:, 1]

    ytestfinal = np.round(ytestrf)
    yvalfinal = np.round(yvalrf)
             
    return yvalfinal, ytestfinal


def process_philippine(X_train, Y, X_valid, X_test, global_params):
    feats_to_use = np.array([False, False, False, False, False, False, False, False, False,
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
    
    param = {'n_estimators': 1000,
             'criterion' : 'entropy',
             'max_depth':None,
             'max_features' : 'auto',
             'min_samples_leaf' : 1,
         
             'n_jobs':global_params['n_jobs'],
    }
    
    clf = RandomForestClassifier(**param)
    print clf
    clf.fit(X_train[:,feats_to_use],Y)
    
    pr_val_rf = clf.predict_proba(X_valid[:,feats_to_use])
    pr_test_rf = clf.predict_proba(X_test[:,feats_to_use])
    
    
    param = {'n_neighbors':5,
         'weights':'distance',
         'metric':'canberra',
        }
    
    clf = KNeighborsClassifier(**param)
    
    print clf
    clf.fit(X_train[:,feats_to_use],Y)
    
    pr_val_knn = clf.predict_proba(X_valid[:,feats_to_use])
    pr_test_knn = clf.predict_proba(X_test[:,feats_to_use])

    
    alpha = 0.1
    mix =  (1-alpha)*pr_val_rf+alpha* pr_val_knn
    pr_val =  mix[:,0] < mix[:,1]

    alpha = 0.1
    mix =  (1-alpha)*pr_test_rf+alpha* pr_test_knn
    pr_test =  mix[:,0] < mix[:,1]
    
    return pr_val, pr_test
    
def apply_cross_validation(X_train, Y_train, n_folds, clf, clf_call):
    kfold = cross_validation.KFold(len(X_train), n_folds=n_folds)
    
    cross_vals = []
    for train, test in kfold:       
        XX = eval(clf_call)
        YY = Y_train[test]
        [cXX, cYY] = libscores.normalize_array(XX, YY)
        cur = (libscores.bac_metric(cXX[np.newaxis, :], cYY[np.newaxis, :]))
        cross_vals.append(cur)
    return np.mean(cross_vals)
