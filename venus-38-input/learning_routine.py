import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier,  ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import Isomap

def default_prediction(X_train, y_train, X_valid, X_test):
	y_valid = np.zeros(X_valid.shape[0])
	y_test  = np.zeros(X_valid.shape[0]) 
	return y_valid, y_test 


def process_christine(Xtrain, ytrain, Xval, Xtest):
    print 'ITS A CHRISTINE TIME !!!'
    print
    
    goods = np.array([False, True, False, False, False, False, False, False, False, True, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, True, False, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, False, True, True, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, True, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, True, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, False, True, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, True, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, True, True, False, False, True, False, False, False, True, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    
    t0 = time.time()
             
    modelrf = RandomForestClassifier(n_estimators = 2000, criterion = 'entropy', n_jobs = -1)
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
    
    ytestfinal = np.round(0.8 * ytestrf + 0.2 * ytestknn)
    yvalfinal = np.round(0.8 * yvalrf + 0.2 * yvalknn)
             
    return yvalfinal, ytestfinal


def process_jasmine(X_train, Y, X_valid, X_test):
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
    
    param = {'n_estimators': 5000,
             'criterion' : 'gini',
             'max_depth':None,
             'max_features' : 'auto',
             'min_samples_leaf' : 1,
             'n_jobs':-1,
    }
    
    clf = RandomForestClassifier(**param)
    print clf
    clf.fit(X_train[:,feats_to_use],Y)
    
    pr_val = clf.predict(X_valid[:,feats_to_use])
    pr_test = clf.predict(X_test[:,feats_to_use])
    return pr_val, pr_test


def process_madeline(X_train, Y, X_valid, X_test):
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
        
    clf = RandomForestClassifier(n_estimators=5000, n_jobs=-1, criterion='entropy', bootstrap=False)
    print clf
    clf.fit(X_train[:,feats_to_use],Y)
    
    pr_val = clf.predict(X_valid[:,feats_to_use])
    pr_test = clf.predict(X_test[:,feats_to_use])
    return pr_val, pr_test


def process_sylvine(Xtrain, ytrain, Xval, Xtest):
    print 'ITS A SYLVINE TIME'
    print
    
    t0 = time.time()
         
    goods = np.array([False, False, False, False, False, False, True, False, True, True, False, False,
 False, False, True, True, False, False, False, True])

    Xnewtrain = np.array(Xtrain[:, goods])
    Xnewtest = np.array(Xtest[:, goods])
    Xnewval = np.array(Xval[:, goods])

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
        
    modelrf = ExtraTreesClassifier(n_estimators = 10000, n_jobs = -1)
    modelrf.fit(Xnewtrain, ytrain)
    
    print 'STASON ET DONE'
    print (time.time() - t0) / 60.
             
    ytestrf = modelrf.predict_proba(Xnewtest)[:, 1]
    yvalrf = modelrf.predict_proba(Xnewval)[:, 1]


    ytestfinal = np.round(ytestrf)
    yvalfinal = np.round(yvalrf)
             
    return yvalfinal, ytestfinal


def process_philippine(X_train, Y, X_valid, X_test):
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
    
    param = {'n_estimators': 5000,
             'criterion' : 'entropy',
             'max_depth':None,
             'max_features' : 'auto',
             'min_samples_leaf' : 1,
         
             'n_jobs':-1,
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