import os
import sys, getopt
import time

import numpy

import theano
import theano.tensor as T
from sklearn import preprocessing
from nn import NN

from logistic_sgd import LogisticRegression, load_data


def fit_predict(data, labels, test_datasets = [], learning_rate=0.01, L1_reg=0.00, L2_reg=0.000001, n_epochs=200, batch_size = 100):
    data = numpy.nan_to_num(data)
    data_num = data
    if n_epochs > 50:
        n_epochs = 50
            # above all the data are said to be numerical; below one-hot-encoding sketch for categorical variables (\in 0..K-1)
            #ohe = preprocessing.OneHotEncoder()
            #ohe.fit(data_cat)
            #data_cat = ohe.transform(data_cat).toarray()
            #data = numpy.hstack((data_num, data_cat))

    NUM_TRAIN = len(data)
    if NUM_TRAIN % batch_size != 0: #if the last batch is not full, just don't use the remainder
        whole = (NUM_TRAIN / batch_size) * batch_size
        data = data[:whole]
        NUM_TRAIN = len(data) 
    #labels  = numpy.loadtxt('orange_train.solution')
    
    ### normalization by each column
    num_len = len(data_num[0])
    ameanW = [0] * num_len
    astdW = [0] * num_len
    for i in range(num_len):
         ameanW[i] = numpy.mean(data[:,i])
         data[:,i] = (data[:,i] - ameanW[i])
         astdW[i] = numpy.std(data[:,i]) + 0.1
         data[:,i] = (data[:,i]) / astdW[i]

    # random permutation
    indices = numpy.random.permutation(NUM_TRAIN)
    data, labels = data[indices, :], labels[indices]
    

    # batch_size == 100. We will use 98% of the data for training, and each 99th and 100th element to validate the NN while training
    is_train = numpy.array( ([0]* (batch_size - 2) + [1, 1]) * (NUM_TRAIN / batch_size))
    
    # now we split the dataset to test and valid datasets
    train_set_x, train_set_y = numpy.array(data[is_train==0]), labels[is_train==0]
    valid_set_x, valid_set_y = numpy.array(data[is_train==1]), labels[is_train==1]

    # compute number of minibatches 
    n_train_batches = len(train_set_y) / batch_size
    n_valid_batches = len(valid_set_y) / batch_size


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    # allocate symbolic variables for the data
    epoch = T.scalar()
    index = T.lscalar()  # index to a [mini]batch
    
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    #inone2 = T.matrix('inone2') 
    rng = numpy.random.RandomState(8000)

    # construct the NN class
    classifier = NN(
        rng=rng,
        input=x,
        n_in= len(data[0]),
        n_hidden1=20,
        n_hidden2=10,
        n_out=2
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch

    train_set_x = theano.shared(numpy.asarray(train_set_x, dtype=theano.config.floatX))
    train_set_y = T.cast(theano.shared(numpy.asarray(train_set_y, dtype=theano.config.floatX)), 'int32')
    valid_set_y = T.cast(theano.shared(numpy.asarray(valid_set_y, dtype=theano.config.floatX)), 'int32')
    valid_set_x = theano.shared(numpy.asarray(valid_set_x, dtype=theano.config.floatX)) 
    

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    best_iter = 0
    test_score = 0.
    start_time = time.clock()
    epoch = 0

    # here is an example how to print the current value of a Theano variable: print test_set_x.shape.eval()
    
    # start training
    while (epoch < n_epochs):
        epoch = epoch + 1   
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (epoch) % 50  == 0 and minibatch_index==0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

    # start predicting
    RET = []
    for it in range(len(test_datasets)):
        test_data = test_datasets[it]
        N = len(test_data)
        N_fill = N
        if N % batch_size != 0:
            N_fill = N + batch_size - (N % batch_size)
        print '....N = ', N, N_fill
        test_data = numpy.nan_to_num(test_data)
     
        # normalize by each column using coeffs from training data
        test_data_num = test_data
        for i in range(num_len):
            test_data_num[:, i] = (test_data_num[:,i] - ameanW[i]) / astdW[i]
        test_data = theano.shared(numpy.asarray(test_data, dtype=theano.config.floatX))
    
        # just zeroes
        test_labels = T.cast(theano.shared(numpy.asarray(numpy.zeros(batch_size), dtype=theano.config.floatX)), 'int32')
    
        ppm = theano.function([index], classifier.logRegressionLayer.pred_probs(),
            givens={
                x: test_data[index * batch_size: (index + 1) * batch_size],
                y: test_labels
            }, on_unused_input='warn')

        # p : predictions, we need to take column 1 for the class 1. p is 3-dim: (# loop iterations x batch_size x 2)
        p = [list(ppm(ii)[:, 1]) for ii in xrange( N_fill / batch_size)]  
        p_one = sum(p, [])
        p_one = numpy.array(p_one).reshape((N))
        RET.append(p_one)
    end_time = time.clock()              
    print >> sys.stderr, ('The code ran for %.2fm' % ((end_time - start_time) / 60.))
    return RET



