#!/usr/bin/env python

# Merge two cristines and and SVM




# Specify the parameters common for all datasets
params = {}
params['n_jobs'] = 40 # number of cores to use on *your* machine
params['n_folds'] = 11 # number of cross-validation splits
params['cheating'] = True # choose between dataset specific
                                 # algoriths or general AutoML learning

ACTIVE_DATASETS = 'p' # first letters of the datasets that you want
                # 'cjmps' # to process on your machine.
                          # All datasets are processed on codelab.
                          

verbose = True # outputs messages to stdout and stderr for debug purposes
# Debug level:
############## 
# 0: run the code normally, using the time budget of the tasks
# 1: run the code normally, but limits the time to max_time
# 2: run everything, but do not train, generate random outputs in max_time
# 3: stop before the loop on datasets
# 4: just list the directories and program version
debug_mode = 0


import datetime
zipme = True # use this flag to enable zipping of your code submission
the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
submission_filename = '../automl_sample_submission_' + the_date

# I/O defaults
##############
default_input_dir = "/home/tko/sizov/AutoML/data/round_1"
default_output_dir = "res"

# =========================== END USER OPTIONS ================================
version = 2.7

# General purpose functions
import os
from sys import argv, path
import numpy as np
import time
overall_start = time.time()

# Our directories
# Note: On codalab, there is an extra sub-directory called "program"
running_on_codalab = False
run_dir = os.path.abspath(".")
codalab_run_dir = os.path.join(run_dir, "program")
if os.path.isdir(codalab_run_dir): 
    run_dir=codalab_run_dir
    running_on_codalab = True
    print "Running on Codalab!"
    params['n_jobs'] = -1 # Take as many cores as they have
    
lib_dir = os.path.join(run_dir, "lib")
res_dir = os.path.join(run_dir, "res")

# Our libraries  
path.append (run_dir)
path.append (lib_dir)
import data_io                       # general purpose input/output functions
from data_io import vprint           # print only in verbose mode
from data_manager import DataManager # load/save data and get info about them
import learning_routine
# import run_nn

if debug_mode >= 4 or running_on_codalab: # Show library version and directory structure
    data_io.show_version()
    data_io.show_dir(run_dir)

# =========================== BEGIN PROGRAM ================================

if __name__=="__main__" and debug_mode<4:	
    #### Check whether everything went well (no time exceeded)
    execution_success = True
    
    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
    else:
        input_dir = argv[1]
        output_dir = os.path.abspath(argv[2]);
    # Move old results and create a new output directory 
    if not(running_on_codalab):
        data_io.mvdir(output_dir, '../'+output_dir+'_'+the_date) 
    data_io.mkdir(output_dir) 
    
    #### INVENTORY DATA (and sort dataset names alphabetically)
    datanames = data_io.inventory_data(input_dir)
    
    #### DEBUG MODE: Show dataset list and STOP
    if debug_mode>=3:
        data_io.show_io(input_dir, output_dir)
        print('\n****** Sample code version ' + str(version) + ' ******\n\n' + '========== DATASETS ==========\n')        	
        data_io.write_list(datanames)      
        datanames = [] # Do not proceed with learning and testing
        
    # ==================== @RESULT SUBMISSION (KEEP THIS) =====================
    # Always keep this code to enable result submission of pre-calculated results
    # deposited in the res/ subdirectory.
    if len(datanames)>0:
        vprint( verbose,  "************************************************************************")
        vprint( verbose,  "****** Attempting to copy files (from res/) for RESULT submission ******")
        vprint( verbose,  "************************************************************************")
        OK = data_io.copy_results(datanames, res_dir, output_dir, verbose) # DO NOT REMOVE!
        if OK: 
            vprint( verbose,  "[+] Success")
            datanames = [] # Do not proceed with learning and testing
        else:
            vprint( verbose, "======== Some missing results on current datasets!")
            vprint( verbose, "======== Proceeding to train/test:\n")
    # =================== End @RESULT SUBMISSION (KEEP THIS) ==================

    # ================ @CODE SUBMISSION (SUBTITUTE YOUR CODE) ================= 
    overall_time_budget = 0
    for basename in datanames: # Loop over datasets
        
        vprint( verbose,  "************************************************")
        vprint( verbose,  "******** Processing dataset " + basename.capitalize() + " ********")
        vprint( verbose,  "************************************************")
        
        # ======== Learning on a time budget:
        # Keep track of time not to exceed your time budget. Time spent to inventory data neglected.
        start = time.time()
        
        # ======== Creating a data object with data, informations about it
        vprint( verbose,  "======== Reading and converting data ==========")
        D = DataManager(basename, input_dir, verbose=verbose)
        #D = DataManager(basename, input_dir, replace_missing=True, filter_features=True, verbose=verbose)
        print D
        #####################################################################################################################################################
        # The parameters unique for each dataset
        params['metric'] = D.info['metric'] 
        params['is_sparse'] = ( D.info['is_sparse'] == 1)

        #######################################################################################################################################
        ###
        #Do learning here
        if basename == 'christine':
            print('process christine')
            if running_on_codalab or (basename[0] in ACTIVE_DATASETS):
                #Y_valid, Y_test = learning_routine.process_christine(D.data['X_train'], D.data['Y_train'], D.data['X_valid'], D.data['X_test'], params)
                Y_valid, Y_test = learning_routine.process_my_christine(D.data['X_train'], D.data['Y_train'], D.data['X_valid'], D.data['X_test'], params)
            else:
                Y_valid, Y_test = [[], []]
                
        elif basename == 'jasmine':
            print('process jasmine')
            if running_on_codalab or (basename[0] in ACTIVE_DATASETS):
                #Y_valid, Y_test = learning_routine.process_jasmine(D.data['X_train'], D.data['Y_train'], D.data['X_valid'], D.data['X_test'], params)
                Y_valid, Y_test = learning_routine.default_prediction(D.data['X_train'], D.data['Y_train'], D.data['X_valid'], D.data['X_test'], params)
            else:
                Y_valid, Y_test = [[], []]
                
        elif basename == 'madeline':
            print('process madeline')
            if running_on_codalab or (basename[0] in ACTIVE_DATASETS):
                Y_valid, Y_test = learning_routine.process_madeline(D.data['X_train'], D.data['Y_train'], D.data['X_valid'], D.data['X_test'], params)    
            else:
                Y_valid, Y_test = [[], []]
        
        elif basename == 'philippine':
            print('process philippine')
            if running_on_codalab or (basename[0] in ACTIVE_DATASETS):
                Y_valid, Y_test = learning_routine.process_philippine(D.data['X_train'], D.data['Y_train'], D.data['X_valid'], D.data['X_test'], params)    
            else:
                Y_valid, Y_test = [[], []]
            
        elif basename == 'sylvine':
            print('process sylvine')
            if running_on_codalab or (basename[0] in ACTIVE_DATASETS):
                Y_valid, Y_test = learning_routine.process_sylvine(D.data['X_train'], D.data['Y_train'], D.data['X_valid'], D.data['X_test'], params)    
            else:
                Y_valid, Y_test = [[], []]
            
        else:
            print("============ Unknown dataset!: {}".format(basename))
            Y_valid, Y_test = learning_routine.default_prediction(D.data['X_train'], D.data['Y_train'], D.data['X_valid'], D.data['X_test'], params)

        ###
        
        time_spent = time.time() - start
        vprint( verbose,  "[+] Prediction success, time spent so far %5.2f sec" % (time.time() - start))
        # Write results
        filename_valid = basename + '_valid_000.predict'
        data_io.write(os.path.join(output_dir,filename_valid), Y_valid)
        filename_test = basename + '_test_000.predict'
        data_io.write(os.path.join(output_dir,filename_test), Y_test)
        vprint( verbose,  "[+] Results saved, time spent so far %5.2f sec" % (time.time() - start))         
        time_spent = time.time() - start 
        
    if zipme and not(running_on_codalab):
        vprint( verbose,  "========= Zipping this directory to prepare for submit ==============")
        data_io.zipdir(submission_filename + '.zip', ".")
    	
    overall_time_spent = time.time() - overall_start
    if execution_success:
        vprint( verbose,  "[+] Done")
        vprint( verbose,  "[+] Overall time spent %5.2f sec " % overall_time_spent + "::  Overall time budget %5.2f sec" % overall_time_budget)
    else:
        vprint( verbose, "[E] Some error")
              
    if running_on_codalab: 
        if execution_success:
            exit(0)
        else:
            exit(1)
