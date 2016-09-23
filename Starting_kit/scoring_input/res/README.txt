In the mode “Results Scoring Only”, the Codalab platform copies everything that was submitted to the “res” subdirectory (including the code, if any). In the code execution mode, it writes the outputs of the program to the “res” directory. A metadata file is added showing the execution time.

It is convenient to be able to submit both results and code because one phase might use result submission and the next one code submission. 
Phase 1: Development phase, submit results on validation data. Attach code.
Phase 2: No submissions, the last submission of phase 1 is forwarded and executed on test data.