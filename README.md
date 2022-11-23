# CI_Coursework
To run code:
- install anaconda 
- navigate to project location in termial
- conda env create -f comp_env.yml
- packages should install without error if not then follow 'To make env in ubuntu'
- conda activate comp_intel
Now you should be able to run any of the scripts with python [script_name].py

To make env in ubuntu:
- conda create --name comp_intel python=3.8
- run 'train_sgd.py' and 'train_pso.py' and install the packages that are missing as said by the error messages
- Once you get a successful run of both scripts you're done
