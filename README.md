# Decision-Tree

This project is an exercise for CSCI 4350: Introduction to Artificial Intelligence. The goal of this project is to understand   
the basics of decision trees, information theory for calculating entropy, and probability/statistics. 

In this project I built a decision tree that uses entropy of data for node/attribute selection. There were two continous value datasets used in this project

#### iris.txt
A link to the original data set, with additional information can be found [here](https://archive.ics.uci.edu/ml/datasets/Iris)

#### cancer.txt
A link to the original data set, with additional information can be found [here](https://archive.ics.uci.edu/ml/datasets/Breast+Tissue)


## To run the project
To run the project you must use the split.bash file (a utility file to shuffle, split into test/train sets, and run the final script)
That being said make sure that split.bash is executable by running (while in project directory)  

``` chmod 700 split.bash```  

then while running split.bash make sure to specify the amount of examples you'd like to reserve as a test set

``` ./split.bash <size of test set> "python3 id3.py" ```


## id3.py
id3.py will use the data to build an id3 decision tree and then use the test set to make predicitons. The program outputs the number of 
examples that it classified correctly. 
