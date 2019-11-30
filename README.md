# Decision-Tree

This project is an exercise for CSCI 4350: Introduction to Artificial Intelligence. The goal of this project is to understand   
the basics of decision trees, information theory for calculating entropy, and probability/statistics. 

In this project I built a decision tree that uses entropy of data for node/attribute selection. There were two continous value datasets used in this project

#### iris.txt
A link to the original data set, with additional information can be found [here](https://archive.ics.uci.edu/ml/datasets/Iris)

#### cancer.txt
A link to the original data set, with additional information can be found [here](https://archive.ics.uci.edu/ml/datasets/Breast+Tissue)

The project used both of these datasets to build an id3 decision tree for each and then test them by running each different test size (n) 100 times (each outputting the number of correct classificitons). 

##### Iris
```n=[1,5,10,25,50,75,100,125,140,145,149]``` 1100 runs total  
##### Cancer 
```n=[1,5,10,25,50,75,90,100,104]``` 900 runs total  

## To run the project
To run the project you must use the split.bash file (a utility file to shuffle, split into test/train sets, and run the final script)
That being said make sure that split.bash is executable by running (while in project directory)  

``` chmod 700 split.bash```  

then while running split.bash make sure to specify the amount of examples you'd like to reserve as a test set

``` ./split.bash <size of test set> "python3 id3.py" ```


## The files

### id3.py
id3.py will use the data to build an id3 decision tree and then use the test set to make predicitons. The program outputs the number of examples that it classified correctly. 

### tree.py
a simple utility file to help id3.py do its job

### report.pdf
a full breakdown of the work down in this project

### split.bash
(Created by Joshua Phillips)
A bash script used to run the project. It takes a specified datastet.txt file, shuffles it, split it into a test.txt file and a train.txt file and uses them as input to a specified command (in our case, id3.py)

### iris_results.txt & cancer_results.txt & plots.ipynb
 iris_results.txt & cancer_results.txt are the reuslts from running the project with the differing test/train splits. Each column represents the split with each value being the number of correct classifications. 
 
plots.ipynb uses the iris_results.txt & cancer_results.txt to generate these plots: 
![](https://firebasestorage.googleapis.com/v0/b/github-images.appspot.com/o/Screen%20Shot%202019-11-29%20at%2018.21.41.png?alt=media&token=ee5a43f9-1c3e-468b-9685-19292a263b11)





