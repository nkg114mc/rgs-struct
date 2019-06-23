
## Randomized Greedy Search for Structured Prediction

### Related Publication
[Randomized Greedy Search for Structured Prediction: Amortized Inference and Learning](http://people.oregonstate.edu/~machao/homepage/xxxxxxxxxxx.pdf) <br/>
 Chao Ma, F A Rezaur Rahman Chowdhury, Aryan Deshwal, Md Rakibul Islam, Janardhan Rao Doppa, and Dan Roth


### Requirement Packages
The project is coded in both Java and Scala. The Scala code is only employed in the NLP applications.
Please include Java 8 JRE library and Scala 2.11 library in project or building path.

You also need to install the [Xgboost JVM-Package](http://xgboost.readthedocs.io/en/latest/jvm/index.html). Xgboost is only used for an alternamtive approach to learn the initialization classifier $h$. Note that Xgboost is not mandatory because you can always employ Logistic Regression model for $h$.

<!-- All other libraries has been included in the repository. -->
All other libraries can be downloaded from here.


### Directory Tree of the Project
The root of the project contains four folders:
```
.
|   +-- datasets
|   +-- logistic_models
|   +-- rgs-proj
|   +-- sl-config
```
- **datasets**: The sub directory contains all data set files (We will describe more details about the Data Files section).
- **rgs-proj**: The source code directory that contains all Java and Scala files.
- **logistic_models**: The models files of the initialization classifier $h$ in RGS($\alpha$) (See more details of **IID Classifier** paragraph in section 3 in the paper).
- **sl-config**: The configuration files of structured SVM learner for different task and datasets. 

### Data Files


### Quick Start

The main java file to run the project is `experiment.RndLocalSearchExperiment.java`.

The following is the description of running option:

`-name`: Dataset name. Candidate names: {HW_SMALL, HW_LARGE, NETTALK_STREE, NETTALK_PHONEME,
		YEAST, ENRON, BIBTEX, BOOKMARKS, ...(more detailed data file description will be provided in dataset section) }
`-startyp` = Initialization state generation type. Candidate values: { 
  UNIFORM_INIT (corresponds to RGS(0)),
  LOGISTIC_INIT (corresponds to RGS(1.0)), 
  ALPHA_INIT (corresponds to RGS($\alpha$)) 
  }
`-initAlpha`: The alpha value for RGS($\alpha$). This is only appilcable when `starttyp` is set to be `ALPHA_INIT`.

`-restartTrain`: Number of random restart for training.
`-restartTest`: Number of random restart for testing.
`-mlType`: Multi-label loss function type. Candidate values: { HAMMING_LOSS, EXMPF1_LOSS, EXMPACC_LOSS }
`-debug`: Output debug info in logistic regression or not.

`-cfgPath`: SSVM config file path (See more details about [Illinois-SL](https://github.com/CogComp/illinois-sl)).
`-svmModelPath`: SSVM model file path.
`-logsPath`: Logistic regression model path. This is model file of initialization classifier $h$ in RGS($\alpha$).

`-usePairFeat`: Use binary feature when it is true.
`-useTernFeat`: Use tenery feature when it is true.
`-useQuadFeat`: Use quatery feature when it is true.

`-runEvalLearn`: Do E-learning when it is true.
`-eIter`: Number of iterations of E-learning.

`-doCostCache` Do cost weight caching? Once the cost weight was cached, during testing the system will directly load the existing weight vector instead of re-running the training.

For example, if you want run the system on multi-label dataset Yeast, you can compile the project to a runable jar file `rgs.jar` with , and run with follow command:
```
java -jar rgs.jar -name YEAST -startyp LOGISTIC_INIT -restartTrain 1 -restartTest 1 -usePairFeat true -useTernFeat false -useQuadFeat false -mlType EXMPF1_LOSS -runEvalLearn false -doCostCache false
```

**Contact**<br/>
Chao Ma (machao@engr.orst.edu, nkg114mc@hotmail.com)<br/>
Oregon State University.

