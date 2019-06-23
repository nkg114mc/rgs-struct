package experiment;

import java.util.List;
import java.util.Random;

import edu.berkeley.nlp.futile.fig.basic.Option;
import edu.berkeley.nlp.futile.fig.exec.Execution;
import edu.illinois.cs.cogcomp.sl.core.AbstractInferenceSolver;
import edu.illinois.cs.cogcomp.sl.core.SLModel;
import edu.illinois.cs.cogcomp.sl.core.SLParameters;
import edu.illinois.cs.cogcomp.sl.core.SLProblem;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearning.AbstractRegressionLearner;
import elearning.ElearningArg;
import elearning.XgbRegressionLearner;
import essvm.SSVMCopyDCDLearner;
import essvm.TrainResult;
import essvm.TrainSnapshot;
import experiment.RndLocalSearchExperiment.DataSetName;
import experiment.RndLocalSearchExperiment.InitType;
import experiment.RndLocalSearchExperiment.MulLbLossType;
import general.AbstractActionGenerator;
import general.AbstractFeaturizer;
import general.AbstractLossFunction;
import general.FactorGraphBuilder.FactorGraphType;
import imgseg.ImageDataReader;
import imgseg.ImageInstance;
import imgseg.ImageSegEvaluator;
import imgseg.ImageSegFeaturizer;
import imgseg.ImageSegLabel;
import imgseg.ImageSegMain;
import init.HwFastSamplingRndGenerator;
import init.ImageSegSamplingRndGenerator;
import init.MultiLabelSamplingRndGenerator;
import init.RandomStateGenerator;
import init.SeqSamplingRndGenerator;
import init.UniformRndGenerator;
import multilabel.MultiLabelFeaturizer;
import multilabel.MultiLabelNew;
import multilabel.data.Dataset;
import multilabel.data.DatasetReader;
import multilabel.instance.Label;
import multilabel.utils.WeightDumper;
import search.GreedySearcher;
import search.SeachActionGenerator;
import search.ZobristKeys;
import search.loss.SearchLossHamming;
import sequence.hw.HandWritingMain;
import sequence.hw.HwDataReader;
import sequence.hw.HwFeaturizer;
import sequence.hw.HwInstance;
import sequence.hw.HwLabelSet;
import sequence.hw.HwSearchInferencer;
import sequence.nettalk.NtkDataReader;
import sequence.nettalk.NtkFeaturizer;
import sequence.nettalk.NtkPhonemeLabelSet;
import sequence.nettalk.NtkStressLabelSet;

public class TrainingSpeedUpExperiments implements Runnable {

	@Option(gloss = "SSVM config file path.")
	public static String cfgPath = "";//"sl-config/search-DCD.config";
	@Option(gloss = "Logistic Regression model path.")
	public static String logsPath = "";///"../logistic_models/xxx.logistic";
	@Option(gloss = "SSVM model file path.")
	public static String svmModelPath = "";//"../logistic_models/xxx.ssvm";

	@Option(gloss = "Use binary feature?")
	public static boolean usePairFeat = true;
	@Option(gloss = "Use tenery feature?")
	public static boolean useTernFeat = true;
	@Option(gloss = "Use quatery feature?")
	public static boolean useQuadFeat = true;

	@Option(gloss = "Number of random restart runs.")
	public static int restart = 1;
	@Option(gloss = "Initialzier type")
	public static InitType startyp = InitType.UNIFORM_INIT;
	@Option(gloss = "Dataset name")
	public static DataSetName name = DataSetName.HW_SMALL;
	@Option(gloss = "Multi-label loss function type")
	public static MulLbLossType mlType = MulLbLossType.HAMMING_LOSS;
	@Option(gloss = "Output debug info in logistic regression or not")
	public static boolean debug = false;


	// about evaluation function learning

	@Option(gloss = "Do E-learning?")
	public static boolean runEvalLearn = true;
	@Option(gloss = "E-learning iterations")
	public static int eIter = 10;
	@Option(gloss = "Considering instance weight or not in eval learning")
	public static boolean useInstWght = false;
	@Option(gloss = "Trajectories in eval test")
	public static int evalTestRestartNum = 20;

	//@Option(gloss = "E-learning stop critieta")
	//public static StopType evalStop =  StopType.ITER_STOP;
	//public static int[] candidateEStopIters = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	//public static double[] candidateEStopPerct = { 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };


	@Option(gloss = "E featurizer ordering")
	public static int eFeatOrder = 2;

	public static ElearningArg eArgs;
	
	
	public static final String SPEED_FOLDER_PATH = "./../SpeedTrain";

	public static void main(String[] args) {
		TrainingSpeedUpExperiments trman = new TrainingSpeedUpExperiments();
		Execution.run(args, trman); // add .class here if that class should receive command-line args
	}

	private void printParams() {
		System.out.println("//////////////////////////////////////");
		System.out.println(" Name = " + name.toString());
		System.out.println(" cfgPath = " + cfgPath);
		System.out.println(" logsPath = " + logsPath);
		System.out.println(" svmModelPath = " + svmModelPath);
		System.out.println(" InitType = " + startyp.toString());
		System.out.println(" Restart = " + restart);
		System.out.println(" MultiLabelType = " + mlType.toString());
		System.out.println(" usePairFeat = " + usePairFeat);
		System.out.println(" useTernFeat = " + useTernFeat);
		System.out.println(" useQuadFeat = " + useQuadFeat);
		System.out.println(" LogsDbg = " + debug);
		System.out.println("//////////////////////////////////////");
		System.out.println();
	}

	@Override
	public void run() {

		DataSetName ds = name;

		eArgs = new ElearningArg();
		eArgs.runEvalLearn = runEvalLearn;
		eArgs.elearningIter = eIter;
		eArgs.considerInstWght = useInstWght;
		//eArgs.doEvalTest = doEvalTest;//runEvalLearn;
		eArgs.restartNumTest = evalTestRestartNum;

		if (eFeatOrder == 1) {
			eArgs.useFeat2 = false; eArgs.useFeat3 = false; eArgs.useFeat4 = false; 
		} else if (eFeatOrder == 2) {
			eArgs.useFeat2 = true; eArgs.useFeat3 = false; eArgs.useFeat4 = false; 
		} else if (eFeatOrder == 3) {
			eArgs.useFeat2 = true; eArgs.useFeat3 = true; eArgs.useFeat4 = false;
		} else if (eFeatOrder == 4) {
			eArgs.useFeat2 = true; eArgs.useFeat3 = true; eArgs.useFeat4 = true;
		} else {
			throw new RuntimeException("E feature order " + eFeatOrder + " is not valid!");
		}

		// have a look at params
		printParams();

		
		// Sequence
		if (ds == DataSetName.HW_SMALL) {
			runHandWritingSmall(true, "../sl-config/hw-small-search-DCD.config");
		} else if (ds == DataSetName.HW_LARGE) {
			runHandWritingSmall(false, "../sl-config/hw-search-DCD.config");
		} else if (ds == DataSetName.NETTALK_STREE) {
			runNetTalkStree();
		} else if (ds == DataSetName.NETTALK_PHONEME) {
			runNetTalkPhoneme();

			
		// Multi-label
		} else if (ds == DataSetName.YEAST) {
			runMulLbData("yeast");
		} else if (ds == DataSetName.ENRON) {
			runMulLbData("enron");
		} else if (ds == DataSetName.CAL500) {
			runMulLbData("CAL500");
		} else if (ds == DataSetName.COREL5K) {
			runMulLbData("Corel5k");
		} else if (ds == DataSetName.MEDIAMILL) {
			runMulLbData("mediamill");

		// Image Segmentation
		} else if (ds == DataSetName.MSRC21) {
			runMSRC21();

		} else {
			throw new RuntimeException("Unknown dataset name ...");
		}



		System.out.println("Done.");
	}



	public static void runMulLbData(String datasetName) {

		try {

			String dsName = datasetName;
			DatasetReader dataSetReader = new DatasetReader();
			Dataset ds = dataSetReader.readDefaultDataset(dsName);

			// load data

			String cfgPath = RndLocalSearchExperiment.cfgPath;
			if (cfgPath.equals("")) {
				cfgPath = "../sl-config/" + dsName + "-search-DCD.config";
			}
			String logsPath = RndLocalSearchExperiment.logsPath;
			if (logsPath.equals("")) {
				logsPath = "../logistic_models/" + dsName + ".logistic";
			}
			String svmPath = RndLocalSearchExperiment.svmModelPath;
			if (svmPath.equals("")) {
				svmPath = "../logistic_models/" + dsName + ".ssvm";
			}

			///////////////////////////////////////////////////////////////////

			//String dsName = ds.name;

			InitType initType = startyp;
			MulLbLossType lossTyp = mlType;
			int nrnd = restart;

			// load data
			List<List<HwInstance>> trtstInsts =  DatasetReader.convertToHwInstances(ds);
			System.out.println("List count = " + trtstInsts.size());


			RandomStateGenerator initStateGener = null;
			System.out.println("InitType = " + initType.toString());
			if (initType == InitType.UNIFORM_INIT) {
				initStateGener = new UniformRndGenerator(new Random());
			} else if (initType == InitType.LOGISTIC_INIT) {
				initStateGener = MultiLabelSamplingRndGenerator.loadGenrIfExist(logsPath, dsName, trtstInsts.get(0), trtstInsts.get(1), Label.MULTI_LABEL_DOMAIN, false);
			}
			System.out.println("=======");

			AbstractActionGenerator actGener = new SeachActionGenerator();
			AbstractLossFunction searchLossFunc = MultiLabelNew.buildLossFunction(lossTyp);

			List<SLProblem> slproblems = HwDataReader.convertToSLProblem(trtstInsts);

			//////////////////////////////////////////////////////////////////////
			// train
			SLModel model = new SLModel();
			SLProblem spTrain = slproblems.get(0);

			ZobristKeys abkeys = new ZobristKeys(500, Label.MULTI_LABEL_DOMAIN.length);
			MultiLabelFeaturizer fg = new MultiLabelFeaturizer(ds.getLabelDimension(), ds.getFeatureDimension(), true, false, false);

			GreedySearcher searcher = new GreedySearcher(FactorGraphType.MultiLabelGraph, fg, nrnd, actGener, initStateGener, searchLossFunc, abkeys);
			model.infSolver = new HwSearchInferencer(searcher);
			model.featureGenerator = fg;

			SLParameters para = new SLParameters();
			para.loadConfigFile(cfgPath);
			para.TOTAL_NUMBER_FEATURE = fg.getFeatLen();

			//////////////////////////////////////////////////////////////

			AbstractFeaturizer efr = new MultiLabelFeaturizer(ds.getLabelDimension(), ds.getFeatureDimension(), eArgs.useFeat2, eArgs.useFeat3, eArgs.useFeat4);
			TrainSpeedResult res = speedUpTrainWithEval(dsName, spTrain, trtstInsts.get(0), searcher, model.infSolver, fg, para, efr, initType);


			SLModel tmpMd = new SLModel();
			tmpMd.infSolver = model.infSolver;

		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println("MultiLabel All Done!");
	}


	///////////////////////////////////////////
	///////////////////////////////////////////
	///////////////////////////////////////////

	public static void runHandWritingSmall(boolean isSmall, String configFilePath) {

		try {

			int folderIndex = 0;
			String dsName = HandWritingMain.getDsName(isSmall, folderIndex);

			String logsPath = "../logistic_models/"+dsName+".logistic";
			String svmPath = "../logistic_models/"+dsName+".ssvm";
			int nrnd = restart;
			InitType initType = startyp;
			MulLbLossType lossTyp = mlType;

			// load data
			HwLabelSet hwLabels = new HwLabelSet();
			// load data
			HwDataReader rder = new HwDataReader();
			List<List<HwInstance>> trtstInsts = rder.readData("../datasets/hw", hwLabels, folderIndex, isSmall);
			System.out.println("List count = " + trtstInsts.size());
			List<SLProblem> slproblems = HwDataReader.convertToSLProblem(trtstInsts);

			RandomStateGenerator initStateGener = null;
			if (initType == InitType.UNIFORM_INIT) {
				initStateGener = new UniformRndGenerator(new Random());
			} else if (initType == InitType.LOGISTIC_INIT) {
				initStateGener = HwFastSamplingRndGenerator.loadGenrIfExist(logsPath, dsName, trtstInsts.get(0), trtstInsts.get(1), hwLabels.getLabels(), false, -1);
			}
			System.out.println("=======");

			AbstractLossFunction lossfunc = new SearchLossHamming();

			//////////////////////////////////////////////////////////////////////
			// train
			SLModel model = new SLModel();
			SLProblem spTrain = slproblems.get(0);

			// initialize the inference solver
			ZobristKeys abkeys = new ZobristKeys(500, hwLabels.getLabels().length);
			HwFeaturizer fg = new HwFeaturizer(hwLabels.getLabels(), HwFeaturizer.HwSingleLetterFeatLen, usePairFeat, useTernFeat, useQuadFeat);
			AbstractActionGenerator actGener = new SeachActionGenerator();

			GreedySearcher searcher = new GreedySearcher(FactorGraphType.SequenceGraph, fg, nrnd, actGener, initStateGener, lossfunc, abkeys);
			model.infSolver = new HwSearchInferencer(searcher);
			model.featureGenerator = fg;

			SLParameters para = new SLParameters();
			para.loadConfigFile(configFilePath);
			para.TOTAL_NUMBER_FEATURE = fg.getFeatLen();

			TrainSpeedResult res = speedUpTrainWithEval(dsName, spTrain, trtstInsts.get(0), searcher, model.infSolver, fg, para, null, initType);

		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	public static void runHandWritingLarge() {
		//HwLabelSet hwLabels = new HwLabelSet();
		//HwDataReader rder = new HwDataReader();
		//List<List<HwInstance>> trtstInsts = rder.readData("../datasets/hw", hwLabels, 0, false);
		//runOtherSeqData("HW-Large", trtstInsts, hwLabels.getLabels());

		//"../sl-config/hw-search-DCD.config"
	}

	public static void runNetTalkStree() {
		try {

			String logsPath = "../logistic_models/nettalk_stress.logistic";
			String svmPath = "../logistic_models/nettalk_stress.ssvm";
			String configFilePath = "../sl-config/stress-search-DCD.config";
			int nrnd = restart;
			InitType initType = startyp;
			MulLbLossType lossTyp = mlType;

			// load data
			NtkStressLabelSet stLabels = new NtkStressLabelSet();
			NtkDataReader rder = new NtkDataReader();
			List<List<HwInstance>> trtstInsts = rder.readData("../datasets/nettalk_stress_train.txt", "../datasets/nettalk_stress_test.txt", stLabels.getLabels());
			System.out.println("List count = " + trtstInsts.size());
			List<SLProblem> slproblems = HwDataReader.convertToSLProblem(trtstInsts);

			RandomStateGenerator initStateGener = null;
			if (initType == InitType.UNIFORM_INIT) {
				initStateGener = new UniformRndGenerator(new Random());
			} else if (initType == InitType.LOGISTIC_INIT) {
				initStateGener = SeqSamplingRndGenerator.loadGenrIfExist(logsPath, "nettalk_stress", trtstInsts.get(0), trtstInsts.get(1), stLabels.getLabels(), false, -1);
			}
			System.out.println("=======");

			AbstractLossFunction lossfunc = new SearchLossHamming();

			//////////////////////////////////////////////////////////////////////
			// train
			SLModel model = new SLModel();
			SLProblem spTrain = slproblems.get(0);

			// initialize the inference solver
			ZobristKeys abkeys = new ZobristKeys(500, stLabels.getLabels().length);
			HwFeaturizer fg = new HwFeaturizer(stLabels.getLabels(), NtkFeaturizer.NetTalkSingleFeatLen, usePairFeat, useTernFeat, useQuadFeat);
			AbstractActionGenerator actGener = new SeachActionGenerator();

			GreedySearcher searcher = new GreedySearcher(FactorGraphType.SequenceGraph, fg, nrnd, actGener, initStateGener, lossfunc, abkeys);
			model.infSolver = new HwSearchInferencer(searcher);
			model.featureGenerator = fg;

			SLParameters para = new SLParameters();
			para.loadConfigFile(configFilePath);
			para.TOTAL_NUMBER_FEATURE = fg.getFeatLen();


			//////////////////////////////////////////////////////////////

			AbstractFeaturizer efr = new HwFeaturizer(stLabels.getLabels(), NtkFeaturizer.NetTalkSingleFeatLen, eArgs.useFeat2, eArgs.useFeat3, eArgs.useFeat4);
			TrainSpeedResult res = speedUpTrainWithEval("nettalk_stress", spTrain, trtstInsts.get(0), searcher, model.infSolver, fg, para, efr, initType);

			//dumpTrainSpeedCurveCsv("../TrSpeed", res, "nettalk_stress", initType, evalStop);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void runNetTalkPhoneme() {

		try {

			String logsPath = "../logistic_models/nettalk_phoneme.logistic";
			String svmPath = "../logistic_models/nettalk_phoneme.ssvm";
			String configFilePath = "../sl-config/phoneme-search-DCD.config";
			int nrnd = restart;
			InitType initType = startyp;
			MulLbLossType lossTyp = mlType;
			/*
		try {
			NetTalkPhonemeMain.runLearning(startyp, restart, 
					                       "../sl-config/phoneme-search-DCD.config", 
					                       "../logistic_models/nettalk_phoneme.logistic", 
					                       "../logistic_models/nettalk_phoneme.ssvm",
					                       usePairFeat, useTernFeat, useQuadFeat, eArgs);
		} catch (Exception e) {
			e.printStackTrace();
		}*/

			// load data
			NtkPhonemeLabelSet phLabels = new NtkPhonemeLabelSet();
			NtkDataReader rder = new NtkDataReader();
			List<List<HwInstance>> trtstInsts = rder.readData("../datasets/nettalk_phoneme_train.txt", "../datasets/nettalk_phoneme_test.txt", phLabels.getLabels());
			System.out.println("List count = " + trtstInsts.size());

			RandomStateGenerator initStateGener = null;
			if (initType == InitType.UNIFORM_INIT) {
				initStateGener = new UniformRndGenerator(new Random());
			} else if (initType == InitType.LOGISTIC_INIT) {
				initStateGener = SeqSamplingRndGenerator.loadGenrIfExist(logsPath, "nettalk_phoneme", trtstInsts.get(0), trtstInsts.get(1), phLabels.getLabels(), false, -1);
			}
			System.out.println("=======");

			AbstractLossFunction lossfunc = new SearchLossHamming();
			AbstractActionGenerator actGener = new SeachActionGenerator();

			List<SLProblem> slproblems = HwDataReader.convertToSLProblem(trtstInsts);

			//////////////////////////////////////////////////////////////////////
			// train
			SLModel model = new SLModel();
			SLProblem spTrain = slproblems.get(0);

			// initialize the inference solver
			ZobristKeys abkeys = new ZobristKeys(500, phLabels.getLabels().length);
			HwFeaturizer fg = new HwFeaturizer(phLabels.getLabels(), NtkFeaturizer.NetTalkSingleFeatLen, usePairFeat, useTernFeat, useQuadFeat);//true, true, true);
			//model.infSolver = new HwInferencer(fg);

			GreedySearcher searcher = new GreedySearcher(FactorGraphType.SequenceGraph, fg, nrnd, actGener,initStateGener, lossfunc, abkeys);
			model.infSolver = new HwSearchInferencer(searcher);
			//model.infSolver = new HwSearchInferencer(fg,1,abkeys);
			//model.infSolver = new HwViterbiInferencer(fg);
			model.featureGenerator = fg;

			SLParameters para = new SLParameters();
			para.loadConfigFile(configFilePath);
			para.TOTAL_NUMBER_FEATURE = fg.getFeatLen();

			TrainSpeedResult res = speedUpTrainWithEval("nettalk_phoneme", spTrain, trtstInsts.get(0), searcher, model.infSolver, fg, para, null, initType);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}


	/////////////////////////////////////////////////////////////////

	public static void runMSRC21() {

		try {

			ImageSegLabel[] labels = ImageSegLabel.loadLabelFromFile("../msrc/imageseg_label_color_map.txt");
			String[] labelNames = ImageSegLabel.getStrLabelArr(labels, false);
			String[] labelNamesFull = ImageSegLabel.getStrLabelArr(labels, true);

			ImageSegEvaluator.initRgbToLabel(labels);

			ImageDataReader reader = new ImageDataReader("../msrc");
			ImageSegEvaluator evaluator = new ImageSegEvaluator(reader.getDebugFolder());

			List<ImageInstance> trainInsts = ImageSegMain.loadFromListFile(reader, "../msrc/Train.txt", labelNames, true);		
			List<ImageInstance> testInsts = ImageSegMain.loadFromListFile(reader, "../msrc/Test.txt", labelNamesFull, true);
			
			List<HwInstance> trnHwInsts = ImageSegMain.imageInstToHwInst(trainInsts);
			List<HwInstance> tstHwInsts = ImageSegMain.imageInstToHwInst(testInsts);

			String svmCfgFile = "../sl-config/msrc21-search-DCD.config";
			String modelLogsFn = "../logistic_models/msrc21.logistic";
			String modelSvFn = "../logistic_models/msrc21.ssvm";


			int nrnd = restart;
			InitType initType = startyp;
			MulLbLossType lossTyp = mlType;		
			String configFilePath = svmCfgFile;
			String dsName = "msrc21";

			SLProblem spTrain = ImageDataReader.ExampleListToSLProblem(trainInsts);

			RandomStateGenerator initStateGener = null;
			if (initType == InitType.UNIFORM_INIT) {
				initStateGener = new UniformRndGenerator(new Random());
			} else if (initType == InitType.LOGISTIC_INIT) {
				initStateGener = ImageSegSamplingRndGenerator.loadGenrIfExist(modelLogsFn, dsName, trainInsts, testInsts, labelNames, false, -1);
			}
			System.out.println("=======");

			AbstractLossFunction lossfunc = new SearchLossHamming();


			//////////////////////////////////////////////////////////////////////
			// train
			SLModel model = new SLModel();

			// initialize the inference solver
			ZobristKeys abkeys = new ZobristKeys(1000, labelNames.length + 1);
			ImageSegFeaturizer fg = new ImageSegFeaturizer(labelNames, usePairFeat, useTernFeat);// true, true);
			AbstractActionGenerator actGener = new SeachActionGenerator();

			GreedySearcher searcher = new GreedySearcher(FactorGraphType.ImageSegGraph, fg, nrnd, actGener, initStateGener, lossfunc, abkeys);
			model.infSolver = new HwSearchInferencer(searcher);
			model.featureGenerator = fg;

			SLParameters para = new SLParameters();
			para.loadConfigFile(configFilePath);
			para.TOTAL_NUMBER_FEATURE = fg.getFeatLen();

			TrainSpeedResult res = speedUpTrainWithEval(dsName, spTrain, trnHwInsts, searcher, model.infSolver, fg, para, null, initType);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	//////////////////////////////////////////////////////////////////





	public static TrainSpeedResult speedUpTrainWithEval(String dsName,
			SLProblem problem,
			List<HwInstance> trInsts, 
			GreedySearcher gsearcher,
			AbstractInferenceSolver infsolver,
			AbstractFeaturizer fg,
			SLParameters params,
			
			AbstractFeaturizer efr,
			InitType initType) {
		
		try {

			String weightFolder = "./wdumps";
			SeqSamplingRndGenerator.checkArffFolder(weightFolder);
			
			SSVMCopyDCDLearner learner = new SSVMCopyDCDLearner(infsolver, fg, params);
			String runname = "";
			
			int evalIterations = 1;
			AbstractRegressionLearner regser = new XgbRegressionLearner(dsName);
					
			// Case (1)
			
			WeightDumper wdumper1 = new WeightDumper(weightFolder + "/" + dsName + "-restart1", "restart1");
			gsearcher.setRestart(1);
			AbstractInferenceSolver inference1 = new HwSearchInferencer(gsearcher);
			TrainResult trnResRestart1 = new TrainResult(wdumper1);
			//WeightVector cweight1 = learner.trainNormail(problem, params, gsearcher, inference1, trnResRestart1, false);
			WeightVector cweight1 = learner.trainNormail(problem, params, gsearcher, trnResRestart1, false, regser, evalIterations);

			runname = "Restart_1_NoEval";
			System.out.println("");
			System.out.println("=="+runname+"====================");
			trnResRestart1.printTrainResult();
			System.out.println("=================================");
			System.out.println("");
			
			// Case (2)
			
			WeightDumper wdumper2 = new WeightDumper(weightFolder + "/" + dsName + "-restart20", "restart20");
			gsearcher.setRestart(20);
			AbstractInferenceSolver inference2 = new HwSearchInferencer(gsearcher);
			TrainResult trnResRestart20 = new TrainResult(wdumper2);
			//WeightVector cweight2 = learner.trainNormail(problem, params, gsearcher, inference2, trnResRestart20, false);
			WeightVector cweight2 = learner.trainNormail(problem, params, gsearcher,  trnResRestart20, false, regser, evalIterations);
			
			runname = "Restart_20_NoEval";
			System.out.println("");
			System.out.println("=="+runname+"====================");
			trnResRestart20.printTrainResult();
			System.out.println("=================================");
			System.out.println("");
			
			// Case (3)
			WeightDumper wdumper3 = new WeightDumper(weightFolder + "/" + dsName + "-restarte", "restarte");
			gsearcher.setRestart(1);
			AbstractInferenceSolver inference3 = new HwSearchInferencer(gsearcher);
			TrainResult trnResRestart1e = new TrainResult(wdumper3);
			//WeightVector cweight3 = learner.trainNormail(problem, params, gsearcher, inference3, trnResRestart1e, true);
			WeightVector cweight3 = learner.trainNormail(problem, params, gsearcher, trnResRestart1e, false, regser, evalIterations);
			
			runname = "Restart_withEval";
			System.out.println("");
			System.out.println("=="+runname+"====================");
			trnResRestart20.printTrainResult();
			System.out.println("=================================");
			System.out.println("");
			
			/////////////////////////////////////
			
			TrainSpeedResult speedResult = new TrainSpeedResult(trnResRestart1, trnResRestart20, trnResRestart1e);
			
			// compute accuracy for curves
			computeAccuracyAll(speedResult, trInsts, gsearcher, 20);
			
			// dump to file!
			SeqSamplingRndGenerator.checkArffFolder(SPEED_FOLDER_PATH);
			speedResult.dumpTrainSpeedResult(SPEED_FOLDER_PATH, dsName, initType);
			
			return speedResult;
			
		} catch (Exception e) {
			e.printStackTrace();
		}

		return null;
	}

	

	public static void computeAccuracyAll(TrainSpeedResult speedResult, List<HwInstance> trInsts, GreedySearcher gsearcher, int restarts) {
		
		if (speedResult.accurateResult != null) {
			computeTrainAcc(speedResult.accurateResult, trInsts,  gsearcher, restarts);
		}
		
		if (speedResult.inaccurateResult != null) {
			computeTrainAcc(speedResult.inaccurateResult, trInsts,  gsearcher, restarts);
		}
		
		if (speedResult.evalResult != null) {
			computeTrainAcc(speedResult.evalResult, trInsts,  gsearcher, restarts);
		}
		
	}
	
	public static void computeTrainAcc(TrainResult tre, List<HwInstance> trInsts,  GreedySearcher gsearcher, int restarts) {
		for (TrainSnapshot tss : tre.snapshots) {
			WeightVector w = tss.c_weight;
			if (w == null) {
				if (tre.wdumper != null) {
					w = WeightDumper.loadWeightFromFile(tss.weightFilePath);//.loadWeight(iteration)
				} else {
					throw new RuntimeException("Dumper is null!");
				}
				
			}
			tss.trainAccuracy = TrainSnapshot.computeAcc(w, trInsts, gsearcher, restarts);
			System.gc();
		}
	}

}