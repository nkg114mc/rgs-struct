package experiment;

import java.util.List;

import edu.berkeley.nlp.futile.fig.basic.Option;
import edu.berkeley.nlp.futile.fig.exec.Execution;
import edu.illinois.cs.cogcomp.sl.core.SLModel;
import elearning.ElearningArg;
import imgseg.ImageDataReader;
import imgseg.ImageInstance;
import imgseg.ImageSegEvaluator;
import imgseg.ImageSegLabel;
import imgseg.ImageSegMain;
import multilabel.BookMarksMain;
import multilabel.MultiLabelNew;
import sequence.hw.HandWritingMain;
import sequence.nettalk.NetTalkPhonemeMain;
import sequence.nettalk.NetTalkStressMain;

public class RndLocalSearchExperiment implements Runnable {
	
	//@Option(gloss = "Just train logistic model?")
	//public static boolean logsOnly = false;
	
	@Option(gloss = "Run on train/test (false) or train/validation (true) split of data.")
	public static boolean runOnDev = false;
	
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

	//@Option(gloss = "Number of random restart runs.")
	//public static int restart = 1;
	@Option(gloss = "Number of random restart for training.")
	public static int restartTrain = -1;
	@Option(gloss = "Number of random restart for testing.")
	public static int restartTest = -1;
	@Option(gloss = "Initialzier type")
	public static InitType startyp = InitType.UNIFORM_INIT;
	@Option(gloss = "Dataset name")
	public static DataSetName name = DataSetName.HW_SMALL;
	@Option(gloss = "Multi-label loss function type")
	public static MulLbLossType mlType = MulLbLossType.HAMMING_LOSS;
	@Option(gloss = "Output debug info in logistic regression or not")
	public static boolean debug = false;
	@Option(gloss = "Alpha pruner alpha value")
	public static double initAlpha = 0;
	
	// about evaluation function learning

	@Option(gloss = "Do E-learning?")
	public static boolean runEvalLearn = true;
	@Option(gloss = "E-learning iterations")
	public static int eIter = 20;
	@Option(gloss = "Considering instance weight or not in eval learning")
	public static boolean useInstWght = false;
	@Option(gloss = "Trajectories in eval test")
	public static int evalTestRestartNum = 20;
	
	@Option(gloss = "Do eval test?")
	public static boolean doEvalTest = true;
	
	@Option(gloss = "E featurizer ordering")
	public static int eFeatOrder = 4;
	
	@Option(gloss = "Do cost weight caching?")
	public static boolean doCostCache = true;
	
	public static ElearningArg eArgs;
	
	public static CostFuncCacherAndLoader costCacher;

	
	public static enum InitType {
		UNIFORM_INIT, LOGISTIC_INIT, ALLZERO_INIT,
		ALPHA_INIT
	}
	
	public static enum DataSetName {
		HW_SMALL, HW_LARGE, NETTALK_STREE, NETTALK_PHONEME,
		YEAST, ENRON, CAL500, COREL5K, MEDIAMILL, BIBTEX, BOOKMARKS, DELICIOUS,
		MSRC21, 
		ACE05, ONTONOTES5,
		HW_SMALL_FULL, HW_LARGE_FULL,
		HORSE32, HORSE128
	}
	
	public static enum MulLbLossType {
		HAMMING_LOSS, EXMPF1_LOSS, EXMPACC_LOSS
	}
	
	public static void main(String[] args) {
		RndLocalSearchExperiment man = new RndLocalSearchExperiment();
		Execution.run(args, man); // add .class here if that class should receive command-line args
	}
	
	private void printParams() {
		
		System.out.println("//////////////////////////////////////");
		System.out.println(" Name = " + name.toString());
		System.out.println(" cfgPath = " + cfgPath);
		System.out.println(" logsPath = " + logsPath);
		System.out.println(" svmModelPath = " + svmModelPath);
		System.out.println(" InitType = " + startyp.toString());
		//System.out.println(" Restart = " + restart);
		System.out.println(" Train Restart  = " + restartTrain);
		System.out.println(" Test Restart = " + restartTest);
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
		
		// common dataset loader
		CommonDatasetLoader commonDsLdr = new CommonDatasetLoader(runOnDev, CommonDatasetLoader.DEFAULT_TRAIN_SPLIT_RATE);

		DataSetName ds = name;
		
		eArgs = new ElearningArg();
		eArgs.runEvalLearn = runEvalLearn;
		eArgs.elearningIter = eIter;
		eArgs.considerInstWght = useInstWght;
		eArgs.doEvalTest = doEvalTest;//runEvalLearn;
		eArgs.restartNumTest = evalTestRestartNum;
		
		
		String fdr = "../CacheCost";
		costCacher = new CostFuncCacherAndLoader(fdr);
		CostFuncCacherAndLoader.cacheCostWeight = doCostCache;
		
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
		
		// check whether the training and testing restarts are the same
		checkRestartNumber(restartTrain, restartTest);
		
		// have a look at params
		printParams();
		
		
		// Sequence
		if (ds == DataSetName.HW_SMALL) {
			runHandWritingSmall(commonDsLdr);
		} else if (ds == DataSetName.HW_LARGE) {
			runHandWritingLarge(commonDsLdr);

		} else if (ds == DataSetName.HW_LARGE_FULL) {
			runHwLargeFull(commonDsLdr);
		} else if (ds == DataSetName.HW_SMALL_FULL) {
			runHwSmallFull(commonDsLdr);
			
		} else if (ds == DataSetName.NETTALK_STREE) {
			runNetTalkStree(commonDsLdr);
		} else if (ds == DataSetName.NETTALK_PHONEME) {
			runNetTalkPhoneme(commonDsLdr);

			
		// Multi-label
		} else if (ds == DataSetName.YEAST) {
			runMulLbData("yeast", commonDsLdr);
		} else if (ds == DataSetName.ENRON) {
			runMulLbData("enron", commonDsLdr);
		} else if (ds == DataSetName.CAL500) {
			runMulLbData("CAL500", commonDsLdr);
		} else if (ds == DataSetName.COREL5K) {
			runMulLbData("Corel5k", commonDsLdr);
		} else if (ds == DataSetName.MEDIAMILL) {
			runMulLbData("mediamill", commonDsLdr);
			
			
		// Large Multi-label
		} else if (ds == DataSetName.BIBTEX) {
			runMulLbData("bibtex", commonDsLdr);
		} else if (ds == DataSetName.BOOKMARKS) {
			//runMulLbData("bookmarks-umass");
			runBookmarksData("bookmarks-umass", commonDsLdr);
		} else if (ds == DataSetName.DELICIOUS) {
			runMulLbData("delicious", commonDsLdr);
			
			
		// Image Segmentation
		} else if (ds == DataSetName.MSRC21) {
			runMSRC21(commonDsLdr);
		} else if (ds == DataSetName.HORSE32) {
			//
		} else if (ds == DataSetName.HORSE128) {
			//
			
		// Coreference Resolution
		} else if (ds == DataSetName.ACE05) {
			//
			
		} else {
			throw new RuntimeException("Unknown dataset name ...");
		}

		System.out.println("Done.");
	}
	
	public static void checkRestartNumber(int restartTrn, int restartTst) {
		if (restartTrn != restartTst) {
			System.err.println("[WARNING] restartTrain and restartTest are not equal: " + restartTrn + " != " + restartTst);
		}
	}
	
	public static void runMulLbData(String datasetName, CommonDatasetLoader commonDsLdr) {

		String dsName = datasetName;
		
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

		try {
			MultiLabelNew.runLearning(commonDsLdr, dsName, startyp, mlType, restartTrain, restartTest, initAlpha, cfgPath, logsPath, svmPath, eArgs, costCacher);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		System.out.println("MultiLabel All Done!");
	}
	
	public static void runBookmarksData(String datasetName, CommonDatasetLoader commonDsLdr) {

		String dsName = datasetName;
		
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

		try {
			
			InitType initgen = InitType.ALLZERO_INIT;
			//InitType initgen = InitType.UNIFORM_INIT;
			//InitType initgen = InitType.LOGISTIC_INIT;
			
			/*
			BookMarksMain.runLearning(dsName, initgen, optimizeLoss, 1, 
					    "../sl-config/"+dsName+"-search-DCD.config", 
					    //"../sl-config/"+dsName+"-perc.config", 
					    "../logistic_models/"+dsName+".logistic", 
					    "../logistic_models/"+dsName+".ssvm",
					    eLnArg, costchr);
			*/
			
			BookMarksMain.runLearning(commonDsLdr, dsName, initgen, mlType, restartTrain, restartTest, initAlpha, cfgPath, logsPath, svmPath, eArgs, costCacher);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		System.out.println("MultiLabel All Done!");
	}
	
	///////////////////////////////////////////
	///////////////////////////////////////////
	///////////////////////////////////////////

	private static int HW_FOLDER_INDEX = 0;

	public void runHandWritingSmall(CommonDatasetLoader commonDsLdr) { // isSmall
		
		String name = HandWritingMain.getDsName(true, HW_FOLDER_INDEX);
		
		try {
			HandWritingMain.runLearningOneFolder(commonDsLdr, true, HW_FOLDER_INDEX,
												 startyp, restartTrain, restartTest, initAlpha,
												 "../sl-config/hw-small-search-DCD.config", 
												 "../logistic_models/"+name+".logistic", 
												 "../logistic_models/"+name+".ssvm",
												 usePairFeat, useTernFeat, useQuadFeat, eArgs, costCacher);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void runHandWritingLarge(CommonDatasetLoader commonDsLdr) { // !isSmall
		
		String name = HandWritingMain.getDsName(false, HW_FOLDER_INDEX);
		
		try {
			HandWritingMain.runLearningOneFolder(commonDsLdr, false, HW_FOLDER_INDEX,
												 startyp, restartTrain, restartTest, initAlpha,
												 "../sl-config/hw-search-DCD.config", 
												 "../logistic_models/"+name+".logistic", 
												 "../logistic_models/"+name+".ssvm",
												 usePairFeat, useTernFeat, useQuadFeat, eArgs, costCacher);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void runNetTalkStree(CommonDatasetLoader commonDsLdr) {
		try {
			NetTalkStressMain.runLearning(commonDsLdr, startyp, restartTrain, restartTest, initAlpha,
					                      "../sl-config/stress-search-DCD.config", 
					                      "../logistic_models/nettalk_stress.logistic", 
					                      "../logistic_models/nettalk_stress.ssvm",
					                      usePairFeat, useTernFeat, useQuadFeat, eArgs, costCacher);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void runNetTalkPhoneme(CommonDatasetLoader commonDsLdr) {
		try {
			NetTalkPhonemeMain.runLearning(commonDsLdr, startyp, restartTrain, restartTest, initAlpha,
					                       "../sl-config/phoneme-search-DCD.config", 
					                       "../logistic_models/nettalk_phoneme.logistic", 
					                       "../logistic_models/nettalk_phoneme.ssvm",
					                       usePairFeat, useTernFeat, useQuadFeat, eArgs, costCacher);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void runMSRC21(CommonDatasetLoader commonDsLdr) {
		
		//CostFuncCacherAndLoader cCacher = new CostFuncCacherAndLoader(CostFuncCacherAndLoader.defaultFolder);
		//ElearningArg elArg = new ElearningArg();
		//elArg.runEvalLearn = false;
		
		ImageSegLabel[] labels = ImageSegLabel.loadLabelFromFile("../msrc/imageseg_label_color_map.txt");
		String[] labelNames = ImageSegLabel.getStrLabelArr(labels, false);
		String[] labelNamesFull = ImageSegLabel.getStrLabelArr(labels, true);
		
		ImageSegEvaluator.initRgbToLabel(labels);

		//ImageDataReader reader = new ImageDataReader("../msrc");
		//ImageSegEvaluator evaluator = new ImageSegEvaluator(reader.getDebugFolder());
		ImageSegEvaluator evaluator = new ImageSegEvaluator(commonDsLdr.getReaderDbgFolder());
		
		
		//List<ImageInstance> trainInsts = ImageSegMain.loadFromListFile(reader, "../msrc/Train.txt", labelNames, true);		
		//List<ImageInstance> testInsts = ImageSegMain.loadFromListFile(reader, "../msrc/Test.txt", labelNamesFull, true);
		
		List<List<ImageInstance>> trainPredImages = commonDsLdr.getMSRC21TrainPredictInstances(labelNames, true, labelNamesFull, true);
		List<ImageInstance> trainInsts = trainPredImages.get(0);
		List<ImageInstance> testInsts = trainPredImages.get(1);

		String svmCfgFile = "../sl-config/msrc21-search-DCD.config";
		String modelLogsFn = "../logistic_models/msrc21.logistic";
		String modelSvFn = "../logistic_models/msrc21.ssvm";
		
		try {
			
			SLModel slmodel = ImageSegMain.runLearning(trainInsts, testInsts, labelNames, startyp, restartTrain, restartTest, initAlpha, svmCfgFile, modelLogsFn, modelSvFn, usePairFeat, useTernFeat, useQuadFeat, evaluator, labels,  eArgs, costCacher);
			
			//SLModel slmodel = ImageSegMain.runLearning(trainInsts, testInsts, labelNames, InitType.UNIFORM_INIT, 1, svmCfgFile, modelLogsFn, modelSvFn, true, true, true, evaluator, labels, elArg, cCacher);
			//SLModel slmodel = runLearning(testInsts, testInsts, labelNames, InitType.LOGISTIC_INIT, 1, svmCfgFile, modelLogsFn, modelSvFn, true, true, true, evaluator, labels, elArg, cCacher);

			evaluator.evaluate(testInsts, slmodel, labels, true, -1);
		
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		

	}
	
	//////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////
	
	public void runHwSmallFull(CommonDatasetLoader commonDsLdr) { // isSmall
		
		try {
			HandWritingMain.runAllFolderLearning(commonDsLdr, true,// HW_FOLDER_INDEX,
												 startyp, restartTrain, restartTest, initAlpha,
												 "../sl-config/hw-small-search-DCD.config", 
												 //"../logistic_models/"+name+".logistic", 
												 //"../logistic_models/"+name+".ssvm",
												 usePairFeat, useTernFeat, useQuadFeat, eArgs, costCacher);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void runHwLargeFull(CommonDatasetLoader commonDsLdr) { // !isSmall
		
		try {
			HandWritingMain.runAllFolderLearning(commonDsLdr, false, //HW_FOLDER_INDEX,
												 startyp, restartTrain, restartTest, initAlpha,
												 "../sl-config/hw-search-DCD.config", 
												 //"../logistic_models/"+name+".logistic", 
												 //"../logistic_models/"+name+".ssvm",
												 usePairFeat, useTernFeat, useQuadFeat, eArgs, costCacher);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}


}
