package regressioncost;


import java.util.List;
import java.util.Random;

import edu.illinois.cs.cogcomp.sl.core.SLModel;
import edu.illinois.cs.cogcomp.sl.core.SLParameters;
import edu.illinois.cs.cogcomp.sl.core.SLProblem;
import edu.illinois.cs.cogcomp.sl.learner.Learner;
import edu.illinois.cs.cogcomp.sl.learner.LearnerFactory;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearning.AbstractRegressionLearner;
import elearning.ElearningArg;
import elearning.XgbRegressionLearner;
import elearning.einfer.SearchStateScoringFunction;
import elearnnew.SamplingELearning;
import experiment.CommonDatasetLoader;
import experiment.CostFuncCacherAndLoader;
import experiment.ExperimentResult;
import experiment.RndLocalSearchExperiment.DataSetName;
import experiment.RndLocalSearchExperiment.InitType;
import experiment.RndLocalSearchExperiment.MulLbLossType;
import general.AbstractActionGenerator;
import general.AbstractInstance;
import general.AbstractLossFunction;
import general.FactorGraphBuilder.FactorGraphType;
import init.HwFastAlphaGenerator;
import init.HwFastSamplingRndGenerator;
import init.RandomStateGenerator;
import init.UniformRndGenerator;
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

public class HandWritingRegressionRerank {

	/**
	 * Hand-Writing Recognition Problem
	 * 
	 * Chao Ma
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		try {
			runOneFolder(false, 0);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void runOneFolder(boolean isSmall, int df) {
		
		CommonDatasetLoader commonDsLdr = new CommonDatasetLoader(false, CommonDatasetLoader.DEFAULT_TRAIN_SPLIT_RATE);
		
		ElearningArg eArgs = new ElearningArg();
		eArgs.runEvalLearn = false;
		eArgs.elearningIter = 1;
		eArgs.doEvalTest =  false;
		eArgs.restartNumTest = 10;
		
		
		String fdr = "../CacheCost";
		CostFuncCacherAndLoader costCacher = new CostFuncCacherAndLoader(fdr);

		
		//String modelLogsFn = "../logistic_models/"+fdName+".logistic"; 
		//String modelSvFn = "../logistic_models/"+fdName+".ssvm";
		
		String fdName = getDsName(isSmall, df);
		System.out.println("///////////// Start " + fdName + " Learning /////////////////");
		
		String modelLogsFn = "../logistic_models/"+fdName+".logistic"; 
		String modelSvFn = "../logistic_models/"+fdName+".ssvm";
		String svmCfgFile = "../sl-config/hw-search-DCD.config";

		try {
			ExperimentResult re = runLearningOneFolder(commonDsLdr, isSmall, df,
					InitType.UNIFORM_INIT, 10, 10, -1, svmCfgFile, modelLogsFn, modelSvFn,
					true, true, true, eArgs, costCacher);
		} catch (Exception e) {
			e.printStackTrace();
		}


		System.out.println("///////////// Finish " + fdName + " Learning /////////////////");
	}
	
	public static void runAllFolderLearning(CommonDatasetLoader commonDsLdr, boolean isSmall,
									        //InitType initType, int randomNum, double iniAlfa,
									        InitType initType, int restartTrain, int restartTest, double iniAlfa,
									        String svmCfgFile,
									        boolean usePairFeat, boolean useTernFeat, boolean useQuadFeat,
									        ElearningArg evalLearnArg, CostFuncCacherAndLoader costCacher) {
		
		try {
			
			double avgOverall = 0;
			double avgGenacc = 0;
			double[] accFolders = new double[10];
			double[] genFolders = new double[10];
			for (int df = 0; df < 10; df++) {
				String fdName = getDsName(isSmall, df);
				System.out.println("///////////// Start " + fdName + " Learning /////////////////");
				
				String modelLogsFn = "../logistic_models/"+fdName+".logistic"; 
				String modelSvFn = "../logistic_models/"+fdName+".ssvm";

				ExperimentResult re = runLearningOneFolder(commonDsLdr, isSmall, df,
						initType, restartTrain, restartTest, iniAlfa, svmCfgFile, modelLogsFn, modelSvFn,
						usePairFeat, useTernFeat, useQuadFeat,  evalLearnArg, costCacher);

				accFolders[df] = re.overallAcc;
				genFolders[df] = re.generationAcc;
				
				avgOverall += accFolders[df];
				avgGenacc += genFolders[df];
				
				System.out.println("///////////// Finish " + fdName + " Learning /////////////////");
			}
			
			avgOverall /= ((double)10.0);
			avgGenacc /= ((double)10.0);
			
			System.out.println("==== Average Over 10 Folders ====");
			for (int i = 0; i < 10; i++) {
				System.out.println(" Folder " + i + ": " + "Acc = " + accFolders[i] + " Gen = " + genFolders[i]);
			}
			System.out.println("Average-Accuracy = " + avgOverall);
			System.out.println("Average-Generation Acc = " + avgGenacc);
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println("Finish.");
	}
	
	public static String getDsName(boolean isSmall, int folderIndex) {
		String mn = "";
		if (isSmall) {
			mn = "hw-small";
		} else {
			mn = "hw-large";
		}
		String n = mn + "-" + String.valueOf(folderIndex);
		System.out.println("Get name: [" + n + "]");
		return n;
	}
	
	public static ExperimentResult runLearningOneFolder(CommonDatasetLoader commonDsLdr, boolean isSmall, int folderIndex,
			                                InitType initType, int restartTrain, int restartTest, double iniAlfa,
			                                String svmCfgFile, String modelLogsFn, String modelSvFn,
                                            boolean usePairFeat, boolean useTernFeat, boolean useQuadFeat, ElearningArg evalLearnArg, CostFuncCacherAndLoader costCacher) throws Exception {
		
		String dsName = getDsName(isSmall, folderIndex);
		String configFilePath = svmCfgFile;

		HwLabelSet hwLabels = (HwLabelSet) commonDsLdr.getCommonLabelSet(dsName);
		assert (hwLabels != null);
		List<List<HwInstance>> trtstInsts = commonDsLdr.getHwDs(isSmall, folderIndex);
		
		
		System.out.println("List count = " + trtstInsts.size());
		List<SLProblem> slproblems = HwDataReader.convertToSLProblem(trtstInsts);

		RandomStateGenerator initStateGener = null;
		if (initType == InitType.UNIFORM_INIT) {
			initStateGener = new UniformRndGenerator(new Random());
		} else if (initType == InitType.LOGISTIC_INIT) {
			initStateGener = HwFastSamplingRndGenerator.loadGenrIfExist(modelLogsFn, dsName, trtstInsts.get(0), trtstInsts.get(1), hwLabels.getLabels(), false, -1);
		} else if (initType == InitType.ALPHA_INIT) {
			HwFastSamplingRndGenerator tmpr = HwFastSamplingRndGenerator.loadGenrIfExist(modelLogsFn, dsName, trtstInsts.get(0), trtstInsts.get(1), hwLabels.getLabels(), false, -1);
			initStateGener = new HwFastAlphaGenerator(hwLabels.getLabels().length, tmpr.getUnaryWght(), iniAlfa); 
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

		GreedySearcher searcher = new GreedySearcher(FactorGraphType.SequenceGraph, fg, restartTrain, actGener, initStateGener, lossfunc, abkeys);
		model.infSolver = new HwSearchInferencer(searcher);
		model.featureGenerator = fg;

		SLParameters para = new SLParameters();
		para.loadConfigFile(configFilePath);
		para.TOTAL_NUMBER_FEATURE = fg.getFeatLen();

		Learner learner = LearnerFactory.getLearner(model.infSolver, fg, para);
		if (CostFuncCacherAndLoader.cacheCostWeight) {
			WeightVector loadedWv = costCacher.loadCachedWeight(dsName, initType, restartTrain, CostFuncCacherAndLoader.getFeatDim(usePairFeat, useTernFeat, useQuadFeat), MulLbLossType.HAMMING_LOSS, para.C_FOR_STRUCTURE, iniAlfa);
			if (loadedWv != null) { // load failure...
				model.wv = loadedWv;
			} else {
				model.wv = learner.train(spTrain);
				costCacher.saveCachedWeight(model.wv, dsName, initType, restartTrain, CostFuncCacherAndLoader.getFeatDim(usePairFeat, useTernFeat, useQuadFeat), MulLbLossType.HAMMING_LOSS, para.C_FOR_STRUCTURE, iniAlfa); // save
			}
		} else {
			model.wv = learner.train(spTrain);
		}
		

		// test
		//////////////////////////////////
		SLProblem spTest = slproblems.get(1);
		searcher.setRestart(restartTest);
		//ExperimentResult tstRe = HandWritingMain.evaluate(spTest, model);
		//ExperimentResult tstRe = HandWritingMain.averageOfMultiRunEvaluation(spTest, model, evalLearnArg.multiRunTesting); 
		


		// about local optimal?
		//SamplingELearning.exploreLocalOptimal(initStateGener, trtstInsts.get(1), model.wv, searcher, 20, null);
		
		
		////////////////////////////////////////////////////////////////////
		/// Regression as Cost /////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////

		AbstractRegressionLearner regressionTrainer = new XgbRegressionLearner("xgb");
		SearchStateScoringFunction rerankFunc = RegressionRerankLearning.regressionCostRerank(trtstInsts.get(0),
				 searcher,
				 model.wv,
				 restartTrain,
				 regressionTrainer);

		ExperimentResult tstRe = RegressionRerankLearning.evaluate(spTest, model, rerankFunc);
		
		System.out.println("Done.");
		return tstRe;
	}
	

}

