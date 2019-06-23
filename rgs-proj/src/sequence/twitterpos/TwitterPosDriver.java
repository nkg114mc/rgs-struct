package sequence.twitterpos;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import edu.illinois.cs.cogcomp.sl.core.SLModel;
import edu.illinois.cs.cogcomp.sl.core.SLParameters;
import edu.illinois.cs.cogcomp.sl.core.SLProblem;
import edu.illinois.cs.cogcomp.sl.learner.Learner;
import edu.illinois.cs.cogcomp.sl.learner.LearnerFactory;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearning.AbstractRegressionLearner;
import elearning.EInferencer;
import elearning.EfuncInferenceDec2017;
import elearning.ElearningArg;
import elearning.XgbRegressionLearner;
import elearning.einfer.ESearchInferencer;
import elearnnew.SamplingELearning;
import experiment.CommonDatasetLoader;
import experiment.CostFuncCacherAndLoader;
import experiment.ExperimentResult;
import experiment.OneTestingResult;
import experiment.TestingAcc;
import experiment.RndLocalSearchExperiment.InitType;
import experiment.RndLocalSearchExperiment.MulLbLossType;
import general.AbstractActionGenerator;
import general.AbstractInstance;
import general.AbstractLossFunction;
import general.AbstractOutput;
import general.FactorGraphBuilder.FactorGraphType;
import init.RandomStateGenerator;
import init.SeqAlphaGenerator;
import init.SeqSamplingRndGenerator;
import init.UniformRndGenerator;
import search.GreedySearcher;
import search.SeachActionGenerator;
import search.SearchResult;
import search.ZobristKeys;
import search.loss.GoldPredPair;
import search.loss.SearchLossHamming;
import sequence.hw.HandWritingMain;
import sequence.hw.HwDataReader;
import sequence.hw.HwFeaturizer;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;
import sequence.hw.HwSearchInferencer;
import weka.classifiers.functions.Logistic;
import weka.core.Instances;

public class TwitterPosDriver {
	
	public static void main(String[] args) {
		try {
			ElearningArg eLnArg = new ElearningArg();
			CostFuncCacherAndLoader costchr = new CostFuncCacherAndLoader(CostFuncCacherAndLoader.defaultFolder);
			CommonDatasetLoader commonDsLdr = new CommonDatasetLoader();
			runLearning(commonDsLdr, InitType.UNIFORM_INIT, 1, 50, -1,
					"../sl-config/twitterpos-search-DCD.config", "../logistic_models/twitterpos.logistic", "../logistic_models/twitterpos.ssvm",
					true, true, false, eLnArg, costchr);//true, true, true);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void runLearning(CommonDatasetLoader commonDsLdr,
								   InitType initType, int restartTrain, int restartTest,  double iniAlfa,
			                       String svmCfgFile, String modelLogsFn, String modelSvFn,
			                       boolean usePairFeat, boolean useTernFeat, boolean useQuadFeat, ElearningArg evalLearnArg, CostFuncCacherAndLoader costCacher) throws Exception {
		
		String configFilePath = svmCfgFile;

		// load data
		TwitterPosLabelSet stLabels = (TwitterPosLabelSet) commonDsLdr.getCommonLabelSet("twitterpos");
		List<List<HwInstance>> trtstInsts = commonDsLdr.getTwitterPosDs();
		
		System.out.println("List count = " + trtstInsts.size());
		List<SLProblem> slproblems = HwDataReader.convertToSLProblem(trtstInsts);
		
		RandomStateGenerator initStateGener = null;
		if (initType == InitType.UNIFORM_INIT) {
			initStateGener = new UniformRndGenerator(new Random());
		} else if (initType == InitType.LOGISTIC_INIT) {
			initStateGener = SeqSamplingRndGenerator.loadGenrIfExist(modelLogsFn, "twitterpos", trtstInsts.get(0), trtstInsts.get(1), stLabels.getLabels(), false, -1);
		} else if (initType == InitType.ALPHA_INIT) {
			SeqSamplingRndGenerator tmpGenr = SeqSamplingRndGenerator.loadGenrIfExist(modelLogsFn, "twitterpos", trtstInsts.get(0), trtstInsts.get(1), stLabels.getLabels(), false, -1);
			initStateGener = new SeqAlphaGenerator(tmpGenr.getDomainSize(), tmpGenr.getWkInstHeader(), tmpGenr.getLogisticModel(), tmpGenr.getRandom(), iniAlfa);
		}
		System.out.println("=======");
		
		AbstractLossFunction lossfunc = new SearchLossHamming();

		//////////////////////////////////////////////////////////////////////
		// train
		SLModel model = new SLModel();
		SLProblem spTrain = slproblems.get(0);

		// initialize the inference solver
		ZobristKeys abkeys = new ZobristKeys(200, stLabels.getLabels().length);
		HwFeaturizer fg = new HwFeaturizer(stLabels.getLabels(), TwitterPosFeaturizer.LSTMStateLen, usePairFeat, useTernFeat, useQuadFeat);//true, true, true);
		AbstractActionGenerator actGener = new SeachActionGenerator();
		
		GreedySearcher searcher = new GreedySearcher(FactorGraphType.SequenceGraph, fg, restartTrain, actGener, initStateGener, lossfunc, abkeys);
		model.infSolver = new HwSearchInferencer(searcher);
		model.featureGenerator = fg;

		SLParameters para = new SLParameters();
		para.loadConfigFile(configFilePath);
		para.TOTAL_NUMBER_FEATURE = fg.getFeatLen();
		System.err.println("C = " + para.C_FOR_STRUCTURE);
		
	
		Learner learner = LearnerFactory.getLearner(model.infSolver, fg, para);
		if (CostFuncCacherAndLoader.cacheCostWeight) {
			WeightVector loadedWv = costCacher.loadCachedWeight("twitterpos", initType, restartTrain, CostFuncCacherAndLoader.getFeatDim(usePairFeat, useTernFeat, useQuadFeat), MulLbLossType.HAMMING_LOSS, para.C_FOR_STRUCTURE, iniAlfa);
			if (loadedWv != null) { // load failure...
				model.wv = loadedWv;
			} else {
				model.wv = learner.train(spTrain);
				costCacher.saveCachedWeight(model.wv, "twitterpos", initType, restartTrain, CostFuncCacherAndLoader.getFeatDim(usePairFeat, useTernFeat, useQuadFeat), MulLbLossType.HAMMING_LOSS, para.C_FOR_STRUCTURE, iniAlfa); // save
			}
		} else {
			model.wv = learner.train(spTrain);
		}
		model.config =  new HashMap<String, String>();

		// test
		//////////////////////////////////
		SLProblem spTest = slproblems.get(1);
		searcher.setRestart(restartTest);
		HandWritingMain.evaluate(spTest, model);
		//HandWritingMain.averageOfMultiRunEvaluation(spTest, model, evalLearnArg.multiRunTesting); 
		System.out.println("Done.");
		
		// about local optimal?
		//SamplingELearning.exploreLocalOptimal(initStateGener, trtstInsts.get(1), model.wv, searcher, 20, null);
		
	}

}
