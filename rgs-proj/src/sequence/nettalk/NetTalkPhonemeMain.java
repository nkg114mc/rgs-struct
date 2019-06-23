package sequence.nettalk;

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
import elearning.EFunctionLearning;
import elearning.EInferencer;
import elearning.EfuncInferenceDec2017;
import elearning.ElearningArg;
import elearning.XgbRegressionLearner;
import elearning.einfer.ESearchInferencer;
import elearnnew.DummyELearning;
import elearnnew.SamplingELearning;
import experiment.CommonDatasetLoader;
import experiment.CostFuncCacherAndLoader;
import experiment.ExperimentResult;
import experiment.RndLocalSearchExperiment.InitType;
import experiment.RndLocalSearchExperiment.MulLbLossType;
import general.AbstractActionGenerator;
import general.AbstractLossFunction;
import general.FactorGraphBuilder.FactorGraphType;
import init.RandomStateGenerator;
import init.SeqAlphaGenerator;
import init.SeqSamplingRndGenerator;
import init.UniformRndGenerator;
import multilabel.MultiLabelFeaturizer;
import search.GreedySearcher;
import search.SeachActionGenerator;
import search.ZobristKeys;
import search.loss.SearchLossHamming;
import sequence.hw.HandWritingMain;
import sequence.hw.HwDataReader;
import sequence.hw.HwFeaturizer;
import sequence.hw.HwInstance;
import sequence.hw.HwSearchInferencer;

public class NetTalkPhonemeMain {
	
	public static void main(String[] args) {
		try {
			ElearningArg eLnArg = new ElearningArg();
			CostFuncCacherAndLoader costchr = new CostFuncCacherAndLoader(CostFuncCacherAndLoader.defaultFolder);
			//runLearning(InitType.LOGISTIC_INIT, 1, "sl-config/phoneme-search-DCD.config", "../logistic_models/nettalk_phoneme.logistic", "../logistic_models/nettalk_phoneme.ssvm");
			CommonDatasetLoader commonDsLdr = new CommonDatasetLoader();
			runLearning(commonDsLdr, InitType.UNIFORM_INIT, 1, 1, -1,
					    "../sl-config/phoneme-search-DCD.config", 
					    "../logistic_models/nettalk_phoneme.logistic", 
					    "../logistic_models/nettalk_phoneme.ssvm",
					    true, true, true, eLnArg, costchr);//true, true, true);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	
	public static void runLearning(CommonDatasetLoader commonDsLdr, 
								   //InitType initType, int randomNum, double iniAlfa,
								   InitType initType, int restartTrain, int restartTest, double iniAlfa,
			                       String svmCfgFile, String modelLogsFn, String modelSvFn,
			                       boolean usePairFeat, boolean useTernFeat, boolean useQuadFeat, ElearningArg evalLearnArg, CostFuncCacherAndLoader costCacher) throws Exception {

		String configFilePath = svmCfgFile;
		//int nrnd = randomNum;
		
		
		// load data
		//NtkPhonemeLabelSet phLabels = new NtkPhonemeLabelSet();
		//NtkDataReader rder = new NtkDataReader();
		//List<List<HwInstance>> trtstInsts = rder.readData("../datasets/nettalk_phoneme_train.txt", "../datasets/nettalk_phoneme_test.txt", phLabels.getLabels());
		
		NtkPhonemeLabelSet phLabels = (NtkPhonemeLabelSet) commonDsLdr.getCommonLabelSet("nettalk_phoneme");
		List<List<HwInstance>> trtstInsts = commonDsLdr.getNetPhonemeDs();
		
		
		System.out.println("List count = " + trtstInsts.size());
		
		RandomStateGenerator initStateGener = null;
		if (initType == InitType.UNIFORM_INIT) {
			initStateGener = new UniformRndGenerator(new Random());
		} else if (initType == InitType.LOGISTIC_INIT) {
			initStateGener = SeqSamplingRndGenerator.loadGenrIfExist(modelLogsFn, "nettalk_phoneme", trtstInsts.get(0), trtstInsts.get(1), phLabels.getLabels(), false, -1);
		} else if (initType == InitType.ALPHA_INIT) {
			SeqSamplingRndGenerator tmpGenr = SeqSamplingRndGenerator.loadGenrIfExist(modelLogsFn, "nettalk_phoneme", trtstInsts.get(0), trtstInsts.get(1), phLabels.getLabels(), false, -1);
			initStateGener = new SeqAlphaGenerator(tmpGenr.getDomainSize(), tmpGenr.getWkInstHeader(), tmpGenr.getLogisticModel(), tmpGenr.getRandom(), iniAlfa);
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

		GreedySearcher searcher = new GreedySearcher(FactorGraphType.SequenceGraph, fg, restartTrain, actGener,initStateGener, lossfunc, abkeys);
		model.infSolver = new HwSearchInferencer(searcher);
		//model.infSolver = new HwSearchInferencer(fg,1,abkeys);
		//model.infSolver = new HwViterbiInferencer(fg);
		model.featureGenerator = fg;

		SLParameters para = new SLParameters();
		para.loadConfigFile(configFilePath);
		para.TOTAL_NUMBER_FEATURE = fg.getFeatLen();

		Learner learner = LearnerFactory.getLearner(model.infSolver, fg, para);
		//model.wv = learner.train(spTrain);
		if (CostFuncCacherAndLoader.cacheCostWeight) {
			WeightVector loadedWv = costCacher.loadCachedWeight("nettalk_phoneme", initType, restartTrain, CostFuncCacherAndLoader.getFeatDim(usePairFeat, useTernFeat, useQuadFeat), MulLbLossType.HAMMING_LOSS, para.C_FOR_STRUCTURE,iniAlfa);
			if (loadedWv != null) { // load failure...
				model.wv = loadedWv;
			} else {
				model.wv = learner.train(spTrain);
				costCacher.saveCachedWeight(model.wv, "nettalk_phoneme", initType, restartTrain, CostFuncCacherAndLoader.getFeatDim(usePairFeat, useTernFeat, useQuadFeat), MulLbLossType.HAMMING_LOSS, para.C_FOR_STRUCTURE,iniAlfa); // save
			}
		} else {
			model.wv = learner.train(spTrain);
		}
		model.config =  new HashMap<String, String>();

		// test
		//////////////////////////////////
		SLProblem spTest = slproblems.get(1);
		searcher.setRestart(restartTest);
		//HandWritingMain.evaluate(spTest, model);
		HandWritingMain.averageOfMultiRunEvaluation(spTest, model, evalLearnArg.multiRunTesting); 
		System.out.println("Done.");

		// about local optimal?
		SamplingELearning.exploreLocalOptimal(initStateGener, trtstInsts.get(1), model.wv, searcher, 20, null);

	}
	
}
