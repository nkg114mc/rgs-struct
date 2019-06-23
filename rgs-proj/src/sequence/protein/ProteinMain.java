package sequence.protein;

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

public class ProteinMain {
	
	public static void main(String[] args) {
		try {
			ElearningArg eLnArg = new ElearningArg();
			CostFuncCacherAndLoader costchr = new CostFuncCacherAndLoader(CostFuncCacherAndLoader.defaultFolder);
			CommonDatasetLoader commonDsLdr = new CommonDatasetLoader();
			runLearning(commonDsLdr, InitType.LOGISTIC_INIT, 1, 100, -1,
					    "../sl-config/protein-search-DCD.config", 
					    "../logistic_models/protein.logistic", 
					    "../logistic_models/protein.ssvm",
					    true, true, false, eLnArg, costchr);//true, true, true);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void countSize(List<HwInstance> list) {
		int maxSize = 0;
		int minSize = Integer.MAX_VALUE;
		for (HwInstance ins : list) {
			maxSize = Math.max(maxSize, ins.size());
			minSize = Math.min(minSize, ins.size());
		}
		System.out.println("MinSize = " + minSize);
		System.out.println("MaxSize = " + maxSize);
	}
	
	public static void runLearning(CommonDatasetLoader commonDsLdr,
								   InitType initType, int restartTrain, int restartTest, double iniAlfa,
			                       String svmCfgFile, String modelLogsFn, String modelSvFn,
			                       boolean usePairFeat, boolean useTernFeat, boolean useQuadFeat, ElearningArg evalLearnArg, CostFuncCacherAndLoader costCacher) throws Exception {

		String configFilePath = svmCfgFile;
		
		// load data
		ProteinLabelSet phLabels = (ProteinLabelSet) commonDsLdr.getCommonLabelSet("protein");
		List<List<HwInstance>> trtstInsts = commonDsLdr.getProteinDs();
		
		
		System.out.println("List count = " + trtstInsts.size());
		countSize(trtstInsts.get(0));
		countSize(trtstInsts.get(1));
		
		RandomStateGenerator initStateGener = null;
		if (initType == InitType.UNIFORM_INIT) {
			initStateGener = new UniformRndGenerator(new Random());
		} else if (initType == InitType.LOGISTIC_INIT) {
			initStateGener = SeqSamplingRndGenerator.loadGenrIfExist(modelLogsFn, "protein", trtstInsts.get(0), trtstInsts.get(1), phLabels.getLabels(), false, -1);
		} else if (initType == InitType.ALPHA_INIT) {
			SeqSamplingRndGenerator tmpGenr = SeqSamplingRndGenerator.loadGenrIfExist(modelLogsFn, "protein", trtstInsts.get(0), trtstInsts.get(1), phLabels.getLabels(), false, -1);
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
		HwFeaturizer fg = new HwFeaturizer(phLabels.getLabels(), ProteinDataReader.PROTEIN_FEATURE_LENGTH, usePairFeat, useTernFeat, useQuadFeat);//true, true, true);

		GreedySearcher searcher = new GreedySearcher(FactorGraphType.SequenceGraph, fg, restartTrain, actGener,initStateGener, lossfunc, abkeys);
		model.infSolver = new HwSearchInferencer(searcher);
		model.featureGenerator = fg;

		SLParameters para = new SLParameters();
		para.loadConfigFile(configFilePath);
		para.TOTAL_NUMBER_FEATURE = fg.getFeatLen();

		Learner learner = LearnerFactory.getLearner(model.infSolver, fg, para);
		if (CostFuncCacherAndLoader.cacheCostWeight) {
			WeightVector loadedWv = costCacher.loadCachedWeight("protein", initType, restartTrain, CostFuncCacherAndLoader.getFeatDim(usePairFeat, useTernFeat, useQuadFeat), MulLbLossType.HAMMING_LOSS, para.C_FOR_STRUCTURE,iniAlfa);
			if (loadedWv != null) { // load failure...
				model.wv = loadedWv;
			} else {
				model.wv = learner.train(spTrain);
				costCacher.saveCachedWeight(model.wv, "protein", initType, restartTrain, CostFuncCacherAndLoader.getFeatDim(usePairFeat, useTernFeat, useQuadFeat), MulLbLossType.HAMMING_LOSS, para.C_FOR_STRUCTURE,iniAlfa); // save
			}
		} else {
			model.wv = learner.train(spTrain);
		}
		model.config =  new HashMap<String, String>();

		// test
		//////////////////////////////////
		SLProblem spTest = slproblems.get(1);
		searcher.setRestart(restartTest);
		HandWritingMain.averageOfMultiRunEvaluation(spTest, model, evalLearnArg.multiRunTesting); 
		System.out.println("Done.");

		// about local optimal?
		SamplingELearning.exploreLocalOptimal(initStateGener, trtstInsts.get(1), model.wv, searcher, 20, null);

	}
	
}
