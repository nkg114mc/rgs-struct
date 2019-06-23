package horse;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import edu.berkeley.nlp.futile.util.Counter;
import edu.illinois.cs.cogcomp.sl.core.SLModel;
import edu.illinois.cs.cogcomp.sl.core.SLParameters;
import edu.illinois.cs.cogcomp.sl.core.SLProblem;
import edu.illinois.cs.cogcomp.sl.core.SLParameters.LearningModelType;
import edu.illinois.cs.cogcomp.sl.learner.Learner;
import edu.illinois.cs.cogcomp.sl.learner.LearnerFactory;
import edu.illinois.cs.cogcomp.sl.learner.l2_loss_svm.L2LossSSVMLearner;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearning.AbstractRegressionLearner;
import elearning.AlterElearning;
import elearning.EFunctionLearning;
import elearning.EInferencer;
import elearning.EfuncInferenceDec2017;
import elearning.EfuncInferenceJuly2017;
import elearning.ElearningArg;
import elearning.SGDRegressionLearner;
import elearning.XgbRegressionLearner;
import elearning.LowLevelCostLearning.StopType;
import elearning.einfer.ESearchInferencer;
import elearnnew.DummyELearning;
import elearnnew.SamplingELearning;
import experiment.CostFuncCacherAndLoader;
import experiment.ExperimentResult;
import experiment.OneTestingResult;
import experiment.RndLocalSearchExperiment.InitType;
import experiment.RndLocalSearchExperiment.MulLbLossType;
import general.AbstractActionGenerator;
import general.AbstractLossFunction;
import general.FactorGraphBuilder.FactorGraphType;
import imgseg.FracScore;
import init.ImageSegAlphaGenerator;
import init.ImageSegSamplingRndGenerator;
import init.RandomStateGenerator;
import init.UniformRndGenerator;
import multilabel.MultiLabelFeaturizer;
import search.GreedySearcher;
import search.SeachActionGenerator;
import search.ZobristKeys;
import search.loss.SearchLossHamming;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;
import sequence.hw.HwSearchInferencer;
import sequence.hw.HwSegment;

public class Horse32Main {

	public static void main(String[] args) {
		/*
		try {
			
			CostFuncCacherAndLoader cCacher = new CostFuncCacherAndLoader(CostFuncCacherAndLoader.defaultFolder);
			ElearningArg elArg = new ElearningArg();

			ImageDataReader reader = new ImageDataReader("../msrc");
			ImageSegEvaluator evaluator = new ImageSegEvaluator(reader.getDebugFolder());

			List<ImageInstance> trainInsts = loadFromListFile(reader, "../msrc/Train.txt", labelNames, true);
			List<ImageInstance> testInsts = loadFromListFile(reader, "../msrc/Test.txt", labelNamesFull, true);

			String svmCfgFile = "sl-config/msrc21-search-DCD.config";
			String modelLogsFn = "../logistic_models/msrc21.logistic";
			String modelSvFn = "../logistic_models/msrc21.ssvm";
			
			SLModel slmodel = runLearning(trainInsts, testInsts, labelNames, InitType.UNIFORM_INIT, 1, 1, -1, svmCfgFile, modelLogsFn, modelSvFn, true, true, true, evaluator, labels, elArg, cCacher);
			//SLModel slmodel = runLearning(testInsts, testInsts, labelNames, InitType.LOGISTIC_INIT, 1, svmCfgFile, modelLogsFn, modelSvFn, true, true, true, evaluator, labels, elArg, cCacher);

			//evaluator.evaluate(testInsts, slmodel, labels, true, -1);

		} catch (Exception e) {
			e.printStackTrace();
		}*/
	}
	

/*
	public static SLModel runLearning(List<Horse32Instance> trainInsts, List<Horse32Instance> testInsts, String[] labelNames, 
									  InitType initType, int restartTrain, int restartTest, double iniAlfa, 
			                          String svmCfgFile, String modelLogsFn, String modelSvFn,
			                          boolean usePairFeat, boolean useTernFeat, boolean useQuadFeat, 
			                          Horse32Evaluator evaluator,
			                          ElearningArg evalLearnArg, CostFuncCacherAndLoader costCacher) throws Exception {
		String configFilePath = svmCfgFile;
		String dsName = "msrc21";
		SLProblem spTrain = Horse32Reader.ExampleListToSLProblem(trainInsts);
		
		RandomStateGenerator initStateGener = null;
		if (initType == InitType.UNIFORM_INIT) {
			initStateGener = new UniformRndGenerator(new Random());
		} else if (initType == InitType.LOGISTIC_INIT) {
			initStateGener = ImageSegSamplingRndGenerator.loadGenrIfExist(modelLogsFn, dsName, trainInsts, testInsts, labelNames, false, -1);
		} else if (initType == InitType.ALPHA_INIT) {
			ImageSegSamplingRndGenerator tmpGenr = ImageSegSamplingRndGenerator.loadGenrIfExist(modelLogsFn, dsName, trainInsts, testInsts, labelNames, false, -1);
			initStateGener = new ImageSegAlphaGenerator(tmpGenr.getDomainSize(), tmpGenr.getUnaryWght(), iniAlfa);
		} else {
			throw new RuntimeException("Unknown init type: " + initType);
		}
		System.out.println("=======");
		
		AbstractLossFunction lossfunc = new SearchLossHamming();
		

		//////////////////////////////////////////////////////////////////////
		// train
		SLModel model = new SLModel();

		// initialize the inference solver
		ZobristKeys abkeys = new ZobristKeys(2000, labelNames.length + 1);
		Horse32Featurizer fg = new Horse32Featurizer(labelNames, usePairFeat, useTernFeat);
		AbstractActionGenerator actGener = new SeachActionGenerator();

		GreedySearcher searcher = new GreedySearcher(FactorGraphType.ImageSegGraph, fg, restartTrain, actGener, initStateGener, lossfunc, abkeys);
		model.infSolver = new HwSearchInferencer(searcher);
		model.featureGenerator = fg;

		SLParameters para = new SLParameters();
		para.loadConfigFile(configFilePath);
		para.TOTAL_NUMBER_FEATURE = fg.getFeatLen();

		Learner learner = LearnerFactory.getLearner(model.infSolver, fg, para);
		//model.wv = learner.train(spTrain);
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
		
		model.config =  new HashMap<String, String>();
		//model.saveModel(modelSvFn);
		
		//evaluator.evaluate(testInsts, model, labels, false, 1);
		//evaluator.evaluate(testInsts, model, labels, false, 10);
		//evaluator.evaluate(testInsts, model, labels, false, 20);
		//evaluator.evaluate(testInsts, model, labels, false, 50);
		//evaluator.evaluate(testInsts, model, labels, false, randomNum);
		averageMultiRunMSRC21(evaluator, testInsts, model, labels, restartTest, evalLearnArg.multiRunTesting);
		
		////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////
		
		List<HwInstance> trnHwInsts = imageInstToHwInst(trainInsts);
		List<HwInstance> tstHwInsts = imageInstToHwInst(testInsts);
		
		// about local optimal?
		SamplingELearning.exploreLocalOptimal(initStateGener, tstHwInsts, model.wv, searcher, 20, null);
		
		return model;
	}
*/
}
