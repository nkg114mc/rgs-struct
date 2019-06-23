package imgcnn;

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
import elearning.ElearningArg;
import elearning.XgbRegressionLearner;
import elearning.einfer.ESearchInferencer;
import elearnnew.SamplingELearning;
import experiment.CostFuncCacherAndLoader;
import experiment.ExperimentResult;
import experiment.TestingAcc;
import experiment.RndLocalSearchExperiment.InitType;
import experiment.RndLocalSearchExperiment.MulLbLossType;
import general.AbstractActionGenerator;
import general.AbstractLossFunction;
import general.FactorGraphBuilder.FactorGraphType;
import imgseg.ImageInstance;
import imgseg.ImageSegEvaluator;
import imgseg.ImageSegLabel;
import imgseg.ImageSegMain;
import imgseg.ImageUnaryInferencer;
import init.ImageSegAlphaGenerator;
import init.ImageSegSamplingRndGenerator;
import init.RandomStateGenerator;
import init.UniformRndGenerator;
import search.GreedySearcher;
import search.SeachActionGenerator;
import search.SearchResult;
import search.ZobristKeys;
import search.loss.SearchLossHamming;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;
import sequence.hw.HwSearchInferencer;

public class CNNSuperPixelUnary {
	
	public static void main(String[] args) {
		unaryOnly(args);
		//TestUpperBound(args);
	}

	public static void TestUpperBound(String[] args) {
		try {
			
			ImageSegLabel[] labels = ImageSegLabel.loadLabelFromFile("../msrc/imageseg_label_color_map.txt");
			String[] labelNames = ImageSegLabel.getStrLabelArr(labels, false);
			String[] labelNamesFull = ImageSegLabel.getStrLabelArr(labels, true);
			
			ImageSegEvaluator.initRgbToLabel(labels);

			ImageCNNReader reader = new ImageCNNReader("../msrc");
			ImageSegEvaluator evaluator = new ImageSegEvaluator(reader.getDebugFolder());

			List<ImageInstance> trainInsts = ImageCNNMain.loadFromListFile(reader, "../msrc/Train.txt", labelNames, true);
			List<ImageInstance> testInsts = ImageCNNMain.loadFromListFile(reader, "../msrc/Test.txt", labelNamesFull, true);
			
			
			List<ImageInstance> totalInsts = new ArrayList<ImageInstance>();
			totalInsts.addAll(trainInsts);
			totalInsts.addAll(testInsts);
			
			ImageSegMain.computeAvgStructSize(trainInsts);

			
			
			String svmCfgFile = "sl-config/msrc21-search-DCD.config";
			String modelLogsFn = "../logistic_models/msrc21.logistic";
			String modelSvFn = "../logistic_models/msrc21.ssvm";

			//evaluator.evaluate(testInsts, slmodel, labels, true, -1);
			evaluator.evaluateSuperPixelGt(testInsts, labels, true);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	
	public static void unaryOnly(String[] args) {
		
		try {
			
			CostFuncCacherAndLoader cCacher = new CostFuncCacherAndLoader(CostFuncCacherAndLoader.defaultFolder);
			ElearningArg elArg = new ElearningArg();
			//elArg.runEvalLearn = false;
			
			ImageSegLabel[] labels = ImageSegLabel.loadLabelFromFile("../msrc/imageseg_label_color_map.txt");
			String[] labelNames = ImageSegLabel.getStrLabelArr(labels, false);
			//String[] labelNamesFull = ImageSegLabel.getStrLabelArr(labels, true);
			
			ImageSegEvaluator.initRgbToLabel(labels);

			ImageCNNReader reader = new ImageCNNReader("../msrc");
			ImageSegEvaluator evaluator = new ImageSegEvaluator(reader.getDebugFolder());


			//List<ImageInstance> trainInsts = loadFromListFile(reader, "../msrc/Train-small.txt", labelNames, true);
			List<ImageInstance> trainInsts = ImageCNNMain.loadFromListFile(reader, "../msrc/TrainValidation.txt", labelNames, true);
			
			List<ImageInstance> testInsts = ImageCNNMain.loadFromListFile(reader, "../msrc/Test.txt", labelNames,  true);
			
			
			List<ImageInstance> totalInsts = new ArrayList<ImageInstance>();
			totalInsts.addAll(trainInsts);
			totalInsts.addAll(testInsts);

			String svmCfgFile = "sl-config/msrc21-search-DCD.config";
			String modelLogsFn = "../logistic_models/msrc21.logistic";
			String modelSvFn = "../logistic_models/msrc21.ssvm";
			
			//SLModel slmodel = runLearningUnary(trainInsts, testInsts, labelNames, InitType.UNIFORM_INIT, 0, 1,1, svmCfgFile, modelLogsFn, modelSvFn, false, false, false, evaluator, labels, elArg, cCacher);
			SLModel slmodel = runLearningUnary(trainInsts, testInsts, labelNames, InitType.LOGISTIC_INIT, 0, 1, 1, svmCfgFile, modelLogsFn, modelSvFn, false, false, false, evaluator, labels, elArg, cCacher);

			//evaluateUnary(trainInsts, slmodel, labelNames, labels, evaluator, true);
			ImageSegMain.evaluateUnary(testInsts, slmodel, labelNames, labels, evaluator, true);
			

		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	
	
	public static SLModel runLearningUnary(List<ImageInstance> trainInsts, List<ImageInstance> testInsts, String[] labelNames, 
			InitType initType, int restartTrain, int restartTest, double iniAlfa, 
			String svmCfgFile, String modelLogsFn, String modelSvFn,
			boolean usePairFeat, boolean useTernFeat, boolean useQuadFeat, 
			ImageSegEvaluator evaluator, ImageSegLabel[] labels,
			ElearningArg evalLearnArg, CostFuncCacherAndLoader costCacher) throws Exception {
		
/*	
		
		String configFilePath = svmCfgFile;

		String dsName = "msrc21";
		SLProblem spTrain = ImageCNNReader.ExampleListToSLProblem(trainInsts);

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
		ZobristKeys abkeys = new ZobristKeys(1000, labelNames.length + 1);
		ImageCNNFeaturizer fg = new ImageCNNFeaturizer(labelNames, usePairFeat, useTernFeat);
		AbstractActionGenerator actGener = new SeachActionGenerator();

		GreedySearcher searcher = new GreedySearcher(FactorGraphType.ImageSegGraph, fg, restartTrain, actGener, initStateGener, lossfunc, abkeys);
		model.infSolver = new ImageUnaryInferencer(fg);
		model.featureGenerator = fg;

		SLParameters para = new SLParameters();
		para.loadConfigFile(configFilePath);
		para.TOTAL_NUMBER_FEATURE = fg.getFeatLen();

		Learner learner = LearnerFactory.getLearner(model.infSolver, fg, para);
		if (CostFuncCacherAndLoader.cacheCostWeight) {
			WeightVector loadedWv = costCacher.loadCachedWeight(dsName, initType, restartTrain, CostFuncCacherAndLoader.getFeatDim(usePairFeat, useTernFeat, useQuadFeat), MulLbLossType.HAMMING_LOSS, para.C_FOR_STRUCTURE, iniAlfa);
			if (loadedWv != null) {
				model.wv = loadedWv;
			} else {
				model.wv = learner.train(spTrain);
				costCacher.saveCachedWeight(model.wv, dsName, initType, restartTrain, CostFuncCacherAndLoader.getFeatDim(usePairFeat, useTernFeat, useQuadFeat), MulLbLossType.HAMMING_LOSS, para.C_FOR_STRUCTURE, iniAlfa); // save
			}
		} else {
			model.wv = learner.train(spTrain);
		}
		model.config =  new HashMap<String, String>();

		return model;
*/
		return null;
	}
	
}
