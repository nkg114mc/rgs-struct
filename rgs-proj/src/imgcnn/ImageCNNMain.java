package imgcnn;

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
import imgseg.ImageInstance;
import imgseg.ImageSegEvaluator;
import imgseg.ImageSegLabel;
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

public class ImageCNNMain {

	public static void main(String[] args) {
		
		try {
			
			CostFuncCacherAndLoader cCacher = new CostFuncCacherAndLoader(CostFuncCacherAndLoader.defaultFolder);
			ElearningArg elArg = new ElearningArg();
			//elArg.runEvalLearn = false;
			
			ImageSegLabel[] labels = ImageSegLabel.loadLabelFromFile("../msrc/imageseg_label_color_map.txt");
			String[] labelNames = ImageSegLabel.getStrLabelArr(labels, false);
			String[] labelNamesFull = ImageSegLabel.getStrLabelArr(labels, true);
			
			ImageSegEvaluator.initRgbToLabel(labels);

			ImageCNNReader reader = new ImageCNNReader("../msrc");
			ImageSegEvaluator evaluator = new ImageSegEvaluator(reader.getDebugFolder());


			//List<ImageInstance> trainInsts = loadFromListFile(reader, "../msrc/Train-small.txt", labelNames, true);
			//List<ImageInstance> trainInsts = loadFromListFile(reader, "../msrc/TrainValidation.txt", labelNames, true);
			List<ImageInstance> trainInsts = loadFromListFile(reader, "../msrc/Train.txt", labelNames, true);
			//List<ImageInstance> trainInsts = loadFromListFile(reader, "../msrc/Train3.txt", labelNames, true);
			
			List<ImageInstance> testInsts = loadFromListFile(reader, "../msrc/Test.txt", labelNamesFull, true);
			
			
			List<ImageInstance> totalInsts = new ArrayList<ImageInstance>();
			totalInsts.addAll(trainInsts);
			totalInsts.addAll(testInsts);
			computeAvgStructSize(trainInsts);

			String svmCfgFile = "sl-config/msrc21-search-DCD.config";
			String modelLogsFn = "../logistic_models/msrc21.logistic";
			String modelSvFn = "../logistic_models/msrc21.ssvm";
			
			SLModel slmodel = runLearning(trainInsts, testInsts, labelNames, InitType.UNIFORM_INIT, 1, 1, -1, svmCfgFile, modelLogsFn, modelSvFn, true, true, true, evaluator, labels, elArg, cCacher);
			//SLModel slmodel = runLearning(testInsts, testInsts, labelNames, InitType.LOGISTIC_INIT, 1, svmCfgFile, modelLogsFn, modelSvFn, true, true, true, evaluator, labels, elArg, cCacher);



			//evaluateUnary(trainInsts, slmodel, labelNames, labels, evaluator, true);
			evaluateUnary(testInsts, slmodel, labelNames, labels, evaluator, true);
			
			evaluator.evaluate(testInsts, slmodel, labels, true, -1);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void computeAvgStructSize(List<ImageInstance> insts) {
		
		double sum = 0;
		double cnt = 0;
		
		for (ImageInstance inst : insts) {
			sum += ((double) inst.size());
			System.out.println("Image structure size = " + inst.size());
			cnt++;
		}
		
		sum /= cnt;
		
		System.out.println("Avg image structure size = " + sum);
		
	}
	
	public static SLModel runLearning(List<ImageInstance> trainInsts, List<ImageInstance> testInsts, String[] labelNames, 
									  InitType initType, int restartTrain, int restartTest, double iniAlfa, 
			                          String svmCfgFile, String modelLogsFn, String modelSvFn,
			                          boolean usePairFeat, boolean useTernFeat, boolean useQuadFeat, 
			                          ImageSegEvaluator evaluator, ImageSegLabel[] labels,
			                          ElearningArg evalLearnArg, CostFuncCacherAndLoader costCacher) throws Exception {
		String configFilePath = svmCfgFile;
		//int nrnd = randomNum;

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
		ImageCNNFeaturizer fg = new ImageCNNFeaturizer(labelNames, usePairFeat, useTernFeat);// true, true);
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
		
		//int anotherRestart = 20;
		
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
	
	public static ExperimentResult averageMultiRunMSRC21(ImageSegEvaluator evaluator, 
			                                             List<ImageInstance> images,
			                                             SLModel model, 
			                                             ImageSegLabel[] labels, 
			                                             int restarts, 
			                                             int timeToRun) {
		
		try {
			
			assert (timeToRun > 0);
			
			ArrayList<ExperimentResult> allRes = new ArrayList<ExperimentResult>();
			ArrayList<OneTestingResult> allAccs = new ArrayList<OneTestingResult>();
			for (int i = 0; i < timeToRun; i++) {
				System.out.println("==>Testing run " + i + "<==");
				ExperimentResult re = evaluator.evaluate(images, model, labels, false, restarts);
				allRes.add(re);
				allAccs.add(re.testAcc);
			}
			
			// do average
			OneTestingResult.computeAverage(allAccs);
			
			return allRes.get(0);
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return null;// should not reach here
		
	}
	
	public static SLParameters getGivenParams() {
		SLParameters para = new SLParameters();
		
		para.LEARNING_MODEL = LearningModelType.L2LossSSVM;
		para.L2_LOSS_SSVM_SOLVER_TYPE = L2LossSSVMLearner.SolverType.DCDSolver;

		para.NUMBER_OF_THREADS = 1;
		para.C_FOR_STRUCTURE = 0.0001f;
		para.TRAINMINI = false;
		para.TRAINMINI_SIZE = 1000;
		para.STOP_CONDITION = 0.1f;
		para.CHECK_INFERENCE_OPT = false;
		para.MAX_NUM_ITER = 50;
		para.PROGRESS_REPORT_ITER = 10;
		para.INNER_STOP_CONDITION = 0.1f;
		para.MAX_ITER_INNER = 50;
		para.MAX_ITER_INNER_FINAL = 2500;
		para.TOTAL_NUMBER_FEATURE = -1;
		para.CLEAN_CACHE = true;
		para.CLEAN_CACHE_ITER = 5;
		para.DEMIDCD_NUMBER_OF_UPDATES_BEFORE_UPDATE_BUFFER = 100;
		para.DEMIDCD_NUMBER_OF_INF_PARSE_BEFORE_UPDATE_WV = 10;
		para.LEARNING_RATE = 0.01f;
		para.DECAY_LEARNING_RATE = false;
		
		return para;
	}
	
	public static List<HwInstance> imageInstToHwInst(List<ImageInstance> imageInsts) {
		List<HwInstance> hwinsts = new ArrayList<HwInstance>();
		for (ImageInstance imgInst : imageInsts) {
			hwinsts.add(imgInst);
		}
		return hwinsts;
	}
	
	public static List<ImageInstance> loadFromListFile(ImageCNNReader reader, String listfile, String[] labelNames, boolean dropVoid) {
		
		List<ImageInstance> imgs = new ArrayList<ImageInstance>();
		
		List<String> allNames = ImageCNNReader.getNameListFromFile(listfile);
		
		Counter<String> labelCntr = new Counter<String>();
		
		// adjust label domain
		if (dropVoid) {
			String[] arr = new String[21];
			System.arraycopy(labelNames, 0, arr, 0, arr.length);
			labelNames = arr;
		}
		
		for (String nm : allNames) {
			//System.out.println(nm + " ");// + inst.getSize());
			ImageInstance inst = reader.initInstGivenName(nm, labelNames, dropVoid);
			imgs.add(inst);

			for (HwSegment sup : inst.letterSegs) {
				labelCntr.incrementCount(sup.letter, 1);
			}
		}
		System.out.println("Load instance: " + allNames.size());
		
		for (Entry<String,Double> e : labelCntr.entrySet()) {
			System.out.println(e);
		}
		
		return imgs;
	}

	
	public static void evaluateUnary(List<ImageInstance> images, 
			                         SLModel model, 
			                         String[] labelSet, 
			                         ImageSegLabel[] labels, 
			                         ImageSegEvaluator evaluator, 
			                         boolean ifDump) {
		
		ImageSegEvaluator.initRgbToLabel(labels);
		
		double total = 0;
		double acc = 0;
		double avgTruAcc = 0;
		
		FracScore[] fscores = new FracScore[21];
		FracScore[] gtscs = new FracScore[21];
		for (int j = 0; j < 21; j++) {
			fscores[j] = new FracScore();
			gtscs[j] = new FracScore();
		}
		
		for (int i = 0; i < images.size(); i++) {
			
			HwOutput gold = images.get(i).getGoldOutput();
			HwOutput prediction = null;
			try {
				prediction = (HwOutput) model.infSolver.getBestStructure(model.wv, images.get(i));
			} catch (Exception e) {
				e.printStackTrace();
			}
			
			FracScore[] oneResult = ImageSegEvaluator.evaluateOneImage(images.get(i), prediction, labelSet);
			ImageSegEvaluator.accuFracScore(fscores, oneResult);
			FracScore[] gtResult = ImageSegEvaluator.evaluateOneImageGtPic(images.get(i), prediction, labels, labelSet);
			ImageSegEvaluator.accuFracScore(gtscs, gtResult);
			
			for (int j = 0; j < prediction.output.length; j++) {
				total += 1.0;
				if (prediction.output[j] == gold.output[j]){
					acc += 1.0;
				}
			}
			
			if (ifDump) {
				evaluator.dumpImage(images.get(i), prediction, labels);
			}
			
		}
		
		avgTruAcc = avgTruAcc / total;
		double accuracy = acc / total;
		System.out.println("Accuracy = " + acc + " / " + total + " = " + accuracy);
		
		ImageSegEvaluator.printMSRCscore(fscores, labelSet);
		System.out.println("**********************");
		ImageSegEvaluator.printMSRCscore(gtscs, labelSet);
	}
}
