package imgseg;

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
import edu.illinois.cs.cogcomp.sl.util.IFeatureVector;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearning.AbstractRegressionLearner;
import elearning.AlterElearning;
import elearning.EFunctionLearning;
import elearning.EInferencer;
import elearning.EfuncInferenceDec2017;
import elearning.EfuncInferenceJuly2017;
import elearning.ElearningArg;
import elearning.RegressionInstance;
import elearning.SGDRegressionLearner;
import elearning.XgbRegressionLearner;
import elearning.LowLevelCostLearning.StopType;
import elearning.einfer.ESearchInferencer;
import elearning.einfer.SearchStateScoringFunction;
import elearning.einfer.XgbSearchStateScorer;
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
import init.ImageSegAlphaGenerator;
import init.ImageSegSamplingRndGenerator;
import init.RandomStateGenerator;
import init.UniformRndGenerator;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;
import multilabel.MultiLabelFeaturizer;
import search.GreedySearcher;
import search.SeachActionGenerator;
import search.ZobristKeys;
import search.loss.SearchLossHamming;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;
import sequence.hw.HwSearchInferencer;
import sequence.hw.HwSegment;

public class ImageSegXgbUnary {

	public static void main(String[] args) {
		
		try {
			
			CostFuncCacherAndLoader cCacher = new CostFuncCacherAndLoader(CostFuncCacherAndLoader.defaultFolder);
			ElearningArg elArg = new ElearningArg();
			//elArg.runEvalLearn = false;
			
			ImageSegLabel[] labels = ImageSegLabel.loadLabelFromFile("../msrc/imageseg_label_color_map.txt");
			String[] labelNames = ImageSegLabel.getStrLabelArr(labels, false);
			String[] labelNamesFull = ImageSegLabel.getStrLabelArr(labels, true);
			
			ImageSegEvaluator.initRgbToLabel(labels);

			ImageDataReader reader = new ImageDataReader("../msrc");
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
			
			learnWithXgboost(trainInsts, testInsts, labelNames, labels, evaluator);
			//testBooster(testInsts, slmodel, labels, true, -1);
			
			//evaluator.evaluate(testInsts, slmodel, labels, true, -1);

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
	
/*
	public static SLModel runLearning(List<ImageInstance> trainInsts, List<ImageInstance> testInsts, String[] labelNames, 
									  InitType initType, int restartTrain, int restartTest, double iniAlfa, 
			                          String svmCfgFile, String modelLogsFn, String modelSvFn,
			                          boolean usePairFeat, boolean useTernFeat, boolean useQuadFeat, 
			                          ImageSegEvaluator evaluator, ImageSegLabel[] labels,
			                          ElearningArg evalLearnArg, CostFuncCacherAndLoader costCacher) throws Exception {
		String configFilePath = svmCfgFile;
		//int nrnd = randomNum;

		String dsName = "msrc21";
		SLProblem spTrain = ImageDataReader.ExampleListToSLProblem(trainInsts);
		
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
		ImageSegFeaturizer fg = new ImageSegFeaturizer(labelNames, usePairFeat, useTernFeat);// true, true);
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

		averageMultiRunMSRC21(evaluator, testInsts, model, labels, restartTest, evalLearnArg.multiRunTesting);              
		
		
		////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////
		
		List<HwInstance> trnHwInsts = imageInstToHwInst(trainInsts);
		List<HwInstance> tstHwInsts = imageInstToHwInst(testInsts);
		
		// about local optimal?
		SamplingELearning.exploreLocalOptimal(initStateGener, tstHwInsts, model.wv, searcher, 20, null);

		/////////////////////////////////////////////////////////////////////
		// Run Inference with or without evaluation function //
		/////////////////////////////////////////////////////////////////////
		
		if (evalLearnArg.doEvalTest) {
			

			ImageSegFeaturizer efzr = new ImageSegFeaturizer(labelNames, usePairFeat, useTernFeat);
			AbstractRegressionLearner regser = new XgbRegressionLearner(dsName);  //// train an evaluation function
			EInferencer einfr = new ESearchInferencer(null, efzr, abkeys, lossfunc, actGener);// default setting is no evaluation learning ///// do evaluation inference
			int eIter = evalLearnArg.elearningIter;


			// no evaluation
			//EfuncInferenceDec2017.testEvaluationSpeedupJuly2017(tstHwInsts,
			EfuncInferenceJuly2017.testEvaluationSpeedupJuly2017(tstHwInsts,
					model.wv, 
					searcher,
					eIter,
					dsName,
					////////////////////////////////////
					false,
					efzr,
					regser,
					einfr);
		}
		
		return model;
	}
*/
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

	public static List<HwInstance> imageInstToHwInst(List<ImageInstance> imageInsts) {
		List<HwInstance> hwinsts = new ArrayList<HwInstance>();
		for (ImageInstance imgInst : imageInsts) {
			hwinsts.add(imgInst);
		}
		return hwinsts;
	}
	
	public static List<ImageInstance> loadFromListFile(ImageDataReader reader, String listfile, String[] labelNames, boolean dropVoid) {
		
		List<ImageInstance> imgs = new ArrayList<ImageInstance>();
		
		List<String> allNames = ImageDataReader.getNameListFromFile(listfile);
		
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

/*
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
*/
	
	
	public static void learnWithXgboost(List<ImageInstance> trainInsts, 
			 					 List<ImageInstance> testInsts, 
			 					 String[] labelNames, 
			 					 ImageSegLabel[] labels,
								 ImageSegEvaluator evaluator) {
		
		ImageSegFeaturizer fg = new ImageSegFeaturizer(labelNames, false, false);
		String[] labelDomain = fg.alphabet;
		
		List<ImageSuperPixel> trainSupPixs = getSuperPixList(trainInsts);
		List<ImageSuperPixel> testSupPixs = getSuperPixList(testInsts);
		
		DMatrix trainMtrx = createDMatrix(trainSupPixs, fg, true);
		DMatrix testMtrx = createDMatrix(testSupPixs, fg, true);
		
		Booster booster = performLearningGvienTrainTestDMatrix(trainMtrx, testMtrx);
		
		
		testBooster(testInsts, booster, labelNames, labels, evaluator, fg);
		
	}
	
	public static List<ImageSuperPixel> getSuperPixList(List<ImageInstance> insts) {
		ArrayList<ImageSuperPixel> supixs = new ArrayList<ImageSuperPixel>();
		for (int i = 0; i < insts.size(); i++) {
			ImageInstance imginst = insts.get(i);
			for (ImageSuperPixel imgsp : imginst.superPixelArr) {
				supixs.add(imgsp);
			}
		}
		return supixs;
	}
	
	public static void testBooster(List<ImageInstance> images, 
			Booster booster, 
            String[] labelSet, 
            ImageSegLabel[] labels, 
            ImageSegEvaluator evaluator,
            ImageSegFeaturizer featurizer) {

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
				prediction = predictWithBooster(images.get(i), featurizer, booster);//(HwOutput) model.infSolver.getBestStructure(model.wv, images.get(i));
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

			//if (ifDump) {
			//	evaluator.dumpImage(images.get(i), prediction, labels);
			//}

		}

		avgTruAcc = avgTruAcc / total;
		double accuracy = acc / total;
		System.out.println("Accuracy = " + acc + " / " + total + " = " + accuracy);

		ImageSegEvaluator.printMSRCscore(fscores, labelSet);
		System.out.println("**********************");
		ImageSegEvaluator.printMSRCscore(gtscs, labelSet);
	}
	
	public static HwOutput predictWithBooster(ImageInstance inst, ImageSegFeaturizer featurizer, Booster booster) {
		
		try {
			
			String[] labelDomain = featurizer.alphabet;

			HwOutput pred = new HwOutput(inst.size(), labelDomain);
			//HwOutput predCp = new HwOutput(inst.size(), labelDomain);

			for (int i = 0; i < inst.size(); i++) {
				float bestSc = Float.NEGATIVE_INFINITY;
				for (int j = 0; j < labelDomain.length; j++) {
					//predCp.setOutput(i, j);
					HashMap<Integer, Double> fv = featurizer.featurizeSuperPixel(inst.getSuPix(i), j);///featurizer.getUnaryFeatureVector(inst, predCp, i);
					DMatrix mx = createSingleMatrix(fv, false); //wv.dotProduct(fv);
					float[][] result;

					result = booster.predict(mx);

					float sc = result[0][0];

					if (sc > bestSc) {
						bestSc = sc;
						pred.output[i] = j;
					}
				}
			}
			
			return pred;
		
		} catch (XGBoostError e) {
			e.printStackTrace();
		}
		
		return null;
	}
	
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	public static Booster performLearningGvienTrainTestDMatrix(DMatrix trainMax, DMatrix testMax) {
		try {

			//// train
			System.out.println("Trainset size: " + trainMax.rowNum());

			HashMap<String, Object> params = new HashMap<String, Object>();
			params.put("eta", 0.1);
			params.put("max_depth", 10);
			params.put("silent", 0);
			params.put("objective", "rank:pairwise");
			params.put("eval_metric", "pre@1");
			params.put("nthread", 4);

			HashMap<String, DMatrix> watches = new HashMap<String, DMatrix>();
			watches.put("train", trainMax);
			watches.put("test", testMax);

			int round = 300;
			// train a model
			Booster booster = XGBoost.train(trainMax, params, round, watches, null, null);

			//booster.saveModel(file.getAbsolutePath() + "/xgb-regres-"+dsname+".model");
			trainMax.dispose();

			return booster;

		} catch (XGBoostError e) {
			e.printStackTrace();
		}

		return null;
	}

	public static DMatrix createDMatrix(List<ImageSuperPixel> trnData, ImageSegFeaturizer featurizer, boolean verbose) {

		try {

			int totalCnt = 0;
			int maxFeatIdx = 0;
			ArrayList<Float> tlabels = new ArrayList<Float>();
			ArrayList<Float> tdata   = new ArrayList<Float>();
			ArrayList<Long> theaders = new ArrayList<Long>();
			ArrayList<Integer> tindex = new ArrayList<Integer>();
			ArrayList<Integer> tgroup = new ArrayList<Integer>();

			int rowCnt = 0;
			long rowheader = 0L;
			theaders.add(rowheader);
			
			String[] labelDomain = featurizer.alphabet;
			
			for (int idx = 0; idx < trnData.size(); idx++) {
				ImageSuperPixel supix = trnData.get(idx);
				for (int j = 0; j < labelDomain.length; j++) {
					HashMap<Integer, Double> fv = featurizer.featurizeSuperPixel(supix, j);
					
					rowCnt += 1;
					float lbl = 0;
					if (j == supix.getLabel()) lbl = 1;

					HashMap<Integer, Double> feat = fv;
					for (Integer i : feat.keySet()) {
						int fidx = i.intValue() + 1;
						double fval = feat.get(i);

						tdata.add((float)fval);
						tindex.add(fidx);
					}


					long totalFeats = (long)feat.size();

					rowheader += totalFeats;
					theaders.add(rowheader);
					tlabels.add(lbl);
				}
				
				tgroup.add(labelDomain.length);
			}

			float[] splabels = listToArrFloat(tlabels);
			float[] spdata = listToArrFloat(tdata);
			int[] spcolIndex = listToArrInt(tindex);
			long[] sprowHeaders = listToArrLong(theaders);
			int[] spgroups = listToArrInt(tgroup);

			if (verbose) {
				System.out.println("splabels = " + splabels.length);
				System.out.println("spgroups = " + spgroups.length);
				System.out.println("spdata = " + spdata.length);
				System.out.println("spcolIndex = " + spcolIndex.length);
				System.out.println("sprowHeaders = " + sprowHeaders.length);
			}

			DMatrix mx = new DMatrix(sprowHeaders, spcolIndex, spdata, DMatrix.SparseType.CSR, 0);

			mx.setLabel(splabels);
			mx.setGroup(spgroups);

			// print some statistics
			if (verbose) {
				System.out.println("Rows: " + rowCnt);
				//System.out.println("Max feature index: " + maxFeatIdx);
			}
			return (mx);

		} catch (XGBoostError e) {
			e.printStackTrace();
		}

		return null; // should not arrive here
	}
	
	public static DMatrix createSingleMatrix(HashMap<Integer, Double> feat, boolean verbose) {

		try {

			ArrayList<Float> tlabels = new ArrayList<Float>();
			ArrayList<Float> tdata   = new ArrayList<Float>();
			ArrayList<Long> theaders = new ArrayList<Long>();
			ArrayList<Integer> tindex = new ArrayList<Integer>();

			int rowCnt = 0;
			long rowheader = 0L;
			theaders.add(rowheader);

			rowCnt += 1;
			float lbl = 0f;

			for (Integer i : feat.keySet()) {
				int fidx = i.intValue() + 1;
				double fval = feat.get(i);

				tdata.add((float)fval);
				tindex.add(fidx);
			}


			long totalFeats = (long)feat.size();

			rowheader += totalFeats;
			theaders.add(rowheader);
			tlabels.add(lbl);

			float[] splabels = listToArrFloat(tlabels);
			float[] spdata = listToArrFloat(tdata);
			int[] spcolIndex = listToArrInt(tindex);
			long[] sprowHeaders = listToArrLong(theaders);

			if (verbose) {
				System.out.println("splabels = " + splabels.length);
				System.out.println("spdata = " + spdata.length);
				System.out.println("spcolIndex = " + spcolIndex.length);
				System.out.println("sprowHeaders = " + sprowHeaders.length);
			}

			DMatrix mx = new DMatrix(sprowHeaders, spcolIndex, spdata, DMatrix.SparseType.CSR, 0);
			mx.setLabel(splabels);

			if (verbose) {
				System.out.println("Rows: " + rowCnt);
			}
			return (mx);

		} catch (XGBoostError e) {
			e.printStackTrace();
		}

		return null; // should not arrive here
	}
	
	private static float[] listToArrFloat(List<Float> list) {
		float[] arr = new float[list.size()];
		for (int i = 0; i < list.size(); i++) {
			arr[i] = list.get(i);
		}
		return arr;
	}
	
	private static int[] listToArrInt(List<Integer> list) {
		int[] arr = new int[list.size()];
		for (int i = 0; i < list.size(); i++) {
			arr[i] = list.get(i);
		}
		return arr;
	}
	
	private static long[] listToArrLong(List<Long> list) {
		long[] arr = new long[list.size()];
		for (int i = 0; i < list.size(); i++) {
			arr[i] = list.get(i);
		}
		return arr;
	}
}
