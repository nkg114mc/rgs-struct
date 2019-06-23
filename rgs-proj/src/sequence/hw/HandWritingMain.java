package sequence.hw;

import java.util.ArrayList;
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
import elearning.EfuncInferenceJuly2017;
import elearning.ElearningArg;
import elearning.XgbRegressionLearner;
import elearning.einfer.ESearchInferencer;
import elearnnew.SamplingELearning;
import experiment.CommonDatasetLoader;
import experiment.CostFuncCacherAndLoader;
import experiment.ExperimentResult;
import experiment.OneTestingResult;
import experiment.RndLocalSearchExperiment.InitType;
import experiment.RndLocalSearchExperiment.MulLbLossType;
import experiment.TestingAcc;
import general.AbstractActionGenerator;
import general.AbstractInstance;
import general.AbstractLossFunction;
import general.AbstractOutput;
import general.FactorGraphBuilder.FactorGraphType;
import init.HwFastAlphaGenerator;
import init.HwFastSamplingRndGenerator;
import init.RandomStateGenerator;
import init.UniformRndGenerator;
import search.GreedySearcher;
import search.SeachActionGenerator;
import search.SearchResult;
import search.ZobristKeys;
import search.loss.GoldPredPair;
import search.loss.SearchLossHamming;

public class HandWritingMain {

	/**
	 * Hand-Writing Recognition Problem
	 * 
	 * Chao Ma
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		try {
			//runLearning(false);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void runOneFolder(boolean isSmall, int df) {
		
		String fdName = getDsName(isSmall, df);
		System.out.println("///////////// Start " + fdName + " Learning /////////////////");
		
		String modelLogsFn = "../logistic_models/"+fdName+".logistic"; 
		String modelSvFn = "../logistic_models/"+fdName+".ssvm";

		//ExperimentResult re = runLearningOneFolder(isSmall, df,
		//		initType, randomNum, svmCfgFile, modelLogsFn, modelSvFn,
		//		usePairFeat, useTernFeat, useQuadFeat,  evalLearnArg, costCacher);
		
		System.out.println("///////////// Finish " + fdName + " Learning /////////////////");
	}
/*
	public static void runLearning(boolean isSmall, int folderIndex, InitType initType, int randomNum, String svmCfgFile, String modelLogsFn, String modelSvFn,
            boolean usePairFeat, boolean useTernFeat, boolean useQuadFeat, ElearningArg evalLearnArg) throws Exception {

		String configFilePath = "sl-config/hw-search-DCD.config";
		int nrnd = 1;

		HwLabelSet hwLabels = new HwLabelSet();
		// load data
		HwDataReader rder = new HwDataReader();
		List<List<HwInstance>> trtstInsts = rder.readData("../datasets/hw", hwLabels, 0, isSmall);
		System.out.println("List count = " + trtstInsts.size());
		List<SLProblem> slproblems = HwDataReader.convertToSLProblem(trtstInsts);

		AbstractLossFunction lossfunc = new SearchLossHamming();
		
		//////////////////////////////////////////////////////////////////////
		// train
		SLModel model = new SLModel();
		SLProblem spTrain = slproblems.get(0);

		// initialize the inference solver
		ZobristKeys abkeys = new ZobristKeys(100, hwLabels.getLabels().length);
		HwFeaturizer fg = new HwFeaturizer(hwLabels.getLabels(), HwFeaturizer.HwSingleLetterFeatLen, true, true, true);
		//model.infSolver = new HwInferencer(fg);//HwSearchInferencer(fg);
		AbstractActionGenerator actGener = new SeachActionGenerator();
		
		RandomStateGenerator initStateGener = new UniformRndGenerator(new Random());
		GreedySearcher searcher = new GreedySearcher(FactorGraphType.SequenceGraph, fg, nrnd, actGener, initStateGener, lossfunc, abkeys);
		model.infSolver = new HwSearchInferencer(searcher);
		//model.infSolver = new HwViterbiInferencer(fg);
		model.featureGenerator = fg;

		SLParameters para = new SLParameters();
		para.loadConfigFile(configFilePath);
		para.TOTAL_NUMBER_FEATURE = fg.getFeatLen();

		Learner learner = LearnerFactory.getLearner(model.infSolver, fg, para);
		model.wv = learner.train(spTrain);
		model.config =  new HashMap<String, String>();

		// test
		//////////////////////////////////
		SLProblem spTest = slproblems.get(1);
		evaluate(spTest, model);
		
		////
		
		
		System.out.println("Done.");
	}
*/
	
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
			                                //InitType initType, int randomNum, double iniAlfa,
			                                InitType initType, int restartTrain, int restartTest, double iniAlfa,
			                                String svmCfgFile, String modelLogsFn, String modelSvFn,
                                            boolean usePairFeat, boolean useTernFeat, boolean useQuadFeat, ElearningArg evalLearnArg, CostFuncCacherAndLoader costCacher) throws Exception {
		
		String dsName = getDsName(isSmall, folderIndex);
		String configFilePath = svmCfgFile;
		//int nrnd = randomNum;

		// load data
		//HwLabelSet hwLabels = new HwLabelSet();
		// load data
		//HwDataReader rder = new HwDataReader();
		//List<List<HwInstance>> trtstInsts = rder.readData("../datasets/hw", hwLabels, folderIndex, isSmall);
		
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
		ExperimentResult tstRe = HandWritingMain.averageOfMultiRunEvaluation(spTest, model, evalLearnArg.multiRunTesting); 
		System.out.println("Done.");


		// about local optimal?
		SamplingELearning.exploreLocalOptimal(initStateGener, trtstInsts.get(1), model.wv, searcher, 20, null);
		
		return tstRe;
	}
	
	public static ExperimentResult evaluate(SLProblem sp, SLModel model) throws Exception {
		double total = 0;
		double acc = 0;
		double avgTruAcc = 0;

		HwSearchInferencer searchInfr = (HwSearchInferencer)(model.infSolver);
		AbstractLossFunction lossfunc = searchInfr.gsearcher.getLossFunc();
		
		System.err.println("TestRestart = " + searchInfr.getSearcher().randInitSize);
		
		List<GoldPredPair> results = new ArrayList<GoldPredPair>();
		
		for (int i = 0; i < sp.instanceList.size(); i++) {
			HwOutput gold = (HwOutput) sp.goldStructureList.get(i);
			SearchResult infrRe = searchInfr.runSearchInference(model.wv, null, sp.instanceList.get(i), gold);
			HwOutput prediction = (HwOutput)(infrRe.predState.structOutput);
			//HwOutput prediction = (HwOutput) searchInfr.getBestStructure(model.wv, sp.instanceList.get(i));
			for (int j = 0; j < prediction.output.length; j++) {
				total += 1.0;
				if (prediction.output[j] == gold.output[j]){
					acc += 1.0;
				}
			}
			
			GoldPredPair re = new GoldPredPair((AbstractInstance)(sp.instanceList.get(i)), (AbstractOutput)gold, (AbstractOutput)prediction);
			results.add(re);
			
			// sum true Acc
			avgTruAcc += infrRe.accuarcy;
		}
		
		avgTruAcc = avgTruAcc / total;
		double accuracy = acc / total;
		System.out.println("Accuracy = " + acc + " / " + total + " = " + accuracy);
		System.out.println("Accuracy = " + accuracy);
		
		double genAcc = avgTruAcc;
		double selAcc = genAcc - accuracy;
		
		if (genAcc < accuracy) {
			throw new RuntimeException("[ERROR]Generation accuracy is less than final output accuracy: " + genAcc + " < " + accuracy);
		}
		
		System.out.println("Generation Acc = " + genAcc);
		System.out.println("Selection AccDown = " + selAcc);
		
		double lossFuncAcc = lossfunc.computeMacroFromResults(results);
		System.out.println("LossFunction computed accuracy = " + lossFuncAcc);
		
		
		ExperimentResult expRslt = new ExperimentResult();
		expRslt.lossName = "";
		expRslt.overallAcc = accuracy;
		expRslt.generationAcc = genAcc;
		expRslt.selectionAcc = selAcc;
		
		
		expRslt.addAcc(new TestingAcc("HammingAcc", expRslt.overallAcc));
		expRslt.addAcc(new TestingAcc("GenerationAcc", expRslt.generationAcc));
		
		return expRslt;
	}
	
	public static ExperimentResult averageOfMultiRunEvaluation(SLProblem sp, SLModel model, int timeToRun) {
		try {
			
			assert (timeToRun > 0);
			
			ArrayList<ExperimentResult> allRes = new ArrayList<ExperimentResult>();
			ArrayList<OneTestingResult> allAccs = new ArrayList<OneTestingResult>();
			for (int i = 0; i < timeToRun; i++) {
				System.out.println("==>Testing run " + i + "<==");
				ExperimentResult re = HandWritingMain.evaluate(sp, model);
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

}



/*
public static void trainSequenceModel(String trainingDataPath, String configFilePath, String modelPath)
		throws Exception {
	SLModel model = new SLModel();
	SLProblem sp = SequenceIOManager.readProblem(trainingDataPath, false);
	
	// initialize the inference solver
	model.infSolver = new SequenceInferenceSolver();
	
	SLParameters para = new SLParameters();
	para.loadConfigFile(configFilePath);
	SequenceFeatureGenerator fg = new SequenceFeatureGenerator();
	para.TOTAL_NUMBER_FEATURE = SequenceIOManager.numFeatures * SequenceIOManager.numLabels + SequenceIOManager.numLabels +
			SequenceIOManager.numLabels *SequenceIOManager.numLabels;
	
	Learner learner = LearnerFactory.getLearner(model.infSolver, fg, para);
	model.wv = learner.train(sp);
	model.config =  new HashMap<String, String>();
	model.config.put("numFeatures", String.valueOf(SequenceIOManager.numFeatures));
	model.config.put("numLabels", String.valueOf(SequenceIOManager.numLabels));
	model.saveModel(modelPath);
}

public static void testSequenceModel(String modelPath, String testDataPath, String predictionFileName)
		throws Exception {
	SLModel model = SLModel.loadModel(modelPath);
	SequenceIOManager.numFeatures = Integer.valueOf(model.config.get("numFeatures"));
	SequenceIOManager.numLabels = Integer.valueOf(model.config.get("numLabels"));
	SLProblem sp = SequenceIOManager.readProblem(testDataPath, true);
	
	System.out.println("Acc = " + Evaluator.evaluate(sp, model.wv, model.infSolver,predictionFileName));
}
*/
