package multilabel;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import berkeleyentity.MyTimeCounter;
import edu.illinois.cs.cogcomp.sl.core.SLModel;
import edu.illinois.cs.cogcomp.sl.core.SLParameters;
import edu.illinois.cs.cogcomp.sl.core.SLProblem;
import edu.illinois.cs.cogcomp.sl.learner.Learner;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearning.ElearningArg;
import elearnnew.SamplingELearning;
import essvm.SSVMBookMarksDCDLearner;
import essvm.SSVMCopySPerceptronLearner;
import experiment.CommonDatasetLoader;
import experiment.CostFuncCacherAndLoader;
import experiment.ExperimentResult;
import experiment.OneTestingResult;
import experiment.RndLocalSearchExperiment.InitType;
import experiment.RndLocalSearchExperiment.MulLbLossType;
import experiment.TestingAcc;
import general.AbstractActionGenerator;
import general.AbstractLossFunction;
import general.FactorGraphBuilder.FactorGraphType;
import init.MultiLabelAllZeroGenerator;
import init.MultiLabelXgbRndGenerator;
import init.RandomStateGenerator;
import init.UniformRndGenerator;
import multilabel.data.DatasetReader;
import multilabel.data.HwInstanceDataset;
import multilabel.evaluation.MultiLabelEvaluator;
import multilabel.instance.Label;
import multilabel.utils.UtilFunctions;
import search.GreedySearcher;
import search.SeachActionGenerator;
import search.SearchResult;
import search.ZobristKeys;
import search.loss.SearchLossExmpAcc;
import search.loss.SearchLossExmpF1;
import search.loss.SearchLossHamming;
import sequence.hw.HwDataReader;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;
import sequence.hw.HwSearchInferencer;

public class BookMarksMain {

	public static void main(String[] args) {
		
		MultiLabelNew.Param paras = MultiLabelNew.parseArgs(args);
		MultiLabelNew.checkParameter(paras);
		
		String dsName = "bookmarks-umass";//"yeast";//"bibtex";//

		paras.name = dsName;
		
		ElearningArg eLnArg = new ElearningArg();
		eLnArg.useFeat2 = true;
		//eLnArg.useFeat2 = false;
		//eLnArg.useFeat3 = false;
		//eLnArg.useFeat4 = false;

		eLnArg.runEvalLearn = false;
		eLnArg.doEvalTest = false;
		
		eLnArg.assignSvmC = paras.ssvmc; // user assigned C
		
		CostFuncCacherAndLoader costchr = new CostFuncCacherAndLoader(CostFuncCacherAndLoader.defaultFolder);
		
		CommonDatasetLoader commonDsLdr = new CommonDatasetLoader();
		
		try {
			//MulLbLossType optimizeLoss = MulLbLossType.HAMMING_LOSS;
			MulLbLossType optimizeLoss = MulLbLossType.EXMPF1_LOSS;
			
			InitType initgen = InitType.ALLZERO_INIT;
			//InitType initgen = InitType.UNIFORM_INIT;
			//InitType initgen = InitType.LOGISTIC_INIT;
			
			runLearning(commonDsLdr, dsName, initgen, optimizeLoss, 1,1, -2,
					    "../sl-config/"+dsName+"-search-DCD.config", 
					    //"../sl-config/"+dsName+"-perc.config", 
					    "../logistic_models/"+dsName+".logistic", 
					    "../logistic_models/"+dsName+".ssvm",
					    eLnArg, costchr);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		System.out.println("All Done!");
	}
	
	public static void runLearning(CommonDatasetLoader commonDsLdr, 
			   						String dsName, InitType initType, MulLbLossType lossTyp,  
			   						//int randomNum,  double iniAlfa,
			   						int restartTrain, int restartTest,  double iniAlfa,
			   						String svmCfgFile, String modelLogisticFn, String modelSvmFn, 
			   						ElearningArg evalLearnArg, CostFuncCacherAndLoader costCacher) throws Exception {
		
		String configPath = svmCfgFile;
		//int nrnd = randomNum;
		
		// load data
		HwInstanceDataset ds = DatasetReader.loadHwInstances(dsName);
		
		RandomStateGenerator initStateGener = null;
		System.out.println("InitType = " + initType.toString());
		if (initType == InitType.UNIFORM_INIT) {
			initStateGener = new UniformRndGenerator(new Random());
		} else if (initType == InitType.LOGISTIC_INIT) {
			//initStateGener = MultiLabelSamplingRndGenerator.loadGenrIfExist(modelLogisticFn, dsName, ds.getTrainExamples(), ds.getTestExamples(), Label.MULTI_LABEL_DOMAIN, false);
			initStateGener = MultiLabelXgbRndGenerator.loadGenrIfExist(modelLogisticFn, dsName, ds.getTrainExamples(), ds.getTestExamples(), Label.MULTI_LABEL_DOMAIN, false);
		} else if (initType == InitType.ALLZERO_INIT) {
			initStateGener = new MultiLabelAllZeroGenerator(ds.getLabelDimension(), Label.MULTI_LABEL_DOMAIN.length);
		}
		System.out.println("=======");
		
		AbstractActionGenerator actGener = new SeachActionGenerator();
		AbstractLossFunction searchLossFunc = buildLossFunction(lossTyp);
		
		List<SLProblem> slproblems = HwDataReader.convertToSLProblem(ds.getInstListList());

		//////////////////////////////////////////////////////////////////////
		// train
		SLModel model = new SLModel();
		SLProblem spTrain = slproblems.get(0);

		// initialize the inference solver
		ZobristKeys abkeys = new ZobristKeys(500, Label.MULTI_LABEL_DOMAIN.length);
		MultiLabelFeaturizer fg = new MultiLabelFeaturizer(ds.getLabelDimension(), ds.getFeatureDimension(), evalLearnArg.useFeat2, evalLearnArg.useFeat3, evalLearnArg.useFeat4); //false, false, false);

		
		GreedySearcher searcher = new GreedySearcher(FactorGraphType.MultiLabelGraph, fg, restartTrain, actGener, initStateGener, searchLossFunc, abkeys);
		model.infSolver = new HwSearchInferencer(searcher);
		model.featureGenerator = fg;

		SLParameters para = new SLParameters();
		para.loadConfigFile(configPath);
		para.TOTAL_NUMBER_FEATURE = fg.getFeatLen();
		if (evalLearnArg.assignSvmC > 0) {
			float oldc = para.C_FOR_STRUCTURE;
			para.C_FOR_STRUCTURE = evalLearnArg.assignSvmC;
			System.err.println("Change C from configed " + oldc + " to " + para.C_FOR_STRUCTURE);
		}

		//Learner learner = LearnerFactory.getLearner(model.infSolver, fg, para);
		//Learner learner = new SSVMGurobiSolverWithEval(model.infSolver, fg, para, null);
		//Learner learner = new SSVMCopyDCDLearner(model.infSolver, fg, para);
		
		//Learner learner = new SSVMCopySPerceptronLearner(model.infSolver, fg, para);
		Learner learner = new SSVMBookMarksDCDLearner(model.infSolver, fg, para, ds.getLabelDimension());
		
		WeightVector initwv = new WeightVector(para.TOTAL_NUMBER_FEATURE);
		System.err.println("weightLength1 = " + initwv.getLength() + " " + para.TOTAL_NUMBER_FEATURE);
		System.err.println("SovlerType = " + para.L2_LOSS_SSVM_SOLVER_TYPE.toString());
		System.err.println("C = " + para.C_FOR_STRUCTURE);
		System.err.println("IterNum = " + para.MAX_NUM_ITER);
		System.err.println("InerIter = " + para.MAX_ITER_INNER);
		System.err.println("learnRate = " + para.LEARNING_RATE);
		//para.MAX_NUM_ITER = 3;
		
		
		MyTimeCounter trnTimer = new MyTimeCounter("Traing time counter");
		trnTimer.start();
		long trn_t1 = trnTimer.getMilSecondSnapShot();

		if (CostFuncCacherAndLoader.cacheCostWeight) {
			WeightVector loadedWv = costCacher.loadCachedWeight(dsName, initType, restartTrain, CostFuncCacherAndLoader.getFeatDim(evalLearnArg.useFeat2, evalLearnArg.useFeat3, evalLearnArg.useFeat4), lossTyp, para.C_FOR_STRUCTURE, iniAlfa);
			if (loadedWv != null) { // load failure...
				model.wv = loadedWv;
			} else {
				model.wv = learner.train(spTrain, initwv);
				costCacher.saveCachedWeight(model.wv, dsName, initType, restartTrain, CostFuncCacherAndLoader.getFeatDim(evalLearnArg.useFeat2, evalLearnArg.useFeat3, evalLearnArg.useFeat4), lossTyp, para.C_FOR_STRUCTURE, iniAlfa); // save
			}
		} else {
			model.wv = learner.train(spTrain, initwv);
		}
		model.config = new HashMap<String, String>();
		
		long trn_t2 = trnTimer.getMilSecondSnapShot();
		trnTimer.end();
		trnTimer.printSecond("Training-" + dsName);
		System.out.println("Train milli sec: " + (trn_t2 - trn_t1));

		
		

		// test
		//////////////////////////////////
		System.out.println("Done training...");
		searcher.setRestart(restartTest);
		averageOfMultiRunEvaluationMl(ds, model, searchLossFunc, evalLearnArg.multiRunTesting);
		
		// about local optimal?
		SamplingELearning.exploreLocalOptimal(initStateGener, ds.getTestExamples(), model.wv, searcher, 20, null);
		
		////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////
	}
	
	public static void saveWeightWithC(String folder, WeightVector wv, String name, InitType initTp, int restarts, int featDim, MulLbLossType loss, float ssvmC) {
		String wf = "cached_" + name + "_" + CostFuncCacherAndLoader.initTypeStr(initTp, -1) + "_" + "restr" + String.valueOf(restarts) + "_" + "feat" + String.valueOf(featDim) + "_" + "loss" + String.valueOf(loss)+ "_" + "c" + String.valueOf(ssvmC) + ".cost";
		String fileNm = folder + "/" + wf;
		UtilFunctions.saveObj(wv, fileNm);
		System.out.println("SSaavvee weight to file: [" + fileNm + "] with dimension " + wv.getLength());
	}
	
	public static AbstractLossFunction buildLossFunction(MulLbLossType lossTyp) {
		if (lossTyp == MulLbLossType.HAMMING_LOSS) {
			return (new SearchLossHamming());
		} else if (lossTyp == MulLbLossType.EXMPF1_LOSS) {
			return (new SearchLossExmpF1());
		} else if (lossTyp == MulLbLossType.EXMPACC_LOSS) {
			return (new SearchLossExmpAcc());
		}// ?
		return null;
	}
	
	public static ExperimentResult averageOfMultiRunEvaluationMl(HwInstanceDataset ds, SLModel model, AbstractLossFunction searchLossFunc, int timeToRun) {
		try {
			
			assert (timeToRun > 0);
			
			ArrayList<ExperimentResult> allRes = new ArrayList<ExperimentResult>();
			ArrayList<OneTestingResult> allAccs = new ArrayList<OneTestingResult>();
			for (int i = 0; i < timeToRun; i++) {
				
				MyTimeCounter tstTimer = new MyTimeCounter("Test time counter");
				tstTimer.start();
				long tst_t1 = tstTimer.getMilSecondSnapShot();
				
				System.out.println("==>Testing run " + i + "<==");
				ExperimentResult re = evaluate(ds.name + "-test", ds.getTestExamples(), model, searchLossFunc);
				
				long tst_t2 = tstTimer.getMilSecondSnapShot();
				tstTimer.end();
				tstTimer.printSecond("Test-" + ds.name);
				System.out.println("Test milli sec: " + (tst_t2 - tst_t1));
				
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
	
	public static ExperimentResult evaluate(String name, List<HwInstance> tstExs, SLModel model, AbstractLossFunction searchLossFunc) throws Exception {
		
		double total = 0;
		double instCnt = 0;
		double acc = 0;
		double accumAcc = 0;
		double avgTruAcc = 0;
		
		HwSearchInferencer searchInfr = (HwSearchInferencer)(model.infSolver);

		for (int i = 0; i < tstExs.size(); i++) {
			instCnt++;
			HwInstance ins = tstExs.get(i);
			HwOutput golds = ins.getGoldOutput();
			SearchResult infrRe = searchInfr.runSearchInference(model.wv, null, ins, golds);
			HwOutput prediction = (HwOutput)(infrRe.predState.structOutput);
			ins.setPredict(prediction);
			
			double predAcc = searchLossFunc.computeZeroOneAcc(ins, golds,  prediction).getVal();
			
			/////////////////////////////////
			
			for (int j = 0; j < prediction.output.length; j++) {
				total += 1.0;
				if (prediction.output[j] == golds.output[j]){
					acc += 1.0;
				}
			}
			// sum true Acc
			avgTruAcc += infrRe.accuarcy;
			// sum pred Acc
			accumAcc += predAcc;
		}
		
		MultiLabelEvaluator evaluator = new MultiLabelEvaluator();
		List<TestingAcc> scorerScores = evaluator.evaluationHwInstance(name, tstExs);//evaluator.evaluationDataSet(name, tstExs);
		
		///////////////////////////////////////
		///////////////////////////////////////
		System.out.println("LossName = " + searchLossFunc.getClass().getSimpleName());
		if (searchLossFunc.getClass().getSimpleName().equals("SearchLossHamming")) {
			avgTruAcc = avgTruAcc / total;
			accumAcc = accumAcc / total;
		} else {
			avgTruAcc = avgTruAcc / instCnt;
			accumAcc = accumAcc / instCnt;
		}
		
		
		double accuracyHamm = acc / total;
		System.out.println("Accuracy = " + acc + " / " + total + " = " + accuracyHamm);
		
		double genAcc = avgTruAcc;
		double selAcc = genAcc - accumAcc;
		
		if (genAcc < accumAcc) {
			throw new RuntimeException("[ERROR]Generation accuracy is less than final output accuracy: " + genAcc + " < " + accumAcc);
		}
		
		System.out.println("Overall Acc = " + accumAcc);
		System.out.println("Generation Acc = " + genAcc);
		System.out.println("Selection AccDown = " + selAcc);
		System.out.println("//////////////////////////////////////");
		System.out.println();
		
		
		ExperimentResult expRes = new ExperimentResult();
		expRes.addAccBatch(scorerScores);
		expRes.addAcc(new TestingAcc("OverallAcc",  accumAcc));
		expRes.addAcc(new TestingAcc("GenerationAcc", genAcc));

		return expRes;
	}


	
}
