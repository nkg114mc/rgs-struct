package multilabel;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import berkeleyentity.MyTimeCounter;
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
import general.AbstractLossFunction;
import general.FactorGraphBuilder.FactorGraphType;
import init.MultiLabelAlphaGenerator;
import init.MultiLabelSamplingRndGenerator;
import init.RandomStateGenerator;
import init.UniformRndGenerator;
import multilabel.data.Dataset;
import multilabel.data.DatasetReader;
import multilabel.data.HwInstanceDataset;
import multilabel.dimenreduct.CsspModel;
import multilabel.dimenreduct.CsspReducer;
import multilabel.dimenreduct.LabelDimensionReducer;
import multilabel.evaluation.MultiLabelEvaluator;
import multilabel.instance.Featurizer;
import multilabel.instance.Label;
import multilabel.learning.StructOutput;
import multilabel.learning.cost.CostFunction;
import multilabel.learning.cost.CostLearning;
import multilabel.learning.cost.RankingCostFunction;
import multilabel.learning.heuristic.HeuristicLearning;
import multilabel.learning.search.BreathFirstSearcher;
import multilabel.pruner.LabelPruner;
import multilabel.pruner.LambdaMartLabelPruner;
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

public class MultiLabelNew {

	public static class Param {
		public String name;
		public String mlcPath;
		///////////////////
		public boolean doDR;
		public int drtopK;
		
		public boolean doPruning; 
		public int pruneTopK; 
		public String prunerMdlPath;
		public boolean doPrunLearning;
		
		public boolean doHeurLearning;
		public String heurMdlPath;

		public int beamSize;
		public int maxDepth;
		
		public float ssvmc;
	}
	
	public static Param parseArgs(String[] args) {
		Param pargs = new Param();
		
		//// default
		pargs.ssvmc = -1;
		
		for (int i = 0; i < args.length; i++) {

			if (args[i].equals("-name")) {
				pargs.name = (args[i + 1]);
			} else if (args[i].equals("-mlc")) {
				pargs.mlcPath = (args[i + 1]);

			// pruning
			} else if (args[i].equals("-doprune")) {
				pargs.doPruning = Boolean.parseBoolean(args[i + 1]);
			} else if (args[i].equals("-pruneTopk")) {
				pargs.pruneTopK = Integer.parseInt(args[i + 1]);
			} else if (args[i].equals("-pmodel")) {
				pargs.prunerMdlPath = (args[i + 1]);
			} else if (args[i].equals("-plearn")) {
				pargs.doPrunLearning = Boolean.parseBoolean(args[i + 1]);

			// dr
			} else if (args[i].equals("-dodr")) {
				pargs.doDR = Boolean.parseBoolean(args[i + 1]);
			} else if (args[i].equals("-drTopK")) {
				pargs.drtopK = Integer.parseInt(args[i + 1]);

			// about search
			} else if (args[i].equals("-beam")) {
				pargs.beamSize = Integer.parseInt(args[i + 1]);
			} else if (args[i].equals("-depth")) {
				pargs.maxDepth = Integer.parseInt(args[i + 1]);

			
			// heuristic
			} else if (args[i].equals("-doheur")) {
				pargs.doHeurLearning = Boolean.parseBoolean(args[i + 1]);
			} else if (args[i].equals("-hmodel")) {
				pargs.heurMdlPath = (args[i + 1]);
			
			
			// ssvm
			} else if (args[i].equals("-ssvmc")) {
				pargs.ssvmc = Float.parseFloat(args[i + 1]);
			}
			
		}
		
		
		return pargs;
	}

	
	public static void main(String[] args) {
		
		Param paras = parseArgs(args);
		checkParameter(paras);
		
		String dsName = "yeast";//"bibtex";//"enron";//"bookmarks-umass";//
		//DatasetReader dataSetReader = new DatasetReader();
		//Dataset ds = dataSetReader.readDefaultDataset(dsName);

		paras.name = dsName;
		
		ElearningArg eLnArg = new ElearningArg();
		eLnArg.useFeat2 = true;
		//eLnArg.useFeat2 = false;
		//eLnArg.useFeat3 = false;
		//eLnArg.useFeat4 = false;
		
		
		eLnArg.runEvalLearn = false;
		//eLnArg.doEvalTest = false;
		
		CostFuncCacherAndLoader costchr = new CostFuncCacherAndLoader(CostFuncCacherAndLoader.defaultFolder);
		
		CommonDatasetLoader commonDsLdr = new CommonDatasetLoader();
		
		try {
			//MulLbLossType optimizeLoss = MulLbLossType.HAMMING_LOSS;
			MulLbLossType optimizeLoss = MulLbLossType.EXMPF1_LOSS;
			
			InitType initgen = InitType.UNIFORM_INIT;
			//InitType initgen = InitType.LOGISTIC_INIT;
			
			runLearning(commonDsLdr, dsName, initgen, optimizeLoss, 1, 1, -1,
					    "../sl-config/"+dsName+"-search-DCD.config", 
					    "../logistic_models/"+dsName+".logistic", 
					    "../logistic_models/"+dsName+".ssvm",
					    eLnArg, costchr);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		System.out.println("All Done!");
	}
	
	public static void checkParameter(Param para) {
		System.out.println("Dataset name = " + para.name);
		System.out.println("Mlc folder = " + para.mlcPath);
		
		System.out.println("doDR = " + para.doDR);
		System.out.println("drtopK = " + para.drtopK);
		
		System.out.println("doPruning = " + para.doPruning);
		System.out.println("prunLearning = " + para.doPrunLearning);
		System.out.println("pruneTopK = " + para.pruneTopK);
		System.out.println("prunePath = " + para.prunerMdlPath);

		
		System.out.println("beamSize = " + para.beamSize);
		System.out.println("maxDept = " + para.maxDepth);
		
		System.out.println("ssvmc = " + para.ssvmc);
	}
	
	
	public static void runLearning(CommonDatasetLoader commonDsLdr, 
								   //String dsName, InitType initType, MulLbLossType lossTyp,  int randomNum,  double iniAlfa,
								   String dsName, InitType initType, MulLbLossType lossTyp,  int restartTrain, int restartTest,  double iniAlfa,
			                       String svmCfgFile, String modelLogisticFn, String modelSvmFn, 
			                       ElearningArg evalLearnArg, CostFuncCacherAndLoader costCacher) throws Exception {
		
		String configPath = svmCfgFile;
		//int nrnd = randomNum;
		
		// load data
		//HwInstanceDataset ds = DatasetReader.loadHwInstances(dsName);
		HwInstanceDataset ds = commonDsLdr.getMultiLabelDs(dsName);
		
		RandomStateGenerator initStateGener = null;
		System.out.println("InitType = " + initType.toString());
		if (initType == InitType.UNIFORM_INIT) {
			initStateGener = new UniformRndGenerator(new Random());
		} else if (initType == InitType.LOGISTIC_INIT) {
			initStateGener = MultiLabelSamplingRndGenerator.loadGenrIfExist(modelLogisticFn, dsName, ds.getTrainExamples(), ds.getTestExamples(), Label.MULTI_LABEL_DOMAIN, false);
		} else if (initType == InitType.ALPHA_INIT) {
			MultiLabelSamplingRndGenerator tmpr = MultiLabelSamplingRndGenerator.loadGenrIfExist(modelLogisticFn, dsName, ds.getTrainExamples(), ds.getTestExamples(), Label.MULTI_LABEL_DOMAIN, false);
			initStateGener = new MultiLabelAlphaGenerator(tmpr.getLableSize(), tmpr.getDomainSizne(), tmpr.getWkHeader(), tmpr.logisticList(), tmpr.getRandom(), iniAlfa);
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

		Learner learner = LearnerFactory.getLearner(model.infSolver, fg, para);
		//Learner learner = new SSVMGurobiSolverWithEval(model.infSolver, fg, para, null);
		//Learner learner = new SSVMCopyDCDLearner(model.infSolver, fg, para);
		WeightVector initwv = new WeightVector(para.TOTAL_NUMBER_FEATURE);
		System.err.println("weightLength1 = " + initwv.getLength() + " " + para.TOTAL_NUMBER_FEATURE);
		System.err.println("SovlerType = " + para.L2_LOSS_SSVM_SOLVER_TYPE.toString());
		System.err.println("C = " + para.C_FOR_STRUCTURE);
		//para.MAX_NUM_ITER = 3;
		
		MyTimeCounter trnTimer = new MyTimeCounter("Traing time counter");
		trnTimer.start();
		long trn_t1 = trnTimer.getMilSecondSnapShot();
		
		//model.wv = learner.train(spTrain, initwv);
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
		
		System.err.println("TestRestart = " + searchInfr.getSearcher().randInitSize);

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
	
	
	public static void copyStruct(StructOutput gold, HwOutput golds) {
		if (gold.size() != golds.size()) {
			throw new RuntimeException(gold.size() + " != " + golds.size());
		}
		for (int i = 0; i < gold.size(); i++) {
			golds.setOutput(i, gold.getValue(i));
		}
	}
	
	public static void runCompleteTrainingPreprocess(Dataset ds, Param pargs) { 
		
		MultiLabelEvaluator evaluator = new MultiLabelEvaluator();
		
		////
		//// DIMENSION REDUCTION
		////
		
		CsspReducer reducer = null;
		if (pargs.doDR) {
			String[] rdPaths = CsspModel.getDefaultModelPath(pargs.mlcPath, ds.name, pargs.drtopK);
			String vPath = rdPaths[0];
			String pPath = rdPaths[1];
			
			reducer = new CsspReducer(vPath, pPath);
			CsspReducer.doDimensionReduction(ds.getTrainExamples(), reducer);
			CsspReducer.doDimensionReduction(ds.getTestExamples(), reducer);
		}

		////
		//// PRUNER LEARNING
		////
		
		////
		//// PRUNING
		////
		
		if (pargs.doPruning) {
			// pruner training
			LabelPruner pruner = null;
			if (pargs.doPrunLearning) {
				pruner = LambdaMartLabelPruner.trainPrunerRanklib(ds, pargs.pruneTopK, "R");
			} else {
				pruner = new LambdaMartLabelPruner(pargs.prunerMdlPath, pargs.pruneTopK);
			}
			
			//PruningEvaluator.evaluatePruner(ds.getTestExamples(), pruner);
			
			// do pruning
			LabelPruner.pruneExamples(ds.getTrainExamples(), pruner);
			LabelPruner.pruneExamples(ds.getTestExamples(), pruner);
		}
		
		////////////////////////////////////////
		// Test upper bound before search
		////////////////////////////////////////

		////
		//// HEURISTIC LEARNING
		////
		CostFunction heurFunc = null;
		if (pargs.doHeurLearning) {
			int hIteration = 0;
			ArrayList<CostFunction> allHeuristics = null;
			try {
				allHeuristics = BreathFirstSearcher.heuristicDaggerIterations(ds.getTrainExamples(), ds.getTestExamples(), pargs.beamSize, pargs.maxDepth, ds.name, hIteration);
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
			heurFunc = allHeuristics.get(0);
		} else {
			heurFunc = new RankingCostFunction(new Featurizer());
			heurFunc.loadModel(pargs.heurMdlPath);
		}
		// test generation loss
		HeuristicLearning.testGenerationLoss(ds, pargs.beamSize, pargs.maxDepth, heurFunc);

		////
		//// COST LEARNING
		////
		// TODO
		CostLearning.costLearning(ds, ds.getTrainExamples(), ds.getTestExamples(), pargs.beamSize, pargs.maxDepth, heurFunc);

		
		////
		//// DIMENSION REDUCTION RECONSTRUCT
		////
		if (pargs.doDR) {
			//LabelDimensionReducer.doDrReconstruct(ds.getTrainExamples(), reducer);
			LabelDimensionReducer.doDrReconstruct(ds.getTestExamples(), reducer);
		}
		
		////
		//// EVALUATION
		////
		
		// scoring!
		evaluator.evaluationDataSet(ds.name, ds.getTestExamples());
	}
	
	
}
