package multilabel;

import java.io.FileNotFoundException;
import java.util.ArrayList;

import multilabel.data.Dataset;
import multilabel.data.DatasetReader;
import multilabel.dimenreduct.CsspModel;
import multilabel.dimenreduct.CsspReducer;
import multilabel.dimenreduct.LabelDimensionReducer;
import multilabel.evaluation.MultiLabelEvaluator;
import multilabel.instance.Featurizer;
import multilabel.learning.Predictor;
import multilabel.learning.cost.CostFunction;
import multilabel.learning.cost.CostLearning;
import multilabel.learning.cost.RankingCostFunction;
import multilabel.learning.heuristic.HeuristicLearning;
import multilabel.learning.search.BreathFirstSearcher;
import multilabel.pruner.LabelPruner;
import multilabel.pruner.LambdaMartLabelPruner;
import multilabel.pruner.PruningEvaluator;



public class MultiLabelMain {
	
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
	}
	
	public static Param parseArgs(String[] args) {
		Param pargs = new Param();
		
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
			}
		}
		
		
		return pargs;
	}
	
	public static void main(String[] args) {
		
		Param paras = parseArgs(args);
		checkParameter(paras);
		
		DatasetReader dataSetReader = new DatasetReader();
		Dataset ds = dataSetReader.readDefaultDataset(paras.name);
		
		// work stream
		//runCompleteTrainingPreprocess(ds, paras.mlcPath, 
		//		                      paras.doDR, paras.doPruning, paras.drtopK, paras.pruneTopK); 
				                      //paras.beamSize, paras.maxDepth);
		runCompleteTrainingPreprocess(ds, paras);
		
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
	}
	
	//public static void runCompleteTrainingPreprocess(Dataset ds, String mlcPath,
	//		                                         boolean doDR, boolean doPruning, int drtopK, int pruneTopK) { 
			                                         ///int beamSize, int maxDepth) {
	
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
		//// "GROUND TRUTH PREDICT"
		////
/*
		ArrayList<Example> testExs = ds.getTestExamples();
		for (Example exmp : testExs) {
			ArrayList<Label> allLbl = exmp.getLabel();
			StructOutput bestOutWithPrunning = SearchState.getAllZeroState(exmp.labelDim()).getOutput();
			for (Label l : allLbl) {
				int idx = l.originIndex;
				if (l.isPruned) {
					bestOutWithPrunning.setValue(idx, 0); // pruned default value: 0
				} else {
					bestOutWithPrunning.setValue(idx, l.value);
				}
			}
			exmp.predictOutput = bestOutWithPrunning;
		}
*/
	
		
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

	
	
	
	
	
	
	
	
	
	
	
	
	public static void oldTrain(String[] args) {
		
		DatasetReader dataSetReader = new DatasetReader();

		//Dataset ds = dataSetReader.loadDataSetCSV(TRAIN_FEATURE_FILE_LOCATION, TRAIN_LABEL_FILE_LOCATION,
		//		                                  TEST_FEATURE_FILE_LOCATION, TEST_LABEL_FILE_LOCATION);
		
		String name = "emotions";//"scene";//"emotions";
		//String xmlFile = "/scratch/large_multi-label/multi_label_project/largemultilabel/small-Datasets/scene/scene.xml";
		//String testArffFile = "/scratch/large_multi-label/multi_label_project/largemultilabel/small-Datasets/scene/scene-test.arff";
		//String trainArffFile = "/scratch/large_multi-label/multi_label_project/largemultilabel/small-Datasets/scene/scene-train.arff";
		String xmlFile = "ML_datasets/"+name+"/"+name+".xml";
		String testArffFile = "ML_datasets/"+name+"/"+name+"-test.arff";
		String trainArffFile = "ML_datasets/"+name+"/"+name+"-train.arff";
		Dataset ds = dataSetReader.loadDataSetArff(name, trainArffFile, testArffFile, xmlFile);
		
		
		//PrunerLearning.dumpRanklibFeatureFile(ds.getTrainExamples(), name + "_train_feat.txt");
		//PrunerLearning.dumpRanklibFeatureFile(ds.getTestExamples(), name + "_test_feat.txt");
		
		//PrunerLearning.dumpWekaFeatureFile(ds.getTrainExamples(), "sence_train_feat.arff");
		//PrunerLearning.dumpWekaFeatureFile(ds.getTestExamples(), "sence_test_feat.arff");
		

		String modelFn = "pruner_rank/emotions_rank_lambdamart_p3.txt";
		LabelPruner pruner = new LambdaMartLabelPruner(modelFn, 4);
		PruningEvaluator.evaluatePruner(ds.getTestExamples(), pruner);

		ds.name = name;
		//RegressionCostFuncLearning.runLearning(ds);
		//RegressionCostFuncLearning.runLearningExhaustive(ds);
		
		/*
		RegressionCostFunction cf = new RegressionCostFunction(new Featurizer());
		cf.loadModel("/home/mc/workplace/large_multi-label/my_largelabel/emotions_regression5.model");
		//cf.loadModel("emotions_regression3.model");
		//cf.loadModel("emotions_regression_offtraj1.model");
		//Predictor.doTesting(ds, cf, null); // new RandomCostFunction()
		Predictor.doTestingExhaustive(ds, cf, 3, null);
	*/
		

		RankingCostFunction cf = new RankingCostFunction(new Featurizer());
		cf.loadModel("emotions_rankcost_lambdamart.txt"); //
		//cf.loadModel("scene_rank_lambdamart2.txt");
		//Predictor.doTesting(ds, cf, null);
		Predictor.doTestingExhaustive(ds, cf, 3, null);

		
		//RegressionCostFuncLearning.runLearningOffTraj(ds, cf);
		
		//RegressionCostFuncLearning rlearner = new RegressionCostFuncLearning();
		//rlearner.testOnTrainingData(ds, cf);
		
		System.out.println("All Done!");
		
	}

}
