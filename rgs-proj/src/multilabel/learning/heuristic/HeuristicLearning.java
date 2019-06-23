package multilabel.learning.heuristic;

import multilabel.instance.Example;

import java.io.FileNotFoundException;
import java.util.ArrayList;

import multilabel.learning.cost.CostFunction;
import multilabel.learning.search.BreathFirstSearcher;
import multilabel.learning.search.OldGreedySearcher;
import multilabel.learning.search.OldSearchState;
import multilabel.learning.search.SearcherInfo;
import multilabel.data.Dataset;
import multilabel.data.DatasetReader;
import multilabel.evaluation.MultiLabelEvaluator;

public class HeuristicLearning {
	
	
	public static void main(String[] args) {
		
		DatasetReader dataSetReader = new DatasetReader();

		//Dataset ds = dataSetReader.loadDataSetCSV(TRAIN_FEATURE_FILE_LOCATION, TRAIN_LABEL_FILE_LOCATION,
		//		                                  TEST_FEATURE_FILE_LOCATION, TEST_LABEL_FILE_LOCATION);
		
		String name = "emotions";//"scene";
		String xmlFile = "ML_datasets/"+name+"/"+name+".xml";
		String testArffFile = "ML_datasets/"+name+"/"+name+"-test.arff";
		String trainArffFile = "ML_datasets/"+name+"/"+name+"-train.arff";
		Dataset ds = dataSetReader.loadDataSetArff(name, trainArffFile, testArffFile, xmlFile);
		ds.name = name;

		int beamSize = 2;
		int mDepth = 3;

		//runLearning(ds, beamSize, mDepth);
		//RegressionCostFuncLearning.runLearningExhaustive(ds);
		
		ArrayList<CostFunction> allHeuristics = null;
		try {
			 allHeuristics = BreathFirstSearcher.heuristicDaggerIterations(ds.getTrainExamples(), ds.getTestExamples(), beamSize, mDepth, ds.name, 0);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		// test
		testGenerationLoss(ds, beamSize, mDepth, allHeuristics.get(0));

		System.out.println("All Done!");
	}
	
	public static void runLearning(Dataset ds, int beamSize, int mDepth) {
		try {
			BreathFirstSearcher.heuristicDaggerIterations(ds.getTrainExamples(), ds.getTestExamples(), beamSize, mDepth, ds.name, 0);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}


	public static void testGenerationLoss(Dataset ds, int beamSize, int mDepth, CostFunction hfunc) {

		BreathFirstSearcher bsearcher = new BreathFirstSearcher();

		ArrayList<Example> testExs = ds.getTestExamples();
		for (int i = 0; i < testExs.size(); i++) {
			Example ex = testExs.get(i);
			OldSearchState init = OldSearchState.getAllZeroState(ex.labelDim());
			OldSearchState bestState = bsearcher.beamSearchTestOracleSelection(ex, init,  beamSize, mDepth, hfunc);
			ex.predictOutput = bestState.getOutput(); // assign prediction
		}
		System.out.println("Err count = " + bsearcher.herrCnt);
		System.out.println("Cor count = " + bsearcher.crrCnt);

		// scoring!
		MultiLabelEvaluator evaluator = new MultiLabelEvaluator();
		evaluator.evaluationDataSet(ds.name, ds.getTestExamples());

		System.out.println("Done predicting!");
	}

}
