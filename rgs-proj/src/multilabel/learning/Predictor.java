package multilabel.learning;

import multilabel.instance.Example;

import java.util.ArrayList;

import multilabel.learning.cost.CostFunction;
import multilabel.learning.search.ExhaustiveSearcher;
import multilabel.learning.search.OldGreedySearcher;
import multilabel.learning.search.OldSearchState;
import multilabel.learning.search.SearcherInfo;
import multilabel.data.Dataset;
import multilabel.evaluation.MultiLabelEvaluator;

/**
 * Search based structure predictor, used in testing only
 * @author machao
 *
 */
public class Predictor {
	
	public static void doTesting(Dataset dataset, CostFunction cfunction, SearcherInfo settings) {
		
		OldGreedySearcher searcher = new OldGreedySearcher();

		ArrayList<Example> testExs = dataset.getTestExamples();
		for (int i = 0; i < testExs.size(); i++) {
			Example ex = testExs.get(i);
			OldSearchState bestState = searcher.beamSearchTest(ex, OldSearchState.getAllZeroState(ex.labelDim()), 1, cfunction);
			ex.predictOutput = bestState.getOutput(); // assign prediction
		}
		
		// scoring!
		MultiLabelEvaluator evaluator = new MultiLabelEvaluator();
		
		
		//evaluator.evaluationDataSet(dataset);
		evaluator.evaluationDataSet(dataset.name, dataset.getTestExamples());
		
		System.out.println("Done predicting!");
	}
	
	
	public static void doTestingExhaustive(Dataset dataset, CostFunction cfunction, int depth, SearcherInfo settings) {
		
		ExhaustiveSearcher searcher = new ExhaustiveSearcher();

		ArrayList<Example> testExs = dataset.getTestExamples();
		for (int i = 0; i < testExs.size(); i++) {
			Example ex = testExs.get(i);
			OldSearchState bestState = searcher.DFSearchTest(ex, OldSearchState.getAllZeroState(ex.labelDim()), depth, cfunction);
			ex.predictOutput = bestState.getOutput(); // assign prediction
		}
		
		// scoring!
		MultiLabelEvaluator evaluator = new MultiLabelEvaluator();
		
		
		//evaluator.evaluationDataSet(dataset);
		evaluator.evaluationDataSet(dataset.name, dataset.getTestExamples());
		
		System.out.println("Done predicting!");
	}
	
	
	
	public static void testCompleteWorkStream() {
		
	}
}
