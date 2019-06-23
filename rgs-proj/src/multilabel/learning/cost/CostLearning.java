package multilabel.learning.cost;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;

import ciir.umass.edu.eval.Evaluator;
import multilabel.data.Dataset;
import multilabel.evaluation.LossFunction;
import multilabel.evaluation.MultiLabelEvaluator;
import multilabel.instance.Example;
import multilabel.instance.Featurizer;
import multilabel.instance.OldWeightVector;
import multilabel.learning.RegressionCostFuncLearning;
import multilabel.learning.StructOutput;
import multilabel.learning.search.BreathFirstSearcher;
import multilabel.learning.search.OldSearchState;


//  /scratch/large_multi-label/largemultilabel_my/my_largelabel/heur_rank/medical_huer_rank_lambdamart_iter0_b10.txt


public class CostLearning {
	
	public static void costLearning(Dataset ds, ArrayList<Example> trainExs, ArrayList<Example> testExs, int beamSize, int mDepth, CostFunction hfunc) {

		BreathFirstSearcher bsearcher = new BreathFirstSearcher();

		HashMap<Example, HashSet<OldSearchState>> allTrainStates = new HashMap<Example, HashSet<OldSearchState>>(); // training
		HashMap<Example, HashSet<OldSearchState>> allTestStates = new HashMap<Example, HashSet<OldSearchState>>(); // testing
		
		// training states
		for (int i = 0; i < trainExs.size(); i++) {
			Example ex = trainExs.get(i);
			OldSearchState init = OldSearchState.getAllZeroState(ex.labelDim());
			HashSet<OldSearchState> genTrainStates = bsearcher.beamSearchGenerationOnly(ex, init,  beamSize, mDepth, hfunc);
			// store generate states
			allTrainStates.put(ex, genTrainStates);
		}
		
		// testing states
		for (int j = 0; j < testExs.size(); j++) {
			Example ex = testExs.get(j);
			OldSearchState init = OldSearchState.getAllZeroState(ex.labelDim());
			HashSet<OldSearchState> genTestStates = bsearcher.beamSearchGenerationOnly(ex, init,  beamSize, mDepth, hfunc);
			// store generate states
			allTestStates.put(ex, genTestStates);
		}

		
		System.out.println("Train cost function...");
		CostFunction cf = null;
		try {
			cf = trainCostFunction(allTrainStates, allTestStates, ds.name);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		System.out.println("Done cost function learning.");
		
		// have a test
		for (int j = 0; j < testExs.size(); j++) {
			Example ex = testExs.get(j);
			OldSearchState init = OldSearchState.getAllZeroState(ex.labelDim());
			OldSearchState bestState = bsearcher.beamSearchTest(ex, init,  beamSize, mDepth, hfunc, cf);
			ex.predictOutput = bestState.getOutput();
		}
		
		// scoring!
		MultiLabelEvaluator evaluator = new MultiLabelEvaluator();
		evaluator.evaluationDataSet("CostFunc", testExs);
		
	}
	
	// Searcher for testing!
	public static CostFunction trainCostFunction(HashMap<Example, HashSet<OldSearchState>> trainStates, 
			                                     HashMap<Example, HashSet<OldSearchState>> testStates, 
			                                     String dsName) throws FileNotFoundException {
		/////// TRAIN ////////

		System.out.println("Dumping Features!");

		// 1) dump file
		String costFolder = "cost_rank";
		String trainFeatFn = costFolder + "/" + dsName + "_cost_train_rankfeat.txt";
		String testFeatFn = costFolder + "/" + dsName + "_cost_test_rankfeat.txt";
		PrintWriter trainWriter = new PrintWriter(trainFeatFn);
		PrintWriter testWriter = new PrintWriter(testFeatFn);
		// dump train
		dumpRankLearning(trainStates, trainWriter);
		// dump validation
		dumpRankLearning(testStates, testWriter);
		trainWriter.close();
		testWriter.close();

		
		// 2) prepare running cmd
		String modelFn = dsName + "_cost_rank_lambdamart.txt";
		String modelPath = costFolder + "/" + modelFn;
		String basicCmd = "-sparse -tree 1000 -leaf 100 -shrinkage 0.1 -tc -1 -mls 1 -estop 50 -ranker 6";
		String metricCmd = " -metric2t " + "P@1";
		String fileCmd = " -train " + trainFeatFn + " -validate " + testFeatFn + " -save " + modelPath;
		String trainCmd = basicCmd + metricCmd + fileCmd;

		// 3) run training
		System.out.println("Call command:\n" + trainCmd);
		String[] cmds = trainCmd.split("\\s+");
		Evaluator.main(cmds);


		////// TEST //////////
		//System.out.println("Start testing!");

		
		//// construct return cost function
		RankingCostFunction cf = new RankingCostFunction(new Featurizer());
		cf.loadModel(modelPath);
		
		return cf;
	}

/*	
	public static final Comparator<Double> AECENT_ORDER = new Comparator<Double>() {
		public int compare(Double d1, Double d2) {
			if (d1 <= d2) return -1;
			return 1;
		}
	};
	
	public static final Comparator<Double> DECENT_ORDER = new Comparator<Double>() {
		public int compare(Double d1, Double d2) {
			if (d1 >= d2) return -1;
			return 1;
		}
	};
*/
	public static void dumpRankLearning(HashMap<Example, HashSet<OldSearchState>> genStates, PrintWriter writer) {

		int exCnt = 0;
		for (Example ex : genStates.keySet()) {
			exCnt++;
			StructOutput truth = ex.getGroundTruthOutput();
			
			HashSet<Double> scores = new HashSet<Double>();
			HashSet<OldSearchState> states = genStates.get(ex);
			for (OldSearchState s : states) {
				
				// the true score that we want to do regression on
				s.trueAccuracy = getTrainTrueAccuracy(s.getOutput(), truth);
				
				scores.add(s.trueAccuracy);
			}
			ArrayList<Double> scList = new ArrayList<Double>(scores);

			// sort true scores
			Collections.sort(scList, RegressionCostFuncLearning.AECENT_ORDER);
			HashMap<Double, Integer> scRank = new HashMap<Double, Integer>();
			for (int i = 0; i < scList.size(); i++) {
				scRank.put(scList.get(i), (i));
			}

			/////////////////////////////////
			Featurizer featurizer = new Featurizer();
			for (OldSearchState s : states) {
				int rk = scRank.get(s.trueAccuracy);
				int rank = 0;
				if (rk == (scList.size() - 1)) { // the first
					rank = 1;
				}
				OldWeightVector fv = featurizer.getFeatureVector(ex, s.getOutput());
				writer.println(rank + " " + "qid:" + exCnt + " " + fv.toSparseRanklibStr());
				//writer.println(rk + " " + "qid:" + exCnt + " " + fv.toSparseRanklibStr());
				//System.out.println(rk + " " + s.trueAccuracy + " qid:" + exCnt + " " + fv.toSparseRanklibStr());
				System.out.println(rank + " " + s.trueAccuracy + " qid:" + exCnt);
			}
		}

	}
	
	// it is actually ground truth accuracy
	public static double getTrainTrueAccuracy(StructOutput pred, StructOutput truth) {
		double acc = LossFunction.computeHammingAccuracy(pred, truth);
		return acc;
	}
	
}
