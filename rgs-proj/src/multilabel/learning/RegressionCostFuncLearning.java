package multilabel.learning;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;

import multilabel.learning.cost.CostFunction;
import multilabel.learning.search.ExhaustiveSearcher;
import multilabel.learning.search.OldGreedySearcher;
import multilabel.learning.search.OldSearchState;
import  weka.core.Attribute;
import  weka.core.FastVector;
import  weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.converters.ArffSaver;
import multilabel.data.Dataset;
import multilabel.evaluation.LossFunction;
import multilabel.evaluation.MultiLabelEvaluator;
import multilabel.instance.Example;
import multilabel.instance.Featurizer;
import multilabel.instance.Label;
import multilabel.instance.OldWeightVector;

public class RegressionCostFuncLearning {

	public static void runLearning(Dataset ds) { // on trajectory
		
		RegressionCostFuncLearning learner = new RegressionCostFuncLearning();
		
		ArrayList<Example> trainExmps = ds.getTrainExamples();
		try {
			learner.onTrajectoryTrain(trainExmps, 1, ds.name + "_train");
			learner.onTrajectoryTrain(ds.getTestExamples(), 1, ds.name + "_test");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	public static void runLearningOffTraj(Dataset ds, CostFunction firstCost) {
		
		RegressionCostFuncLearning learner = new RegressionCostFuncLearning();
		
		ArrayList<Example> trainExmps = ds.getTrainExamples();
		try {
			learner.DaggerIterations(trainExmps, 1, firstCost, ds.name + "_train");
			learner.DaggerIterations(ds.getTestExamples(), 1, firstCost, ds.name + "_test");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	public static void runLearningExhaustive(Dataset ds) {

		RegressionCostFuncLearning learner = new RegressionCostFuncLearning();

		ArrayList<Example> trainExmps = ds.getTrainExamples();
		learner.bruteForceGenerationLearning(trainExmps, 3, ds.name + "_train");
		learner.bruteForceGenerationLearning(ds.getTestExamples(), 3, ds.name + "_test");
	}
	
	
	// Dagger outputs generating
	public void onTrajectoryTrain(ArrayList<Example> exs, int iterations, String dumpFileName) throws FileNotFoundException {

		int beamSize = 1;
		OldGreedySearcher searcher = new OldGreedySearcher();

		ArrayList<SparseInstance> sparseInsts = new ArrayList<SparseInstance>();
		HashMap<Example, HashSet<OldSearchState>> generateStates = new HashMap<Example, HashSet<OldSearchState>>();

		int exCnt = 0;
		for (Example ex : exs) {
			exCnt++;

			OldSearchState init = OldSearchState.getAllZeroState(ex.labelDim());
			HashSet<OldSearchState> uncoveredStates = new HashSet<OldSearchState>();
			OldSearchState finalState = searcher.beamSearch(ex, init, beamSize, true, null, uncoveredStates);
			
			System.out.println("Done Example " + exCnt);

			// dump generate
			dumpGeneratedFeatures(ex, uncoveredStates, sparseInsts);
			generateStates.put(ex, uncoveredStates);
		}

		FastVector attrs = getAttributes(exs.get(0).getFeatureDimen());
		Instances wekaDataSet = new Instances("regression_learning", attrs, sparseInsts.size());
		wekaDataSet.setClassIndex(attrs.size() - 1);
		for (int j = 0; j < sparseInsts.size(); j++) {
			wekaDataSet.add(sparseInsts.get(j));
		}
		saveArff(wekaDataSet, dumpFileName + "_regression_iter" + String.valueOf(0) + ".arff");

		// dump ranklib features
		dumpRankLearning(generateStates, dumpFileName + "_ranking_iter" + String.valueOf(0) + ".txt");
	}
	
	
	
	// Dagger outputs generating
	public void DaggerIterations(ArrayList<Example> exs, int iterations, CostFunction existCfunc, String dumpFileName) throws FileNotFoundException {
		
		int beamSize = 1;
		boolean onTraj = false;
		OldGreedySearcher searcher = new OldGreedySearcher();

		for (int iter = 1; iter <= iterations; iter++) {
			
			ArrayList<SparseInstance> sparseInsts = new ArrayList<SparseInstance>();
			HashMap<Example, HashSet<OldSearchState>> generateStates = new HashMap<Example, HashSet<OldSearchState>>();

			int exCnt = 0;
			for (Example ex : exs) {
				exCnt++;

				OldSearchState init = OldSearchState.getAllZeroState(ex.labelDim());
				HashSet<OldSearchState> uncoveredStates = new HashSet<OldSearchState>();
				searcher.beamSearch(ex, init, beamSize, onTraj, existCfunc, uncoveredStates);

				System.out.println("Done Example " + exCnt);

				// dump generate
				dumpGeneratedFeatures(ex, uncoveredStates, sparseInsts);
				generateStates.put(ex, uncoveredStates);
			}

			FastVector attrs = getAttributes(exs.get(0).getFeatureDimen());
			Instances wekaDataSet = new Instances("regression_learning", attrs, sparseInsts.size());
			wekaDataSet.setClassIndex(attrs.size() - 1);
			for (int j = 0; j < sparseInsts.size(); j++) {
				wekaDataSet.add(sparseInsts.get(j));
			}
			saveArff(wekaDataSet, dumpFileName + "_regression_iter" + String.valueOf(iter) + ".arff");

			// dump ranklib features
			dumpRankLearning(generateStates, dumpFileName + "_ranking_iter" + String.valueOf(iter) + ".txt");
		}
	}
	
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
	
	public void dumpRankLearning(HashMap<Example, HashSet<OldSearchState>> genStates, String fn) {

		try {
			PrintWriter writer = new PrintWriter(fn);

			int exCnt = 0;
			for (Example ex : genStates.keySet()) {
				exCnt++;
				//ArrayList<SearchState> stateList = new ArrayList<SearchState>(genStates.get(ex));

				HashSet<Double> scores = new HashSet<Double>();
				HashSet<OldSearchState> states = genStates.get(ex);
				for (OldSearchState s : states) {
					scores.add(s.trueAccuracy);
				}
				ArrayList<Double> scList = new ArrayList<Double>(scores);

				// sort true scores
				Collections.sort(scList, AECENT_ORDER);
				HashMap<Double, Integer> scRank = new HashMap<Double, Integer>();
				for (int i = 0; i < scList.size(); i++) {
					scRank.put(scList.get(i), (i));
				}

				/////////////////////////////////
				Featurizer featurizer = new Featurizer();
				for (OldSearchState s : states) {
					int rk = scRank.get(s.trueAccuracy);
					//int rank = 0;
					//if (rk == 0) { // the first
					//	rank = 1;
					//}
					OldWeightVector fv = featurizer.getFeatureVector(ex, s.getOutput());
					//writer.println(rank + " " + "qid:" + exCnt + " " + fv.toSparseRanklibStr());
					writer.println(rk + " " + "qid:" + exCnt + " " + fv.toSparseRanklibStr());
					System.out.println(rk + " " + s.trueAccuracy + " qid:" + exCnt + " " + fv.toSparseRanklibStr());
				}

			}
			writer.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

	}
	
	public void testOnTrainingData(Dataset ds, CostFunction cf) {
	
		ArrayList<Example> trainExmps = ds.getTrainExamples();
		
		int beamSize = 1;
		OldGreedySearcher searcher = new OldGreedySearcher();
		
		int exCnt = 0;
		for (Example ex : trainExmps) {
			exCnt++;
			
			OldSearchState init = OldSearchState.getAllZeroState(ex.labelDim());
			HashSet<OldSearchState> uncoveredStates = new HashSet<OldSearchState>();
			searcher.beamSearch(ex, init, beamSize, true, null, uncoveredStates);
			System.out.println("Finish search on example " + exCnt);
			
			for (OldSearchState s : uncoveredStates) {
				s.predScore = cf.getCost(s, ex);
			}
			
			// do selection!
			OldSearchState bestState = searcher.selectBest(ex, uncoveredStates, null, cf);
			ex.predictOutput = bestState.getOutput();
		}
		
		MultiLabelEvaluator eval = new MultiLabelEvaluator();
		eval.evaluationDataSet(ds.name, ds.getTrainExamples());
		
	}
	
	public void bruteForceGenerationLearning(ArrayList<Example> exs, int maxDepth, String dumpFileName)  {

		ExhaustiveSearcher searcher = new ExhaustiveSearcher();

		ArrayList<SparseInstance> sparseInsts = new ArrayList<SparseInstance>();
		HashMap<Example, HashSet<OldSearchState>> generateStates = new HashMap<Example, HashSet<OldSearchState>>();

		int exCnt = 0;
		for (Example ex : exs) {
			exCnt++;

			OldSearchState init = OldSearchState.getAllZeroState(ex.labelDim());
			HashSet<OldSearchState> uncoveredStates = new HashSet<OldSearchState>();
			searcher.DFSearchTrain(ex, init, maxDepth, uncoveredStates);
			
			System.out.println("Done Example " + exCnt);

			// dump generate
			dumpGeneratedFeatures(ex, uncoveredStates, sparseInsts);
			generateStates.put(ex, uncoveredStates);
		}

		FastVector attrs = getAttributes(exs.get(0).getFeatureDimen());
		Instances wekaDataSet = new Instances("regression_learning", attrs, sparseInsts.size());
		wekaDataSet.setClassIndex(attrs.size() - 1);
		for (int j = 0; j < sparseInsts.size(); j++) {
			wekaDataSet.add(sparseInsts.get(j));
		}
		saveArff(wekaDataSet, dumpFileName + "_regression_bruteforce.arff");
		
		// dump ranklib features
		dumpRankLearning(generateStates, dumpFileName + "_ranking_bruteforce.txt");
	}
	
	
	public void saveArff(Instances trInsts, String fn) {
		ArffSaver saver1 = new ArffSaver();
		saver1.setInstances(trInsts);
		try {
			saver1.setFile(new File(fn));
			saver1.setDestination(new File(fn));
			saver1.writeBatch();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void dumpGeneratedFeatures(Example ex, HashSet<OldSearchState> generatedStates, ArrayList<SparseInstance> sparseInsts) {
		
		Featurizer featurizer = new Featurizer();
		for (OldSearchState state : generatedStates) {
			OldWeightVector fv = featurizer.getFeatureVector(ex, state.getOutput());
			double loss = state.trueAccuracy; // hamming accuracy
			
			SparseInstance inst = getWekaInstance(fv, loss);
			sparseInsts.add(inst);
		}
		
	}
	
	public static FastVector getAttributes(int len) { //(Example ex) {
		Attribute ClassAttribute = new Attribute("theClass");

		FastVector attrs = new FastVector(len + 1);
	    for (int i = 0; i < len; i++) {
	      String attrName = "Attr-" + String.valueOf(i);
	      attrs.addElement(new Attribute(attrName));
	    }
	    attrs.addElement(ClassAttribute);
	    
	    return attrs;
	}
	
	public SparseInstance getWekaInstance(OldWeightVector fv, double acc) {
		int dimen = fv.getMaxLength() + 1;

		ArrayList<Integer> idxs = new ArrayList<Integer>(fv.getKeys());
		Collections.sort(idxs);
		
		int[] featIdx = new int[idxs.size() + 1];
		double[] featVals = new double[idxs.size() + 1];

		for (int i = 0; i < idxs.size(); i++) {
			int idx = idxs.get(i).intValue();
			featIdx[i] = idx;
			featVals[i] = fv.get(idx);
		}

		featIdx[idxs.size()] = dimen - 1;
		featVals[idxs.size()] = acc;
		
		SparseInstance wekaInst = new SparseInstance(1.0, featVals, featIdx, dimen);
		return wekaInst;
	}
}
