package multilabel.learning.heuristic;


import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;

import multilabel.instance.Example;
import multilabel.instance.Featurizer;
import multilabel.instance.OldWeightVector;
import multilabel.learning.RegressionCostFuncLearning;
import multilabel.learning.StructOutput;
import multilabel.learning.search.OldGreedySearcher;
import multilabel.learning.search.OldSearchState;

/**
 * For heuristic function training only
 */
public class StateRepository {
	
	Example example;
	int nLabel;
	
	public StateRepository(Example ex) {
		example = ex;
		statesEachStep = new HashMap<Integer, ArrayList<OldSearchState>>();
	}
	
	public Example getExample() {
		return example;
	}
	
	HashMap<Integer, ArrayList<OldSearchState>> statesEachStep;
	
	public void storeState(int depth, ArrayList<OldSearchState> states) {
		if (statesEachStep.containsKey(depth)) {
			throw new RuntimeException("depth "+depth+" states has been existed!");
		}
		statesEachStep.put(depth, states);
	}
	
	public ArrayList<OldSearchState> getStateAtDepth(int depth) {
		ArrayList<OldSearchState> dStates = statesEachStep.get(depth);
		return dStates;
	}
	
	public HashSet<OldSearchState> getAllStates() {
		HashSet<OldSearchState> allStates = new HashSet<OldSearchState>();
		for (ArrayList<OldSearchState> dStates : statesEachStep.values()) {
			allStates.addAll(dStates);
		}
		return allStates;
	}

	//public void computeGenerationLoss() {
	//}
	
	public void dumpRankingLists(ArrayList<OldSearchState> stateList, int qid, PrintWriter writer, int d) {
		
		HashSet<Double> scores = new HashSet<Double>();
		for (OldSearchState s : stateList) {
			scores.add(s.trueAccuracy);
		}
		ArrayList<Double> scList = new ArrayList<Double>(scores);

		// sort true scores
		Collections.sort(scList, RegressionCostFuncLearning.DECENT_ORDER);
		HashMap<Double, Integer> scRank = new HashMap<Double, Integer>();
		for (int i = 0; i < scList.size(); i++) {
			scRank.put(scList.get(i), (i));
		}

		/////////////////////////////////
		Featurizer featurizer = new Featurizer();
		for (OldSearchState s : stateList) {
			int rk = scRank.get(s.trueAccuracy);
			int rank = 0;
			if (rk == 0) { // the first
				rank = 1;
			}
			OldWeightVector fv = featurizer.getFeatureVector(example, s.getOutput());
			writer.println(rank + " " + "qid:" + qid + " " + fv.toSparseRanklibStr());
			//writer.println(rk + " " + "qid:" + exCnt + " " + fv.toSparseRanklibStr());
			//System.out.println(rk + " " + s.trueAccuracy + " qid:" + qid + " " + fv.toSparseRanklibStr());
			StructOutput gold = OldGreedySearcher.extractGoldOutput(example);
			//System.out.println(gold.toString() +"  " + s.getOutput().toString() + " depth = " + d + ":  " + rank + " " + s.trueAccuracy + " qid:" + qid + " ");
		}
		
	}
}
