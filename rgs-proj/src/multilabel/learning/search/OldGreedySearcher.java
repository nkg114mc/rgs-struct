package multilabel.learning.search;

import  java.util.ArrayList;
import  java.util.Collections;
import  java.util.Comparator;
import  java.util.HashSet;

import multilabel.learning.StructOutput;
import multilabel.learning.cost.CostFunction;
import multilabel.learning.cost.RegressionCostFunction;
import multilabel.evaluation.LossFunction;
import multilabel.instance.Example;
import multilabel.instance.Featurizer;
import multilabel.instance.Label;

public class OldGreedySearcher {
	
	// some setting
	boolean applyPruning = true;
	
	
	// general action generator
/*
	public ArrayList<Action> generateActions(SearchState state) {
		
		ArrayList<Action> actions = new ArrayList<Action>();

		StructOutput currentOut = state.getOutput();
		for (int i = 0; i < currentOut.size(); i++) {
			for (int j = 0; j < StructOutput.PossibleValues.length; j++) {
				
				if (currentOut.getValue(i) != StructOutput.PossibleValues[j]) {
					Action act = new Action(i, StructOutput.PossibleValues[j]);
					act.oldValue = currentOut.getValue(i);
					actions.add(act);
					//act.print();
				}
			}
		}
		return actions;
	}
*/
	public ArrayList<Action> generateActions(OldSearchState state) {
		return generateActionsWithSetting(state, true);
	}
	
	public ArrayList<Action> generateActionsWithSetting(OldSearchState state, boolean doPruning) {
		
		ArrayList<Action> actions = new ArrayList<Action>();

		StructOutput currentOut = state.getOutput();
		for (int i = 0; i < currentOut.size(); i++) {
			/*if (doPruning) {
				if () {
					continue; // no need for this position
				}
			}*/
			
			for (int j = 0; j < StructOutput.PossibleValues.length; j++) {
				if (currentOut.getValue(i) != StructOutput.PossibleValues[j]) {
					Action act = new Action(i, StructOutput.PossibleValues[j]);
					act.oldValue = currentOut.getValue(i);
					actions.add(act);
					//act.print();
				}
			}
		}
		return actions;
	}
	
	// ground truth
	public static StructOutput extractGoldOutput(Example example) {
		StructOutput goldOutput = new StructOutput(example.labelDim());
		ArrayList<Label> labels = example.getLabel();
		for (int i = 0; i < example.labelDim(); i++) {
			goldOutput.setValue(labels.get(i).originIndex, labels.get(i).value);
		}
		return goldOutput;
	}
	
	
	public static OldSearchState performActionNewState(OldSearchState state, Action act) {
		OldSearchState newState = state.getSelfCopy();
		newState.getOutput().setValue(act.position, act.newValue);
		return newState;
	}
	
	
	// it is actually ground truth accuracy
	public double getTrainLoss(StructOutput pred, StructOutput truth) {
		double acc = LossFunction.computeHammingAccuracy(pred, truth);
		return acc;
	}
	
	// beam search
	public OldSearchState beamSearch(Example exmp, OldSearchState initState, int beamSize, boolean onTraj, CostFunction costFunc,
			                      HashSet<OldSearchState> hashTable) {

		int maxDepth = initState.size() / 2;
		
		SearchBeam beam = new SearchBeam(beamSize);
		HashTable uncoveredTree = new HashTable(exmp.labelDim() + 2, 3);
		Featurizer featurizer = new Featurizer();
		OldSearchState[] trajectory = new OldSearchState[maxDepth * 2];
		OldSearchState currentState = null;
		
		final StructOutput goldOut = extractGoldOutput(exmp);

		
		// for initial state /////////////
		OldSearchState initCopy = initState.getSelfCopy();
		initCopy.predScore = 0;
		if (!onTraj) { initCopy.predScore = costFunc.getCost(initCopy, exmp); }
		initCopy.trueAccuracy = getTrainLoss(initState.getOutput(), goldOut); // LossFunction.computeCorrectCnt(initCopy.getOutput(), goldOut);
		beam.insert(initCopy);
		uncoveredTree.insert(initCopy);
		//////////////////////////////////
		
		//System.out.println("====== Init ======");
		//initState.print();
		//currentState.print();
		
		for (int depth = 0; depth <= maxDepth; depth++) {
			
			////// pop best
			currentState = beam.popBest(onTraj);
			trajectory[depth] = currentState;
			System.out.println("depth " + depth + " acc = " + currentState.trueAccuracy + " pred = " + currentState.predScore);
			//System.out.println(goldOut.toString());
			
			if (depth == maxDepth) {
				System.out.println("Stop search: reaching max depth!");
				break;
			}
			
			//if (currentState.trueAccuracy == 1.0) {
			//	System.out.println("Stop search: get perfect output in training!");
			//	break;
			//}
			
			///// Start expanding
			/////////////////////////////////////////////////////////////////
			
			ArrayList<Action> acts = generateActions(currentState);
			ArrayList<OldSearchState> childrenStates = new ArrayList<OldSearchState>();
			for (Action act : acts) {
				OldSearchState newState = performActionNewState(currentState, act);
				//act.print();
				//newState.print();
				if (!uncoveredTree.probeExistence(newState)) {
					
					// compute true loss
					newState.trueAccuracy = getTrainLoss(newState.getOutput(), goldOut);// LossFunction.computeHammingAccuracy(newState.getOutput(), goldOut); // LossFunction.computeCorrectCnt(newState.getOutput(), goldOut); 
					// compute our predict score
					newState.fv = null;//featurizer.getFeatureVector(exmp, newState.getOutput());
					newState.predScore = 0;
					if (!onTraj) { newState.predScore = costFunc.getCost(newState, exmp); } // scoring if it is off-trajectory
					
					//System.out.println("acc =  " + newState.trueAccuracy + " pred = " + newState.predScore);
					
					childrenStates.add(newState);
					uncoveredTree.insert(newState);
				}
			}
			System.out.println("branch factor =  " + childrenStates.size());
			System.out.println("Hash table size =  " + uncoveredTree.size());
			
			// need to continue?
			if (childrenStates.size() == 0) {
				break;
			}
			
			
			// check children
			beam.insertAll(childrenStates);
			beam.dropTail(onTraj);
			/////////////////////////////////////////////////////////////////
		}
		
		// return the generated outputs
		hashTable.clear();
		hashTable.addAll(uncoveredTree.getAllElementsHashSet());
		
		
		//// select best
		OldSearchState bestState = null;
		
		
		return currentState;
	}

	
	
	
	
	//////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////
	
	
	
	
	// Searcher for testing!
	public OldSearchState beamSearchTest(Example exmp, OldSearchState initState, int beamSize, CostFunction cfunc) {//, HashSet<SearchState> hashTable) {

		int maxDepth = initState.size() / 2;
		final boolean neverTraj = false; // can not do this during testing
		

		SearchBeam beam = new SearchBeam(beamSize);
		HashTable uncoveredTree = new HashTable(exmp.labelDim() + 2, 3);
		Featurizer featurizer = new Featurizer();
		OldSearchState[] trajectory = new OldSearchState[maxDepth * 2];
		OldSearchState currentState = null;
		
		final StructOutput goldOut = extractGoldOutput(exmp);

		
		///////////
		/////////// GENERATION
		///////////
		
		// for initial state /////////////
		OldSearchState initCopy = initState.getSelfCopy();
		initCopy.predScore = 0;
		initCopy.trueAccuracy = LossFunction.computeHammingAccuracy(initState.getOutput(), goldOut); // LossFunction.computeCorrectCnt(initCopy.getOutput(), goldOut);
		beam.insert(initCopy);
		uncoveredTree.insert(initCopy);
		//////////////////////////////////
		
		for (int depth = 0; depth <= maxDepth; depth++) {
			
			////// pop best
			currentState = beam.popBest(neverTraj);
			trajectory[depth] = currentState;
			System.out.println("depth " + depth + " acc = " + currentState.trueAccuracy + " pred = " + currentState.predScore);
			
			if (depth == maxDepth) {
				System.out.println("Stop search: reaching max depth!");
				break;
			}
			
			//if (currentState.trueAccuracy == 1.0) {
			//	System.out.println("Stop search: get perfect output in training!");
			//	break;
			//}
			
			ArrayList<Action> acts = generateActions(currentState);
			ArrayList<OldSearchState> childrenStates = new ArrayList<OldSearchState>();
			for (Action act : acts) {
				OldSearchState newState = performActionNewState(currentState, act);
				//act.print();
				//newState.print();
				if (!uncoveredTree.probeExistence(newState)) {
					
					// compute true loss
					newState.trueAccuracy = LossFunction.computeHammingAccuracy(newState.getOutput(), goldOut); // LossFunction.computeCorrectCnt(newState.getOutput(), goldOut); 
					// compute our predict score
					newState.fv = null;//featurizer.getFeatureVector(exmp, newState.getOutput());
					newState.predScore = cfunc.getCost(newState, exmp);
					
					System.out.println("acc =  " + newState.trueAccuracy + " pred = " + newState.predScore);
					
					childrenStates.add(newState);
					uncoveredTree.insert(newState);
				}
			}
			System.out.println("branch factor =  " + childrenStates.size());
			System.out.println("Hash table size =  " + uncoveredTree.size());
			
			// need to continue?
			if (childrenStates.size() == 0) {
				break;
			}
			
			// check children
			beam.insertAll(childrenStates);
			beam.dropTail(neverTraj);
			/////////////////////////////////////////////////////////////////
		}
		
		//////
		////// SELECTION
		//////
		
		//RegressionCostFunction cf = new RegressionCostFunction(new Featurizer());
		//cf.loadModel("emotions_regression_offtraj1.model");
		
		//// select best from the generated outputs
		OldSearchState bestState = selectBest(exmp, uncoveredTree.getAllElementsHashSet(), trajectory, cfunc);
		return bestState;
	}
	

	
	// optimization step
	public OldSearchState selectBest(Example exmp, HashSet<OldSearchState> generatedStates, OldSearchState[] trajectory, CostFunction costf) {
		
		double bestScore = -Double.MAX_VALUE;
		OldSearchState bestState = null;
		
		double bestAcc = -10000;
		OldSearchState bestGeneratedState = null;
		
		for (OldSearchState s : generatedStates) {
			
			s.predScore = costf.getCost(s, exmp);

			if (s.predScore > bestScore) {
				bestScore = s.predScore;
				bestState = s;
			}
			/////////////////////////////
			if (s.trueAccuracy > bestAcc) {
				bestAcc = s.trueAccuracy;
				bestGeneratedState = s;
			}
		}
		
		System.out.println("-- bestGeneratedAcc =  " + bestGeneratedState.trueAccuracy);
		System.out.println("-- ourSelectedAcc =  " + bestState.trueAccuracy);
		
		plotGeneratedOutputs(generatedStates);
		
		//return bestGeneratedState;
		return bestState;
	}
	
	public void plotGeneratedOutputs(HashSet<OldSearchState> generatedStates) {
		
		ArrayList<OldSearchState> statelist = new ArrayList<OldSearchState>(generatedStates);
		Collections.sort(statelist, SLOTSTATE_ORDER);
		
		for (int i = 0; i < statelist.size(); i++) {
			OldSearchState s = statelist.get(i);
			System.out.println(i + "," + s.trueAccuracy + "," + s.predScore);
		}
		
	}
	
	static final Comparator<OldSearchState> SLOTSTATE_ORDER = new Comparator<OldSearchState>() {
		public int compare(OldSearchState s1, OldSearchState s2) {
			if (s1.trueAccuracy > s2.trueAccuracy) return -1;
			if (s1.trueAccuracy < s2.trueAccuracy) return 1;
			if (s1.trueAccuracy == s2.trueAccuracy) {
				if (s1.predScore >= s2.predScore) {
					return -1;
				}
				return 1;
			}
			return 1;
		}
	};

}
