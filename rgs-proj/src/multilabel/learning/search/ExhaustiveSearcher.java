package multilabel.learning.search;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;

import multilabel.evaluation.LossFunction;
import multilabel.instance.Example;
import multilabel.instance.Featurizer;
import multilabel.instance.Label;
import multilabel.learning.StructOutput;
import multilabel.learning.cost.CostFunction;

public class ExhaustiveSearcher {

	// some setting
	boolean applyPruning = true;
	
	// only flip 0 to 1
	public ArrayList<Action> generateActionsZeroToOne(OldSearchState state, Example ex,  boolean doPruning) {
		
		ArrayList<Action> actions = new ArrayList<Action>();

		StructOutput currentOut = state.getOutput();
		for (int i = 0; i < currentOut.size(); i++) {
			Label lb = ex.getLabelGivenIndex(i);
			if (!lb.isPruned) {
				if (currentOut.getValue(i) == 0) { // a 0 bit
					Action act = new Action(i, 1);
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
	
	
	public OldSearchState performActionNewState(OldSearchState state, Action act) {
		OldSearchState newState = state.getSelfCopy();
		newState.getOutput().setValue(act.position, act.newValue);
		return newState;
	}
	
	
	// it is actually ground truth accuracy
	public double getTrainLoss(StructOutput pred, StructOutput truth) {
		double acc = LossFunction.computeHammingAccuracy(pred, truth);
		return acc;
	}
	
	
	public void DFSearchTrain(Example exmp, OldSearchState initState, int maxDepth, // CostFunction costFunc,
                              HashSet<OldSearchState> hashTable) {
		
		HashTable uncoveredTree = new HashTable(exmp.labelDim() + 2, 3);
		
		
		// for initial state /////////////
		OldSearchState initCopy = initState.getSelfCopy();
		final StructOutput goldOut = extractGoldOutput(exmp);
		initCopy.predScore = 0;
		initCopy.trueAccuracy = LossFunction.computeHammingAccuracy(initState.getOutput(), goldOut); // LossFunction.computeCorrectCnt(initCopy.getOutput(), goldOut);
		uncoveredTree.insert(initCopy);
		//////////////////////////////////
		
		// GENERATION
		DFSearchGeneration(exmp, initCopy,  maxDepth, uncoveredTree, goldOut);
		// return the generated outputs
		hashTable.clear();
		hashTable.addAll(uncoveredTree.getAllElementsHashSet());
		
		System.out.println("Generate " + uncoveredTree.size() + " outputs");
		
		// SELECTION?
		plotGeneratedOutputs(uncoveredTree.getAllElementsHashSet());
	}
	
	
	// beam search
	public void DFSearchGeneration(Example exmp, OldSearchState currentState, int depth, HashTable uncoveredTree, StructOutput goldOut) { // CostFunction costFunc,

		currentState.fv = null;
		currentState.predScore = 0;
		currentState.trueAccuracy = getTrainLoss(currentState.getOutput(), goldOut); // LossFunction.computeCorrectCnt(initCopy.getOutput(), goldOut);

		// store generated output
		if (!uncoveredTree.probeExistence(currentState)) {
			uncoveredTree.insert(currentState);
		}
		
		//////////////////////////////////
		
		if (depth <= 0) {
			return;
		}
		
		///////////////////////////////////
		
		// expanding
		ArrayList<Action> acts = generateActionsZeroToOne(currentState, exmp, true);
		ArrayList<OldSearchState> childrenStates = new ArrayList<OldSearchState>();
		for (Action act : acts) {
			OldSearchState newState = performActionNewState(currentState, act);
			//act.print();
			//newState.print();
			DFSearchGeneration(exmp, newState, (depth - 1), uncoveredTree, goldOut);
		}
	}

	
	
	
	
	//////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////
	
	// Searcher for testing!
	public OldSearchState DFSearchTest(Example exmp, OldSearchState initState, int maxDepth, CostFunction costFunc) {

		HashTable uncoveredTree = new HashTable(exmp.labelDim() + 2, 3);

		// for initial state /////////////
		OldSearchState initCopy = initState.getSelfCopy();
		final StructOutput goldOut = extractGoldOutput(exmp);
		initCopy.predScore = 0;
		initCopy.trueAccuracy = LossFunction.computeHammingAccuracy(initState.getOutput(), goldOut); // LossFunction.computeCorrectCnt(initCopy.getOutput(), goldOut);
		uncoveredTree.insert(initCopy);
		//////////////////////////////////
		
		///////////
		/////////// GENERATION
		///////////
		
		DFSearchGeneration(exmp, initCopy,  maxDepth, uncoveredTree, goldOut);

		
		//////
		////// SELECTION
		//////
		
		//// select best from the generated outputs
		OldSearchState bestState = selectBest(exmp, uncoveredTree.getAllElementsHashSet(), costFunc);
		return bestState;
	}
	

	
	// optimization step
	public OldSearchState selectBest(Example exmp, HashSet<OldSearchState> generatedStates,  CostFunction costf) {
		
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
				if (s1.predScore > s2.predScore) {
					return -1;
				} else if (s1.predScore < s2.predScore) {
					return 1;
				}
				return 0;
			}
			return 0;
		}
	};

}
