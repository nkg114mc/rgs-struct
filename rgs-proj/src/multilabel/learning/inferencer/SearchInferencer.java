package multilabel.learning.inferencer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

import multilabel.data.Dataset;
import multilabel.evaluation.LossFunction;
import multilabel.evaluation.MultiLabelEvaluator;
import multilabel.instance.Example;
import multilabel.instance.Featurizer;
import multilabel.instance.Label;
import multilabel.learning.StructOutput;
import multilabel.learning.cost.CostFunction;
import multilabel.learning.search.Action;
import multilabel.learning.search.HashTable;
import multilabel.learning.search.SearchBeam;
import multilabel.learning.search.OldSearchState;
import multilabel.pruner.PrunerLearning;

public class SearchInferencer {
	
	int beamSize = 0;
	int maxDepth = 0;
	LinearCostFunction lcf;
	
	public SearchInferencer(int bsize, int maxd) {
		beamSize = bsize;
		maxDepth = maxd;
		lcf = new LinearCostFunction();
	}

	public StructOutput inference(Example example, edu.illinois.cs.cogcomp.sl.util.WeightVector wvUiuc, StructOutput truthOutput, boolean doLossAugment) {
		lcf.loadNewWeightUiuc(example, wvUiuc);
		StructOutput result = null;
		result = inference(example, lcf, truthOutput, doLossAugment);
		return result;
	}
	/*
	public double getPositionLoss(int trueVal, int predVal) {
		if (trueVal != predVal) {
			return 1.0;
		}
		return 0;
	}*/

	public StructOutput inference(Example example, LinearCostFunction cfunc, StructOutput groundTruth, boolean doLossAugment)  {

		/////
		///// GENERATION
		/////

		OldSearchState init = OldSearchState.getAllZeroState(example.labelDim());
		HashSet<OldSearchState> genStates = new HashSet<OldSearchState>();
		breathFirstSearchGenerate(example, init, beamSize, maxDepth, doLossAugment, cfunc, genStates);

		//////
		////// SELECTION
		//////

		//// select best from the generated outputs
		OldSearchState bestState = selectBest(example, genStates, cfunc);
		return bestState.getOutput();

	}

	
	// some setting
	boolean applyPruning = true;
	
	
	// only flip 0 to 1
	public ArrayList<Action> generateActionsZeroToOne(OldSearchState state, Example ex,  boolean doPruning) {
		ArrayList<Action> actions = new ArrayList<Action>();
		StructOutput currentOut = state.getOutput();
		for (int i = 0; i < currentOut.size(); i++) {
			Label lb = ex.getLabelGivenIndex(i);
			//if (!lb.isPruned) {
				if (currentOut.getValue(i) == 0) { // a 0 bit
					Action act = new Action(i, 1);
					act.oldValue = currentOut.getValue(i);
					actions.add(act);
					//act.print();
				}
			//}
		}
		return actions;
	}
	
	public ArrayList<Action> generateActions(OldSearchState state, Example ex) {
		//return generateActionsWithSetting(state, true);
		return generateActionsZeroToOne(state, ex,  false);
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
	
	// number of wrong labels
	public double getErrorCount(StructOutput pred, StructOutput truth, boolean doLossAugment) {
		
		if (doLossAugment == false) {
			return 0; // no loss value
		}
		
		int[] crrWng = LossFunction.computeRightWrong(pred, truth);
		int wrong = crrWng[1];
		return wrong;
	}
	
	// beam search
	public void breathFirstSearchGenerate(Example exmp, OldSearchState initState, int beamSize, int maxDepth, boolean doLossAugment,
			                              CostFunction costFunc,
			                              HashSet<OldSearchState> hashTable) {
		
		boolean onTraj = false;
		
		SearchBeam beam = new SearchBeam(beamSize);
		HashTable uncoveredTree = new HashTable(exmp.labelDim() + 2, 3);
		Featurizer featurizer = new Featurizer();
		//SearchState[] trajectory = new SearchState[maxDepth * 2];
		
		final StructOutput goldOut = extractGoldOutput(exmp);

		// for initial state /////////////
		OldSearchState initCopy = initState.getSelfCopy();
		initCopy.predScore  = costFunc.getCost(initCopy, exmp) + getErrorCount(initCopy.getOutput(), goldOut, doLossAugment);
		initCopy.trueAccuracy = getTrainLoss(initState.getOutput(), goldOut);
		beam.insert(initCopy);
		uncoveredTree.insert(initCopy);
		//////////////////////////////////
		
		for (int depth = 0; depth <= maxDepth; depth++) {
			
			////// pop all top k
			//currentState = beam.popBest(onTraj);
			ArrayList<OldSearchState> topkStates = beam.popAll();
			//trajectory[depth] = currentState;
			
			//System.out.println("depth " + depth + " acc = " + currentState.trueAccuracy + " pred = " + currentState.predScore);
			//System.out.println(goldOut.toString());
			
			if (depth == maxDepth) {
				System.out.println("Stop search: reaching max depth!");
				break;
			}
			
			///// Start expanding
			/////////////////////////////////////////////////////////////////
			
			
			ArrayList<OldSearchState> childrenStates = new ArrayList<OldSearchState>();
			
			for (int ii = 0; ii < topkStates.size(); ii++) {
				OldSearchState iState = topkStates.get(ii);

				ArrayList<Action> acts = generateActions(iState, exmp); // only flip 0 to 1
				for (Action act : acts) {
					OldSearchState newState = performActionNewState(iState, act);
					//act.print();
					//newState.print();
					if (!uncoveredTree.probeExistence(newState)) {

						// compute true loss
						newState.trueAccuracy = getTrainLoss(newState.getOutput(), goldOut);
						newState.predScore  = costFunc.getCost(newState, exmp) + getErrorCount(initCopy.getOutput(), goldOut, doLossAugment);

						//System.out.println("acc =  " + newState.trueAccuracy + " pred = " + newState.predScore);

						childrenStates.add(newState);
						uncoveredTree.insert(newState);
					}
				}

			} // end for ii
			
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
	}
	
	// optimization step
	public OldSearchState selectBest(Example exmp, HashSet<OldSearchState> generatedStates, CostFunction costf) {
		
		double bestScore = -Double.MAX_VALUE;
		OldSearchState bestState = null;
		
		double bestAcc = -10000;
		OldSearchState bestGeneratedState = null;
		
		for (OldSearchState s : generatedStates) {
			
			//s.predScore = costf.getCost(s, exmp);

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
			if (s1.trueAccuracy > s2.trueAccuracy) {
				return -1;
			} else if (s1.trueAccuracy < s2.trueAccuracy) {
				return 1;
			} else if (s1.trueAccuracy == s2.trueAccuracy) {
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
	
	////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////
	
	public static void main(String[] args) {
		/*
		String[] dsNames = { "scene", "emotions", "yeast", "enron", "medical", "LLOG", "SLASHDOT", "tmc2007-500", "genbasebin"};
		for (int i = 0; i < dsNames.length; i++) {
			readArffFileTest(dsNames[i]);
		}*/
		/*
		String[] dsNames = { "CAL500", "bibtex",  "Corel5k", "delicious", "mediamill", "bookmarks"};
		for (int i = 0; i < dsNames.length; i++) {
			readCsvFileTest(dsNames[i]);
		}*/
		
		String name = parseArgs(args);
		System.out.println("Name: " + name);
		
		// read dataset
		Dataset ds = PrunerLearning.readArffFileTest(name);
		
		// use a random model
		edu.illinois.cs.cogcomp.sl.util.WeightVector wrnd = getRandomWeight(ds);
		
		SearchInferencer inferencer = new SearchInferencer(3, 3);
		ArrayList<Example> testExs = ds.getTestExamples();
		for (int i = 0; i < testExs.size(); i++) {
			Example ex = testExs.get(i);
			fillAllFeatOne(ex);
			ex.predictOutput = inferencer.inference(ex, wrnd, null, false);
		}
		
		// scoring!
		MultiLabelEvaluator evaluator = new MultiLabelEvaluator();
		evaluator.evaluationDataSet(ds.name, ds.getTestExamples());
		// done.
	}
	
	private static void fillAllFeatOne(Example ex) {
		ArrayList<Double> feats = ex.getFeat();
		for (int i = 0; i < feats.size(); i++) {
			feats.set(i, 1.0);
		}
	}
	
	private static edu.illinois.cs.cogcomp.sl.util.WeightVector getRandomWeight(Dataset ex) {
		Random rnd = new Random();
		int l = Featurizer.getFeatureDimension(ex.getFeatureDimension(), ex.getLabelDimension()) + 1;
		float[] all1 = new float[l];
		Arrays.fill(all1, 1);
		for (int i = 1; i <= ex.getFeatureDimension(); i++) {
			 all1[i] = -1f;
		}
		int numUnary = ex.getFeatureDimension() * ex.getLabelDimension();
		for (int i = (numUnary + 1); i < l; i++) {
			 all1[i] = 0f;
		}
		/*
		for (int i = 0; i < l; i++) {
			float rndf = 1f;//(rnd.nextFloat() % 1.0f) - 0.5f;
			 all1[i] = rndf;
		}*/
		return (new edu.illinois.cs.cogcomp.sl.util.WeightVector(all1));
	}
	
	public static String parseArgs(String[] args) {
		String name = "?";
		for (int i = 0; i < args.length; i++) {
			if (args[i].equals("-name")) {
				name = (args[i + 1]);
			}
		}
		return name;
	}

}
