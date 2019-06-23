package multilabel.learning.search;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;

import ciir.umass.edu.eval.Evaluator;
import multilabel.learning.StructOutput;
import multilabel.learning.cost.CostFunction;
import multilabel.learning.cost.RankingCostFunction;
import multilabel.learning.cost.RegressionCostFunction;
import multilabel.learning.cost.TrueHammingLossFunction;
import multilabel.learning.heuristic.StateRepository;
import multilabel.pruner.LabelPruner;
import multilabel.pruner.LambdaMartLabelPruner;
import multilabel.pruner.PrunerLearning;
import multilabel.pruner.PruningEvaluator;
import multilabel.pruner.UMassRankLib;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.SparseInstance;
import multilabel.evaluation.LossFunction;
import multilabel.instance.Example;
import multilabel.instance.Featurizer;
import multilabel.instance.Label;
import multilabel.instance.OldWeightVector;

public class BreathFirstSearcher {
	
	// some setting
	boolean applyPruning = true;
	public int herrCnt = 0;
	public int crrCnt = 0;	
	
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
	
	// is the predict output can not lead to gold by just flipping 0 to 1,
	// then it is non consist to gold (return false)
	public boolean isConsistToGold(StructOutput pred, StructOutput truth) {
		int[] result = LossFunction.computeTFPN(pred, truth);
		int fpos = result[2];
		if (fpos == 0) {
			return true;
		}
		return false;
	}
	
	public boolean containPrecision1(ArrayList<OldSearchState> topkStates, Example ex, int d) {
		
		for (int j = 0; j < topkStates.size(); j++) {
			StructOutput gold = OldGreedySearcher.extractGoldOutput(ex);
			double hamming = getTrainLoss(topkStates.get(j).getOutput(), gold);
			//System.out.println(gold.toString() +"  " + topkStates.get(j).getOutput().toString() + " depth = " + d + ":  " + topkStates.get(j).trueAccuracy + " " + hamming + " " + topkStates.get(j).predScore);
		}
		
		///////////////////////////////////////////////	
		for (int j = 0; j < topkStates.size(); j++) {
			if (topkStates.get(j).trueAccuracy == 1.0) {
				return true;
			}
		}
		return false;
	}
	
	// beam search
	public void breathFirstSearchGenerate(Example exmp, OldSearchState initState, int beamSize, int maxDepth, boolean onTraj, CostFunction costFunc,
			                              HashSet<OldSearchState> hashTable, StateRepository heurStates, HashSet<OldSearchState> beamStates) {
		
		SearchBeam beam = new SearchBeam(beamSize);
		HashTable uncoveredTree = new HashTable(exmp.labelDim() + 2, 3);
		//HashSet<SearchState> beamStates = new HashSet<SearchState>(); // the good outputs that are chosen to kept in beam at each step
		Featurizer featurizer = new Featurizer();
		//SearchState[] trajectory = new SearchState[maxDepth * 2];
		
		final StructOutput goldOut = extractGoldOutput(exmp);
		int goldOneCnt = goldOut.getOneCount();
		
		// for initial state /////////////
		OldSearchState initCopy = initState.getSelfCopy();
		initCopy.predScore = 0;
		if (!onTraj) { initCopy.predScore = costFunc.getCost(initCopy, exmp); }
		initCopy.trueAccuracy = getHueristicPrecision(initState.getOutput(), goldOut);//getTrainLoss(initState.getOutput(), goldOut);
		beam.insert(initCopy);
		uncoveredTree.insert(initCopy); // hash table
		
		ArrayList<OldSearchState> zeroStepStates = (new ArrayList<OldSearchState>());
		zeroStepStates.add(initCopy);
		heurStates.storeState(0, zeroStepStates); 
		//////////////////////////////////
		
		for (int depth = 0; depth <= maxDepth; depth++) {
			
			////// pop all top k
			ArrayList<OldSearchState> topkStates = beam.popAll();
			beamStates.addAll(topkStates); // generate states that are remained in beam
			
			// check heur output!
			if (depth > 0) {
				if (depth <= goldOneCnt) {
					boolean isOk = containPrecision1(topkStates, exmp, depth);
					if (!isOk) { herrCnt++; } else { crrCnt++; }
				}
			}

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
						newState.trueAccuracy = getHueristicPrecision(newState.getOutput(), goldOut);//getTrainLoss(newState.getOutput(), goldOut);
						// compute our predict score
						newState.fv = null;//featurizer.getFeatureVector(exmp, newState.getOutput());
						newState.predScore = 0;
						if (!onTraj) { newState.predScore = costFunc.getCost(newState, exmp); } // scoring if it is off-trajectory

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
			
			
			// have a check! (is this step OK? )
			if (onTraj) {
				
			} else {
				
			}
			
			// for heuristic learning
			heurStates.storeState((depth + 1), childrenStates); 
			
			
			// check children
			beam.insertAll(childrenStates);
			beam.dropTail(onTraj);
			/////////////////////////////////////////////////////////////////
		}
		
		// return the generated outputs
		hashTable.clear();
		hashTable.addAll(uncoveredTree.getAllElementsHashSet());
	}

	
	
	
	
	//////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////
	
	
	
	// Searcher for testing! (standard testing)
	public OldSearchState beamSearchTest(Example exmp, OldSearchState initState, int beamSize, int maxDepth, CostFunction hfunc, CostFunction cfunc) {//, HashSet<SearchState> hashTable) {
		final boolean neverTraj = false; // can not do this during testing
		
		///// GENERATION
		HashSet<OldSearchState> genStates = new HashSet<OldSearchState>();
		StateRepository depthStates = new StateRepository(exmp);
		HashSet<OldSearchState> beamStates = new HashSet<OldSearchState>();
		breathFirstSearchGenerate(exmp, initState, beamSize, maxDepth, neverTraj, hfunc, genStates, depthStates, beamStates);

		////// SELECTION
		OldSearchState bestState = selectBest(exmp, beamStates, cfunc);
		return bestState;
	}
	
	
	// do generation only (no selection!)
	public HashSet<OldSearchState> beamSearchGenerationOnly(Example exmp, OldSearchState initState, int beamSize, int maxDepth, CostFunction hfunc) {
		
		final boolean neverTraj = false; // can not do this during testing
		
		/////
		///// GENERATION ONLY
		/////
		
		HashSet<OldSearchState> genStates = new HashSet<OldSearchState>();
		StateRepository depthStates = new StateRepository(exmp);
		HashSet<OldSearchState> beamStates = new HashSet<OldSearchState>();
		breathFirstSearchGenerate(exmp, initState, beamSize, maxDepth, neverTraj, hfunc, genStates, depthStates, beamStates);

		return beamStates;
	}
	

	
	// For generation loss computing
	public OldSearchState beamSearchTestOracleSelection(Example exmp, OldSearchState initState, int beamSize, int maxDepth, CostFunction hfunc) {
		
		final boolean neverTraj = false; // can not do this during testing
		
		/////
		///// GENERATION
		/////
		
		HashSet<OldSearchState> genStates = new HashSet<OldSearchState>();
		StateRepository depthStates = new StateRepository(exmp);
		HashSet<OldSearchState> beamStates = new HashSet<OldSearchState>();
		breathFirstSearchGenerate(exmp, initState, beamSize, maxDepth, neverTraj, hfunc, genStates, depthStates, beamStates);
		
		//////
		////// SELECTION
		//////
		
		//// select best from the generated outputs
		//SearchState bestState = selectBest(exmp, genStates, cfunc);
		TrueHammingLossFunction trueHammingCf = new TrueHammingLossFunction();
		//SearchState bestState = selectBest(exmp, depthStates.getAllStates(), trueHammingCf);
		OldSearchState bestState = selectBest(exmp, beamStates, trueHammingCf);
		return bestState;
	}

	
	// optimization step
	public OldSearchState selectBest(Example exmp, HashSet<OldSearchState> generatedStates,  CostFunction costf) {
		
		double bestScore = -Double.MAX_VALUE;
		OldSearchState bestState = null;
		StructOutput truth = exmp.getGroundTruthOutput();
		
		double bestAcc = -10000;
		OldSearchState bestGeneratedState = null;
		
		for (OldSearchState s : generatedStates) {
			
			s.predScore = costf.getCost(s, exmp);
			s.trueAccuracy = getTrainLoss(s.getOutput(), truth);

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

	
	///////////////////////////////////////////
	
	
/*	
	// Dagger outputs generating
	public static void runDaggerIterations(ArrayList<Example> trainExs,ArrayList<Example> testExs, int iterations, String dumpFileName) throws FileNotFoundException {
		
		int beamSize = 1;
		int maxDepth = 3;
		boolean onTraj = false;
		BreathFirstSearcher searcher = new BreathFirstSearcher();
		
		ArrayList<HashMap<Example, HashSet<SearchState>>> allStates = new ArrayList<HashMap<Example, HashSet<SearchState>>>();
		HashMap<Example, HashTable> dictionary = new HashMap<Example, HashTable>();
		ArrayList<CostFunction> costFunctions = new ArrayList<CostFunction>();
		
		// init dictionary
		for (Example ex : trainExs) {
			HashTable newTb = new HashTable(ex.labelDim() + 2, 3);
			dictionary.put(ex, newTb);
		}

		for (int iter = 0; iter <= iterations; iter++) {
			
			System.out.println("Iteration " + iter);

			HashMap<Example, HashSet<SearchState>> generateStates = new HashMap<Example, HashSet<SearchState>>();

			int exCnt = 0;
			for (Example ex : trainExs) {
				exCnt++;

				SearchState init = SearchState.getAllZeroState(ex.labelDim());
				HashSet<SearchState> uncoveredStates = new HashSet<SearchState>();
				if (iter == 0) { // first iteration
					searcher.breathFirstSearchGenerate(ex, init, beamSize, maxDepth, onTraj, costFunctions.get(iter - 1), uncoveredStates);
				} else {
					searcher.breathFirstSearchGenerate(ex, init, beamSize, maxDepth, onTraj, costFunctions.get(iter - 1), uncoveredStates);
				}
				
				System.out.println("Done Example " + exCnt);

				// store generate
				generateStates.put(ex, uncoveredStates);
				
				HashTable exTb = dictionary.get(ex);
				exTb.insertAllNoRepeat(uncoveredStates);
			}
			
			// store
			allStates.add(generateStates);
			
			RankingCostFunction iterCostFunc = trainCostFunctionRanking(allStates, dictionary, iter, dumpFileName);
			costFunctions.add(iterCostFunc);

		}
	}
*/
	
	
	public static RankingCostFunction trainCostFunctionRanking(ArrayList<HashMap<Example, HashSet<OldSearchState>>> allStates,
			                                                   HashMap<Example, HashTable> dictionary, int iter, 
			                                                   String dumpFileName) {
		/*
		
		/////// TRAIN ////////
		
		System.out.println("Start training!");
		System.out.println("Dumping Features!");
		
		// 1) dump file
		String prunerFolder = "pruner_rank";
		String trainFeatFn = prunerFolder + "/" + ds.name + "_prune_train_feat.txt";
		String testFeatFn = prunerFolder + "/" + ds.name + "_prune_test_feat.txt";
		PrunerLearning.dumpRanklibFeatureFile(ds.getTrainExamples(),  trainFeatFn);
		PrunerLearning.dumpRanklibFeatureFile(ds.getTestExamples(), testFeatFn);
		
		// 2) prepare runing cmd

		String modelFn = ds.name + "_rank_lambdamart_p" + String.valueOf(topK) +".txt";
		String modelPath = prunerFolder + "/" + modelFn;
		String basicCmd = "-sparse -tree 1000 -leaf 20 -shrinkage 0.1 -tc -1 -mls 1 -estop 50 -ranker 6";
		String metricCmd = " -metric2t " + "P@" + String.valueOf(topK);
		String fileCmd = " -train " + trainFeatFn + " -validate " + testFeatFn + " -save " + modelPath;
		String trainCmd = basicCmd + metricCmd + fileCmd;
		 
		// 3) run training
		System.out.println("Call command:\n" + trainCmd);
		String[] cmds = trainCmd.split("\\s+");
		Evaluator.main(cmds);
		
		
		////// TEST //////////
		System.out.println("Start testing!");
		LabelPruner pruner = new LambdaMartLabelPruner(modelPath, topK);
		PruningEvaluator.evaluatePruner(ds.getTestExamples(), pruner);
		*/
		
		return null;
		
		
	}
	
	
	
	//////////////////////////////
	
	// in the non-redundant space, we require the states to have a high precision,
	// when there are more 1 in the predict than in the gold output, then we replace
	// this loss as Hamming loss
	public double getHueristicPrecision(StructOutput pred, StructOutput truth) {
		int[] crrWng = LossFunction.computeTFPN(pred, truth);
		double tp = crrWng[0];
		double tn = crrWng[1];
		double fp = crrWng[2];
		double fn = crrWng[3];
		
		double allOneInPred = (tp + fp);
		double allOneInGold = (tp + fn);
		double pre = tp / (allOneInPred);
		if ((tp == 0) && (allOneInPred == 0)) {
			pre = 1;
		}
		
		//////////////////////////////////////////////////
		if (allOneInPred > allOneInGold) {
			// no way to get perfect output in 0->1 search space, use hamming accuracy instead
			double hamming = LossFunction.computeHammingAccuracy(pred, truth);
			return hamming;
		}
		//////////////////////////////////////////////////

		return pre;
	}
	
	// Dagger heuristic training
	public static ArrayList<CostFunction> heuristicDaggerIterations(ArrayList<Example> trainExs, ArrayList<Example> testExs, int beamSize, int maxDepth, String dsName, int iterations) throws FileNotFoundException {
		
		BreathFirstSearcher searcher = new BreathFirstSearcher();
		
		// for testing only
		HashMap<Example, StateRepository> testStates = dumpTestingHuerStates(testExs, beamSize, maxDepth);
		
		
		ArrayList<HashMap<Example, StateRepository>> allStates = new ArrayList<HashMap<Example, StateRepository>>();
		ArrayList<CostFunction> heurFunctions = new ArrayList<CostFunction>();
		
		for (int iter = 0; iter <= iterations; iter++) {
			
			System.out.println("Iteration " + iter);

			HashMap<Example, StateRepository> generateStates = new HashMap<Example,StateRepository>();

			int exCnt = 0;
			for (Example ex : trainExs) {
				exCnt++;

				OldSearchState init = OldSearchState.getAllZeroState(ex.labelDim());
				HashSet<OldSearchState> uncoveredStates = new HashSet<OldSearchState>();
				StateRepository stateEachDepth = new StateRepository(ex);
				HashSet<OldSearchState> beamStates = new HashSet<OldSearchState>();
				if (iter == 0) { // first iteration
					searcher.breathFirstSearchGenerate(ex, init, beamSize, maxDepth, true, null, uncoveredStates, stateEachDepth, beamStates);
				} else {
					searcher.breathFirstSearchGenerate(ex, init, beamSize, maxDepth, false, heurFunctions.get(iter - 1), uncoveredStates, stateEachDepth, beamStates);
				}
				
				System.out.println("Done Example " + exCnt);

				// store generate
				generateStates.put(ex, stateEachDepth);
			}
			
			// store
			allStates.add(generateStates);
			
			RankingCostFunction iterCostFunc = trainHeurFunctionRanking(allStates, testStates, iter, beamSize, maxDepth, dsName);
			heurFunctions.add(iterCostFunc);
		}
		
		// dagger heuristic function list
		return heurFunctions;
	}
	
	
	public static HashMap<Example, StateRepository> dumpTestingHuerStates(ArrayList<Example> testExs, int beamSize, int maxDepth) {

		BreathFirstSearcher searcher = new BreathFirstSearcher();

		HashMap<Example, StateRepository> generateStates = new HashMap<Example,StateRepository>();
		for (Example ex : testExs) {
			OldSearchState init = OldSearchState.getAllZeroState(ex.labelDim());
			HashSet<OldSearchState> uncoveredStates = new HashSet<OldSearchState>();
			StateRepository stateEachDepth = new StateRepository(ex);
			HashSet<OldSearchState> beamStates = new HashSet<OldSearchState>();
			searcher.breathFirstSearchGenerate(ex, init, beamSize, maxDepth, true, null, uncoveredStates, stateEachDepth, beamStates);

			// store generate
			generateStates.put(ex, stateEachDepth);
		}
		
		return generateStates;
	}
	
	
	public static RankingCostFunction trainHeurFunctionRanking(ArrayList<HashMap<Example, StateRepository>> allStates,
			HashMap<Example, StateRepository> testStates,
			int iter, int beamSize, int maxDepth, String dsName) throws FileNotFoundException {

		/////// TRAIN ////////

		System.out.println("Dumping Features!");

		// 1) dump file
		String prunerFolder = "heur_rank";
		String trainFeatFn = prunerFolder + "/" + dsName + "_heur_train_feat_iter"+String.valueOf(iter)+".txt";
		String testFeatFn = prunerFolder + "/" + dsName + "_heur_test_feat_iter"+String.valueOf(iter)+".txt";
		PrintWriter trainWriter = new PrintWriter(trainFeatFn);
		PrintWriter testWriter = new PrintWriter(testFeatFn);
		// dump train
		int trainQid = 0;
		for (int it = 0; it < allStates.size(); it++) {
			HashMap<Example, StateRepository> iterStates = allStates.get(it);
			for (StateRepository sr : iterStates.values()) {
				int labelCnt = sr.getExample().getOneLabelCount();
				if (labelCnt > maxDepth) labelCnt = maxDepth;
				for (int d = 1; d <= labelCnt; d++) {
					trainQid++;
					sr.dumpRankingLists(sr.getStateAtDepth(d), trainQid, trainWriter, d);
				}
			}
		}
		// dump validation
		int devQid = 0;
		for (StateRepository sr : testStates.values()) {
			int labelCnt = sr.getExample().getOneLabelCount();
			if (labelCnt > maxDepth) labelCnt = maxDepth;
			for (int d = 1; d <= labelCnt; d++) {
				devQid++;
				sr.dumpRankingLists(sr.getStateAtDepth(d), devQid, testWriter, d);
			}
		}
		trainWriter.close();
		testWriter.close();

		
		// 2) prepare running cmd
		String modelFn = dsName + "_huer_rank_lambdamart_iter"+String.valueOf(iter)+"_b" + String.valueOf(beamSize) +".txt";
		String modelPath = prunerFolder + "/" + modelFn;
		String basicCmd = "-sparse -tree 1000 -leaf 20 -shrinkage 0.1 -tc -1 -mls 1 -estop 50 -ranker 6";
		String metricCmd = " -metric2t " + "P@" + String.valueOf(beamSize);
		String fileCmd = " -train " + trainFeatFn + " -validate " + testFeatFn + " -save " + modelPath;
		String trainCmd = basicCmd + metricCmd + fileCmd;

		// 3) run training
		System.out.println("Call command:\n" + trainCmd);
		String[] cmds = trainCmd.split("\\s+");
		Evaluator.main(cmds);


		////// TEST //////////
		//System.out.println("Start testing!");

		
		//// construct return cost function
		
		RankingCostFunction hf = new RankingCostFunction(new Featurizer());
		hf.loadModel(modelPath);
		
		return hf;
	}
}
