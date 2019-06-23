package sequence.hw;

import edu.illinois.cs.cogcomp.sl.core.AbstractInferenceSolver;
import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearning.EInferencer;
import general.AbstractInstance;
import general.AbstractOutput;
import search.GreedySearcher;
import search.SearchResult;

public class HwSearchInferencer extends AbstractInferenceSolver {
	private static final long serialVersionUID = 4874125946571662037L;
	
	GreedySearcher gsearcher;
	
	
	public HwSearchInferencer(GreedySearcher gschr) {
		gsearcher = gschr;
	}
	
	//public HwSearchInferencer(GreedySearcher gschr, EInferencer eInfr) {
	//	gsearcher = gschr;
	//	evalInferencer = eInfr;
	//}
	
	@Override
	public IStructure getBestStructure(WeightVector wv, IInstance input) throws Exception {
		SearchResult result = gsearcher.runSearchWithRestarts(wv, gsearcher.getEvalInferencer(), (AbstractInstance)input, null, false);
		AbstractOutput resultOutput = result.predState.structOutput;
		return ((IStructure)(resultOutput));
	}
	@Override
	public float getLoss(IInstance ins, IStructure goldStructure,  IStructure structure) {
		return gsearcher.getLossFloat((AbstractInstance)ins, goldStructure, structure);
	}

	@Override
	public IStructure getLossAugmentedBestStructure(WeightVector wv, IInstance input, IStructure gold) throws Exception {
		SearchResult result = gsearcher.runSearchWithRestarts(wv, gsearcher.getEvalInferencer(), (AbstractInstance)input, (AbstractOutput)gold, true);
		AbstractOutput resultOutput = result.predState.structOutput;
		return ((IStructure)(resultOutput));
	}
	
	// result more-informative result rather than just output
	public SearchResult runSearchInference(WeightVector wv, EInferencer eifr, IInstance input, IStructure gold) {
		SearchResult result = gsearcher.runSearchWithRestarts(wv, eifr, (AbstractInstance)input, (AbstractOutput)gold, false);
		return (result);
	}
	
	public SearchResult runSearchInferenceMaybeLossAug(WeightVector wv, EInferencer eifr, IInstance input, IStructure gold, boolean losAug) {
		SearchResult result = gsearcher.runSearchWithRestarts(wv, eifr, (AbstractInstance)input, (AbstractOutput)gold, losAug);
		return (result);
	}
	
	
	@Override
	public AbstractInferenceSolver clone() {
		 HwSearchInferencer cp = new  HwSearchInferencer(gsearcher);
		 return cp;
	}
	
	public GreedySearcher getSearcher() {
		return gsearcher;
	}
	
}

/*
public class HwSearchInferencer extends AbstractInferenceSolver {
	

	private static final long serialVersionUID = 1L;
	public Random random = new Random();
	
	private HwFeaturizer featurizer;
	private ZobristKeys zobKeys;
	
	// parameters
	public int randInitSize = 1;
	//public final int MAX_DEPTH = 5000;
	public int maxBranch = -1;
	
	public HwSearchInferencer(HwFeaturizer fzr, int restart, ZobristKeys abkeys) {
		featurizer = fzr;
		randInitSize = restart;
		zobKeys = abkeys;
	}

	@Override
	public Object clone(){
		return new HwSearchInferencer(featurizer, randInitSize, zobKeys);
	}

	@Override
	public IStructure getLossAugmentedBestStructure(WeightVector wv, IInstance input, IStructure gold) {
		
		AbstractInstance x = (AbstractInstance) input;
		AbstractOutput ystar = (AbstractOutput) gold;
		
		boolean doLossAug = (gold != null);
		
		// gold state
		SearchState goldState = new SearchState(ystar);
		
		// init states
		HashSet<SearchState> initStates = generateRandomInitState(random, x, randInitSize);
		
		HashSet<SearchState> generatedStates = new HashSet<SearchState>();
		for (SearchState initState : initStates) {
			SearchState besti = doHillClimbing(wv, x, initState, goldState, Integer.MAX_VALUE, doLossAug);
			generatedStates.add(besti);
		}
		
		// pick the best
		SearchState pred = null;
		float bestScore = Float.NEGATIVE_INFINITY;
		for (SearchState bstate: generatedStates) {
			if (bstate.score > bestScore) {
				bestScore = bstate.score;
				pred = bstate;
			}
		}
		
		//System.out.println("randsearch");
		return (IStructure) (pred.structOutput);
	}
	
	// return best state with hill-climbing
	public SearchState doHillClimbing(WeightVector wv, AbstractInstance ins, SearchState initState, SearchState goldState, int maxDepth, boolean isLossAug) {
		
		// if perform loss augmented inference, then 
		if (isLossAug) {
			if (goldState == null) {
				throw new RuntimeException("Error: perform loss-aug inference but no gold-state!");
			}
		}
		
		SearchHashTable hashTb = new SearchHashTable(zobKeys);
		
		HwFactorGraph factorGraph = new HwFactorGraph((HwInstance)ins, featurizer);
		factorGraph.updateScoreTable(wv.getDoubleArray());
		
		// about initial state
		SearchState currState = initState;
		float currScore = (float)scoring(wv, (IInstance)ins, (IStructure)currState.structOutput);
		if (isLossAug) {
			float loss = getLoss((IInstance)ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput));
			currScore += loss;
		}
		float lastScore = currScore;
		hashTb.insertNewInstance(currState); // insert initial state to hashTb
		
		int depth = 0;
		for (depth = 1; depth < maxDepth; depth++) {
			
			if (depth % 10 == 0) {
				//System.out.println("depth = " + depth + ", " + currScore);
			}

			List<SearchAction> actions = SeachActionGenerator.genAllAction(currState);
			// stop condition 1: No legal actions
			if (actions.size() == 0) {
				break;
			}
			
			// try best
			//SearchAction bestAct1 = pickBestPredAction(actions, wv, ins, currState, goldState);
			//SearchAction bestAct2 = pickBestPredActionFast(factorGraph, actions, wv, ins, currState, goldState, isLossAug);
			SearchAction bestAct3 = pickBestPredActionIncreamentally(factorGraph, actions, wv, ins, hashTb, currState, goldState, isLossAug);
			SearchAction bestAct = bestAct3;
			float bestSc = (float)bestAct.score;
			
			// stop condition 2: Reach hill peak
			currScore = bestSc;
			currState.score = currScore;
			if (currScore <= lastScore) { // reach peak
				break;
			} else {
				currState.doAction(bestAct);
			}
			
			///System.out.println("score = " + currScore);
			// prepare for next depth
			lastScore = currScore;
			hashTb.insertNewInstance(currState); // avoid repeat
		}
		
		//System.out.println("Hash Size = " + hashTb.getSize());
		
		return currState;
	}
	
	public SearchAction pickBestPredAction(List<SearchAction> actions, WeightVector wv, AbstractInstance ins, SearchState currState, SearchState goldState, boolean isLossAug) {
		float bestSc = Float.NEGATIVE_INFINITY;
		SearchAction bestAct = null;
		for (SearchAction act : actions) {
			currState.doAction(act);
			
			act.score = scoring(wv, (IInstance)ins, (IStructure)currState.structOutput);
			if (isLossAug) {
				act.score += getLoss((IInstance)ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput));
			}
			float sc = (float)(act.score);
			if (sc > bestSc) {
				bestSc = sc;
				bestAct = act;
			}
			
			currState.undoAction(act);
		}
		return bestAct;
	}


	
	// a faster version
	public SearchAction pickBestPredActionFast(HwFactorGraph factorGraph, List<SearchAction> actions, WeightVector wv, AbstractInstance ins, SearchState currState, SearchState goldState, boolean isLossAug) {
		
		float bestSc = Float.NEGATIVE_INFINITY;
		SearchAction bestAct = null;
		
		
		if (actions.size() > maxBranch) {
			maxBranch = actions.size();
			System.out.println("MaxBranch = " + maxBranch);
		}
		
		for (SearchAction act : actions) {
			currState.doAction(act);
			
			double loss = 0;
			if (isLossAug) {
				loss = getLoss((IInstance)ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput));
			}
			//double sc1 = scoring(wv, (IInstance)ins, (IStructure)currState.structOutput) + loss;
			double sc2 = factorGraph.computeScoreWithTable(wv.getDoubleArray(), (HwOutput)currState.structOutput) + loss;
			//equalCheck(sc1, sc2);
			
			act.score = sc2;
			float sc = (float)(act.score);
			if (sc > bestSc) {
				bestSc = sc;
				bestAct = act;
			}
			currState.undoAction(act);
		}
		
		return bestAct;
	}
	
	// increamentally compute prediction score
	public SearchAction pickBestPredActionIncreamentally(HwFactorGraph factorGraph, List<SearchAction> actions, WeightVector wv, AbstractInstance ins, SearchHashTable hashTb, SearchState currState, SearchState goldState, boolean isLossAug) {
		
		float bestSc = Float.NEGATIVE_INFINITY;
		SearchAction bestAct = null;
		
		// the old score
		double oldLoss = 0;
		if (isLossAug) {
			if (goldState != null) {
				oldLoss = getLoss((IInstance)ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput));
			}
		}
		double oldPredScore = factorGraph.computeScoreWithTable(wv.getDoubleArray(), (HwOutput)currState.structOutput);
	   	int  oldStateHash = hashTb.computeIndex(currState);
		
		for (SearchAction act : actions) {
			currState.doAction(act);
			
			// is repeated action
			boolean stateIsNotRepeat = (!hashTb.probeExistenceWithAction(currState, oldStateHash, act));
			if (stateIsNotRepeat) {
				double loss = 0;
				if (isLossAug) {
					//double loss1 = getLoss((IInstance)ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput));
					double loss2 = oldLoss + computeZeroOneLossIncreamentally(act, (IInstance)ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput));
					//equalCheck(loss1, loss2);
					loss = loss2;
				}
				//double sc1 = scoring(wv, (IInstance)ins, (IStructure)currState.structOutput) + loss;
				//double sc2 = factorGraph.computeScoreWithTable(wv.getDoubleArray(), (HwOutput)currState.structOutput) + loss;
				double sc3 = oldPredScore + factorGraph.computeScoreDiffWithTable(wv.getDoubleArray(), act, (HwOutput)currState.structOutput) + loss;
				//equalCheckThree(sc1, sc2, sc3);
				//equalCheck(sc2, sc3);

				act.score = sc3;
				float sc = (float)(act.score);
				if (sc > bestSc) {
					bestSc = sc;
					bestAct = act;
				}
			}
			
			currState.undoAction(act);
		}
		
		return bestAct;
	}


	public static double computeZeroOneAccIncreamentally(SearchAction action, IInstance ins, IStructure goldStructure,  IStructure structure) {
		AbstractOutput goldLabeledSeq = (AbstractOutput) goldStructure;
		
		double accuracyChange = 0;
		int goldValue = goldLabeledSeq.getOutput(action.getSlotIdx());
		if (action.getOldVal() == goldValue) { // old is correct
			accuracyChange -= 1.0;
		}
		if (action.getNewVal() == goldValue) {
			accuracyChange += 1.0;
		}
		
		return accuracyChange;
	}
	//// ====LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS ====================================================
	public static double computeZeroOneLossIncreamentally(SearchAction action, IInstance ins, IStructure goldStructure,  IStructure structure) {
		AbstractOutput goldLabeledSeq = (AbstractOutput) goldStructure;
		double lossChange = 0;
		int goldValue = goldLabeledSeq.getOutput(action.getSlotIdx());
		if (action.getOldVal() == goldValue) { // old is correct
			lossChange += 1.0;
		}
		if (action.getNewVal() == goldValue) {
			lossChange -= 1.0;
		}
		return lossChange;
	}
	//// ====LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS ====================================================
	
	public static float computeZeroOneAcc(IInstance ins, IStructure goldStructure,  IStructure structure) {
		AbstractOutput goldLabeledSeq = (AbstractOutput) goldStructure;
		float loss = 0;
		for (int i = 0; i < goldLabeledSeq.size(); i++) {
			if (((AbstractOutput) structure).getOutput(i) != goldLabeledSeq.getOutput(i)) {
				loss += 1.0f;
			}
		}
		return loss;
	}
	
	public static void equalCheck(double d1, double d2) {
		if (Math.abs(d1 - d2) < 0.0000001) {
			// ok
		} else {
			//System.err.println("Inequal value " + d1 + " != " + d2);
			throw new RuntimeException("Inequal value " + d1 + " != " + d2);
		}
	}
	public static void equalCheckThree(double d1, double d2, double d3) {
		if ((Math.abs(d1 - d2) < 0.0000001) && (Math.abs(d1 - d3) < 0.0000001)) {
			// ok
		} else {
			throw new RuntimeException("Inequal value " + d1 + " != " + d2  + " != " + d3);
		}
	}
	
	public static double myDotProduct(HashMap<Integer, Double> fv, WeightVector wv) {
		double re = 0;
		for (int idx : fv.keySet()) {
			re += (wv.get(idx) * fv.get(idx));
		}
		return re;
	}

	public double scoring(WeightVector wv, IInstance ins, IStructure pred) {
		HashMap<Integer, Double> fv = featurizer.featurize((HwInstance)ins, (HwOutput)pred);
		return myDotProduct(fv, wv);
	}
	
	public HashSet<SearchState> generateRandomInitState(Random random, AbstractInstance inst, int stateNum) {
		
		HwInstance x = (HwInstance) inst;
		HashSet<SearchState> rndSet = new HashSet<SearchState>();
		
		// repeat
		int cnt = 0;
		while (true) {
			
			// get a uniform state
			HwOutput output = new HwOutput(inst.size(), x.alphabet);
			for (int j = 0; j < output.size(); j++) {
				output.output[j] = random.nextInt(inst.domainSize()); // purely uniform
			}
			
			SearchState s = new SearchState(output);
			if (!rndSet.contains(s)) {
				rndSet.add(s);
				cnt++;
			}
			
			if (cnt >= stateNum) {
				break;
			}
		}
		
		//System.out.println("done init...");
		return rndSet;
	}
	
	
	public HashSet<SearchState> generateAllZeroInitState(Random random, AbstractInstance inst) {
		HwInstance x = (HwInstance) inst;
		HashSet<SearchState> zSet = new HashSet<SearchState>();
		// get a uniform state
		HwOutput output = new HwOutput(inst.size(), x.alphabet);
		for (int j = 0; j < output.size(); j++) {
			output.output[j] = 0;
		}
		zSet.add(new SearchState(output));
		return zSet;
	}

	@Override
	public IStructure getBestStructure(WeightVector wv, IInstance input) throws Exception {
		return getLossAugmentedBestStructure(wv, input, null);
	}
	@Override
	public float getLoss(IInstance ins, IStructure goldStructure,  IStructure structure) {
		return computeZeroOneAcc(ins, goldStructure, structure);
	}

}
*/

