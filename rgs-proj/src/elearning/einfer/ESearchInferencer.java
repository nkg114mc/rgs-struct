package elearning.einfer;

import java.util.HashMap;
import java.util.List;

import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import elearning.EInferencer;
import general.AbstractActionGenerator;
import general.AbstractFeaturizer;
import general.AbstractInstance;
import general.AbstractLossFunction;
import general.AbstractOutput;
import search.SearchAction;
import search.SearchHashTable;
import search.SearchResult;
import search.SearchState;
import search.SearchTrajectory;
import search.ZobristKeys;
import search.loss.LossScore;


public class ESearchInferencer extends EInferencer {
	
	SearchStateScoringFunction efuncScorer;
	public AbstractFeaturizer efeaturizer;
	
	///////////////////
	
	private ZobristKeys zobKeys;
	private AbstractLossFunction lossFunc;
	private AbstractActionGenerator actionGener;
	
	
	public ESearchInferencer(SearchStateScoringFunction efunc, 
			                 AbstractFeaturizer efzr, 
			                 ZobristKeys zks, 
			                 AbstractLossFunction lossf, 
			                 AbstractActionGenerator agen) {
		efuncScorer = efunc;
		efeaturizer = efzr;
		
		zobKeys = zks;
		lossFunc = lossf;
		actionGener = agen;
	}

	////////////////////////////////////////////////////
	////////////////////////////////////////////////////
	////////////////////////////////////////////////////
	
	@Override
	public List<SearchState> generateMultiInitStates(AbstractInstance inst, AbstractOutput gold, List<SearchState> originalInitStates) {
		return originalInitStates;
	}

	@Override
	public void setEvalScoringFunc(SearchStateScoringFunction efunc) {
		efuncScorer = efunc;
	}

	@Override
	public SearchState generateOneInitState(AbstractInstance x, AbstractOutput gold, SearchState originalInit) {
		
		// gold state
		SearchState goldState = new SearchState(gold);
		
		SearchResult e_result = doEvalHillClimbing(efuncScorer, x, originalInit, goldState, Integer.MAX_VALUE, false);
		SearchState y_end_state = e_result.predState; // this is just y_end
		return y_end_state;
	}
	
	
	public SearchResult generateOneInitStateWithTraj(AbstractInstance x, AbstractOutput gold, SearchState originalInit) {
		
		// gold state
		SearchState goldState = new SearchState(gold);
		
		SearchResult e_result = doEvalHillClimbing(efuncScorer, x, originalInit, goldState, Integer.MAX_VALUE, false);
		//SearchState y_end_state = e_result.predState; // this is just y_end
		return e_result;
	}

	////////////////////////////////////////////////////
	////////////////////////////////////////////////////
	////////////////////////////////////////////////////
	
	// hillClimbing with evaluation

	// return best state with hill-climbing
	public SearchResult	 doEvalHillClimbing(SearchStateScoringFunction efunc,
			                                AbstractInstance ins, SearchState initState, SearchState goldState, int maxDepth, boolean isLossAug) {
	
		// if perform loss augmented inference, then 
		if (isLossAug) {
			if (goldState == null) {
				throw new RuntimeException("Error: perform loss-aug inference but no gold-state!");
			}
		}
		
		SearchHashTable hashTb = new SearchHashTable(zobKeys);

		
		SearchTrajectory trajectory = new SearchTrajectory();
		
		// about initial state
		SearchState currState = initState;
		double currScore = getEScoring(ins, currState.structOutput);
		if (isLossAug) {
			double loss = lossFunc.computeZeroOneLosss((IInstance)ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput)).getVal();
			currScore += loss;
		}
		currState.score = (float)currScore;
		if (goldState.structOutput != null) {
			currState.trueAccFrac = lossFunc.computeZeroOneAcc(ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput));
			currState.trueAcc = currState.trueAccFrac.getVal();
		}
		hashTb.insertNewInstance(currState); // insert initial state to hashTb

		int depth = 0;
		double lastScore = currScore;
		double highestScore = currScore;
		trajectory.concatenateState(currState);
		
		for (depth = 1; depth < maxDepth; depth++) {
			
			List<SearchAction> actions = actionGener.genAllAction(ins, currState);//SeachActionGenerator.genAllAction(currState);
			// stop condition 1: No legal actions
			if (actions.size() == 0) {
				break;
			}
			
			if (depth % 1 == 0) {
				//System.out.println("depth = " + depth + ", mentions = " + initState.structOutput.size() + ", branch = " + actions.size());
				//System.out.println("depth = " + depth + ", hashSize = " + hashTb.getSize());
			}
			
			// try best
			SearchAction bestAct1 = pickBestPredAction(actions, ins, currState, goldState, isLossAug);
			//SearchAction bestAct2 = pickBestPredActionFast(factorGraph, actions, wv, ins, currState, goldState, isLossAug);
			//SearchAction bestAct3 = pickBestPredActionIncreamentally(factorGraph, actions, wv, ins, hashTb, currState, goldState, isLossAug, bTrueAcc);//, bTrueAccFrac);
			SearchAction bestAct = bestAct1;
			if (bestAct == null) { // no Non-repeat actions anymore ...
				break;
			}
			double bestSc = bestAct.score;
			
	
			if (highestScore < bestSc) {
				highestScore = bestSc;
			}
			
			// stop condition 2: Reach hill peak
			currScore = bestSc;
			
			if (currScore <= lastScore) { // reach peak
				break;
			} else {
				currState.doAction(bestAct);
				currState.score = (float)currScore;
				currState.trueAcc = bestAct.acc;
				currState.trueAccFrac = bestAct.accFrac;
			}
			
			lastScore = currScore;
			hashTb.insertNewInstance(currState); // avoid repeat
			
			trajectory.concatenateState(currState); // record the trajectory state
		}
		
		SearchResult ret = new SearchResult();
		ret.predState = currState;
		ret.predScore = currState.score;
		ret.accuarcy = -Double.NEGATIVE_INFINITY;//bestAcc;
		ret.addTraj(trajectory);
		
		return ret;
	}
	
	public SearchAction pickBestPredAction(List<SearchAction> actions, AbstractInstance ins, SearchState currState, SearchState goldState, boolean isLossAug) {
		double bestSc = Double.NEGATIVE_INFINITY;
		SearchAction bestAct = null;
		for (SearchAction act : actions) {
			currState.doAction(act);
			
			act.score = getEScoring(ins, currState.structOutput);//scoring(wv, (IInstance)ins, (IStructure)currState.structOutput);
			act.accFrac = lossFunc.computeZeroOneAcc(ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput));
			act.acc = act.accFrac.getVal();
			if (isLossAug) {
				act.score += lossFunc.computeZeroOneLosss((IInstance)ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput)).getVal();//getLossDouble(ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput));
			}
			double sc = (act.score);
			if (sc > bestSc) {
				bestSc = sc;
				bestAct = act;
			}
			
			currState.undoAction(act);
		}
		return bestAct;
	}
	
	public double getEScoring(AbstractInstance ins, AbstractOutput output) {
		HashMap<Integer, Double> efeature = efeaturizer.featurize(ins, output);
		double sc = efuncScorer.getScoring(ins, efeature);
		return sc;
	}

}