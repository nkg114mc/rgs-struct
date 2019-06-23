package search;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearning.EInferencer;
import general.AbstractActionGenerator;
import general.AbstractFactorGraph;
import general.AbstractFeaturizer;
import general.AbstractInstance;
import general.AbstractLossFunction;
import general.AbstractOutput;
import general.FactorGraphBuilder;
import general.FactorGraphBuilder.FactorGraphType;
import init.RandomStateGenerator;
import search.loss.LossScore;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;

public class GreedySearcher implements Serializable {

	private static final long serialVersionUID = -7072579679479073581L;
	public Random random = new Random();
	public RandomStateGenerator initGenerator;
	
	private AbstractLossFunction lossFunc;
	private FactorGraphType fgType;
	private AbstractFeaturizer featurizer;
	private ZobristKeys zobKeys;
	
	private AbstractActionGenerator actionGener;
	
	////
	private EInferencer cachedEvalInferencer;
	
	// parameters
	public int randInitSize = 1;
	public int maxBranch = -1;
	
	// result info
	private double cachedTruAcc = 0;
	
	public GreedySearcher(FactorGraphType fgt, AbstractFeaturizer fzr, int restart, AbstractActionGenerator agen, RandomStateGenerator rndGener, AbstractLossFunction lossf, ZobristKeys abkeys) {
		fgType = fgt;
		featurizer = fzr;
		randInitSize = restart;
		zobKeys = abkeys;
		initGenerator = rndGener;
		lossFunc = lossf;
		actionGener = agen;
		
		cachedEvalInferencer = null;
	}

	@Override
	public Object clone(){
		return new GreedySearcher(fgType, featurizer, randInitSize,actionGener, initGenerator, lossFunc, zobKeys);
	}

	// e_wght is for evaluation function weight
	public SearchResult runSearchWithRestarts(WeightVector wv, EInferencer einfr, AbstractInstance input, AbstractOutput gold, boolean doLossAug) {
		return runSearchWithRestarts(wv, einfr, randInitSize, input, gold, doLossAug);
	}

	// e_wght is for evaluation function weight
	public SearchResult runSearchWithRestarts(WeightVector wv, EInferencer einfr, int restartNum, AbstractInstance input, AbstractOutput gold, boolean doLossAug) {
	
		AbstractInstance x = (AbstractInstance) input;
		AbstractOutput ystar = (AbstractOutput) gold;
		
		if (doLossAug) {
			if (gold == null) {
				throw new RuntimeException("Gold should not be null if do loss-augmented inference!");
			}
		}
		
		// gold state
		SearchState goldState = new SearchState(ystar);
		
		// init states
		HashSet<SearchState> initStates = initGenerator.generateRandomInitState(x, restartNum);
		List<SearchState> sortedInitStates = SetToList(initStates);
		
		List<SearchTrajectory> e_trajs = null;//new ArrayList<SearchTrajectory>();
		List<SearchTrajectory> trajs = new ArrayList<SearchTrajectory>();
		List<Double> predSc = new ArrayList<Double>();
		float bestScore = Float.NEGATIVE_INFINITY;
		SearchState pred = null;
		double bestTruAcc = Double.NEGATIVE_INFINITY;
		cachedTruAcc = Double.NEGATIVE_INFINITY;
		
		
		//////////////////////////////////////
		if (einfr != null) {
			sortedInitStates = einfr.generateMultiInitStates(input, gold, sortedInitStates);//.generateOneInitState(input, gold, initState);
		}
		//////////////////////////////////////
		
		int bestRestart = -1;
		int restartCnt = 0;
		//for (SearchState initState : initStates) {
		for (int rst = 0; rst < sortedInitStates.size(); rst++) {
			SearchState initState = sortedInitStates.get(rst);
			
			restartCnt++;
	
			SearchState y_end_state = initState; // y_start
			if (einfr != null) {
				 y_end_state = einfr.generateOneInitState(input, gold, initState); // y_end (search inferencer applied here)
			}


			// each restart...
			SearchResult bret = doHillClimbing(wv, x, y_end_state, goldState, Integer.MAX_VALUE, doLossAug);
			
			if (bret.predState.score > bestScore) {
				bestScore = bret.predState.score;
				pred = bret.predState;
				bestRestart = restartCnt - 1;
				
			}
			if (bret.accuarcy > bestTruAcc) {
				bestTruAcc = bret.accuarcy;
			}
			
			//trajs.add(bret.trajectories.get(0));
			trajs.add(bret.getUniqueTraj());
			predSc.add((double)bret.predState.score);
		}
		

		// check generation loss
		if (bestTruAcc != cachedTruAcc) {
			throw new RuntimeException("GenTruAcc: " + bestTruAcc + " != " + cachedTruAcc);
		}
		
		//System.out.println("GenTruAcc: " + bestTruAcc + " == " + cachedTruAcc);
		//System.out.println("randsearch");

		SearchResult finalRe = new SearchResult();
		finalRe.accuarcy = bestTruAcc;
		finalRe.predScore = bestScore;
		finalRe.predState = pred;
		finalRe.trajectories = trajs;
		finalRe.e_trajs = e_trajs;
		for (int d = 0; d < predSc.size(); d++) {
			finalRe.addPredScore(predSc.get(d));
		}
		finalRe.bestRank = bestRestart;
		
		//return (pred.structOutput);
		return finalRe;
	}
	
	// no random restart...
	public SearchResult runSearchGivenInitState(WeightVector wv, 
												AbstractInstance input, 
												AbstractOutput initOutput, 
												AbstractOutput gold, 
												boolean doLossAug) {
		
		AbstractInstance x = (AbstractInstance) input;
		AbstractOutput ystar = (AbstractOutput) gold;
		
		if (doLossAug) {
			if (gold == null) {
				throw new RuntimeException("Gold should not be null if do loss-augmented inference!");
			}
		}
		
		// gold state
		SearchState goldState = new SearchState(ystar);
		
		// initial state
		SearchState initState = new SearchState(initOutput);
	
		// run for just once~
		SearchResult bret = doHillClimbing(wv, x, initState, goldState, Integer.MAX_VALUE, doLossAug);
		return bret;
	}
	
	// return best state with hill-climbing
	//public SearchState doHillClimbing(WeightVector wv, AbstractInstance ins, SearchState initState, SearchState goldState, int maxDepth, boolean isLossAug) {
	public SearchResult	 doHillClimbing(WeightVector wv, AbstractInstance ins, SearchState initState, SearchState goldState, int maxDepth, boolean isLossAug) {
	
		// if perform loss augmented inference, then 
		if (isLossAug) {
			if (goldState == null) {
				throw new RuntimeException("Error: perform loss-aug inference but no gold-state!");
			}
		}
		
		SearchHashTable hashTb = new SearchHashTable(zobKeys);
		
		//HwFactorGraph factorGraph = new HwFactorGraph((HwInstance)ins, featurizer);
		AbstractFactorGraph factorGraph = FactorGraphBuilder.getFactorGraph(fgType, ins, featurizer);
		factorGraph.updateScoreTable(wv.getDoubleArray());
		
		SearchTrajectory trajectory = new SearchTrajectory();
		
		// about initial state
		SearchState currState = initState;
		float currScore = (float)scoring(wv, (IInstance)ins, (IStructure)currState.structOutput);
		if (isLossAug) {
			float loss = getLossFloat(ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput));
			currScore += loss;
		}
		float lastScore = currScore;
		currState.score = currScore;
		double bestAcc = Double.NEGATIVE_INFINITY;
		if (goldState.structOutput != null) {
			bestAcc = computeZeroOneAcc(ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput));
			currState.trueAcc = bestAcc;
			currState.trueAccFrac = lossFunc.computeZeroOneAcc(ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput));
			if (bestAcc > cachedTruAcc) {
				cachedTruAcc = bestAcc;
			}
		}
		hashTb.insertNewInstance(currState); // insert initial state to hashTb

		int depth = 0;
		float highestScore = currScore;
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
			
			//SearchState[] bestTrueAccState = new SearchState[1];
			double[] bTrueAcc = new double[1];
			
			// try best
			//SearchAction bestAct1 = pickBestPredAction(actions, wv, ins, currState, goldState);
			//SearchAction bestAct2 = pickBestPredActionFast(factorGraph, actions, wv, ins, currState, goldState, isLossAug);
			SearchAction bestAct3 = pickBestPredActionIncreamentally(factorGraph, actions, wv, ins, hashTb, currState, goldState, isLossAug, bTrueAcc);//, bTrueAccFrac);
			SearchAction bestAct = bestAct3;
			if (bestAct == null) { // no Non-repeat actions anymore ...
				break;
			}
			float bestSc = (float)bestAct.score;
			
			// store best true accuracy
			if (bTrueAcc[0] > bestAcc) {
				bestAcc = bTrueAcc[0];
			}
			
			if (highestScore < bestSc) {
				highestScore = bestSc;
			}
			
			// stop condition 2: Reach hill peak
			currScore = bestSc;
			
			if (goldState.structOutput != null) {
				if (currState.trueAccFrac == null) {
					throw new RuntimeException("trueAccFrac == null");
				}
			}
			//if (bestAct.accFrac != null) { currState.trueAccFrac = bestAct.accFrac.getSelfCopy(); }
			checkStateAccConsistency(currState);
			if (currScore <= lastScore) { // reach peak
				break;
			} else {
				currState.doAction(bestAct);
				currState.score = currScore;
				currState.trueAcc = bestAct.acc;
				currState.trueAccFrac = bestAct.accFrac;
			}
			
			///System.out.println("score = " + currScore);
			// prepare for next depth
			lastScore = currScore;
			hashTb.insertNewInstance(currState); // avoid repeat
			
			trajectory.concatenateState(currState); // record the trajectory state
		}
		
		//System.out.println("Hash Size = " + hashTb.getSize());
		//System.out.println("Searched depth = " + depth);
		
		
		// check before adding to the traj
		//checkStatePredScoreCorr(depth, ins, currState, wv, goldState, isLossAug);
		assert (highestScore == currState.score);
		
		SearchResult ret = new SearchResult();
		ret.predState = currState;
		ret.predScore = currState.score;
		ret.accuarcy = bestAcc;
		ret.addTraj(trajectory);
		
		return ret;
		//return currState;
	}
	
	private void checkStatePredScoreCorr(int stp, AbstractInstance ins, SearchState state, WeightVector wv,  SearchState goldState, boolean isLossAug) {

		if (goldState.structOutput != null) {
			double computeAcc = computeZeroOneAcc(ins, (IStructure)(goldState.structOutput), (IStructure)(state.structOutput));
			if (computeAcc != state.trueAcc) {
				System.err.println("Step "+stp+" Acc incorr: " + computeAcc + "!=" + state.trueAcc);
				throw new RuntimeException("Accuracy inconsistent!");
			}
		}
		
		double computeSc = scoring(wv, (IInstance)ins, (IStructure)state.structOutput);
		if (isLossAug) {
			float loss = getLossFloat(ins, (IStructure)(goldState.structOutput), (IStructure)(state.structOutput));
			computeSc += loss;
		}
		float computeScf = (float)computeSc;
		if (computeScf != state.score) {
			System.err.println("Step "+stp+" Score incorr: " + computeSc + "!=" + state.score);
			throw new RuntimeException("Score  inconsistent!");
		}
		
	}
	
	private void checkStateAccConsistency(SearchState state) {
		if ((!Double.isInfinite(state.trueAcc)) && (state.trueAccFrac != null)) {
			if (state.trueAccFrac.getVal() != state.trueAcc) {
				throw new RuntimeException("Accuracy inconsistent: " + state.trueAccFrac.getVal() + "!=" + state.trueAcc);
			}			
		}
	}
	
	public SearchAction pickBestPredAction(List<SearchAction> actions, WeightVector wv, AbstractInstance ins, SearchState currState, SearchState goldState, boolean isLossAug) {
		float bestSc = Float.NEGATIVE_INFINITY;
		SearchAction bestAct = null;
		for (SearchAction act : actions) {
			currState.doAction(act);
			
			act.score = scoring(wv, (IInstance)ins, (IStructure)currState.structOutput);
			if (isLossAug) {
				act.score += getLossDouble(ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput));
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
	public SearchAction pickBestPredActionFast(AbstractFactorGraph factorGraph, List<SearchAction> actions, WeightVector wv, AbstractInstance ins, SearchState currState, SearchState goldState, boolean isLossAug) {
		
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
				loss = getLossDouble(ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput));
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
	public SearchAction pickBestPredActionIncreamentally(AbstractFactorGraph factorGraph, 
			                                             List<SearchAction> actions, 
			                                             WeightVector wv, 
			                                             AbstractInstance ins, 
			                                             SearchHashTable hashTb, 
			                                             SearchState currState, 
			                                             SearchState goldState, 
			                                             boolean isLossAug,
			                                             double[] bestTrueAcc) {
		
		double bestTruAcc = Double.NEGATIVE_INFINITY;
		double bestSc = Double.NEGATIVE_INFINITY;
		SearchAction bestAct = null;
		int updateCnt = 0;
		int nonRepeatCnt = 0;
		
		// the old score
		//double oldLoss = 0;
		LossScore oldLossScore = null;
		if (isLossAug) {
			if (goldState.structOutput != null) {
				oldLossScore = lossFunc.computeZeroOneLosss((IInstance)ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput));
				//oldLoss = oldLossScore.getVal();//getLossDouble(ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput));
			}
		}
		//double oldTrueAcc = 0;
		LossScore oldTrueAccScore = null;
		if (goldState.structOutput != null) {
			oldTrueAccScore = lossFunc.computeZeroOneAcc(ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput));
			//oldTrueAcc = oldTrueAccScore.getVal();//computeZeroOneAcc(ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput));
		}
		double oldPredScore = factorGraph.computeScoreWithTable(wv.getDoubleArray(), (HwOutput)currState.structOutput);
	   	int  oldStateHash = hashTb.computeIndex(currState);
		
		for (SearchAction act : actions) {
			currState.doAction(act);
			
			// is repeated action
			boolean stateIsNotRepeat = (!hashTb.probeExistenceWithAction(currState, oldStateHash, act));
			if (stateIsNotRepeat) {
				nonRepeatCnt++;
				
				double loss = 0;
				if (isLossAug) {
					//double loss1 = getLossDouble(ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput));
					LossScore newLossScore = oldLossScore.addWith(lossFunc.computeZeroOneLossIncreamentally(act, (IInstance)ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput)));
					double loss2 = newLossScore.getVal();
					//double loss2 = oldLoss + computeZeroOneLossIncreamentally(act, (IInstance)ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput));
					
					//LossScore loss1Sc = lossFunc.computeZeroOneLosss((IInstance)ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput));
					//equalCheck(loss1, loss2);
					loss = loss2;
				}
				
				double trueAcc = Double.NEGATIVE_INFINITY;
				LossScore trAccFrac = null;
				if (goldState.structOutput != null) {
					LossScore newLossAccScore = oldTrueAccScore.addWith(lossFunc.computeZeroOneAccIncreamentally(act, ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput)));
					trueAcc = newLossAccScore.getVal();
					trAccFrac = newLossAccScore.getSelfCopy();
					assert (trAccFrac != null);
					//trueAcc = oldTrueAcc + computeZeroOneAccIncreamentally(act, ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput));
					//double tacc = computeZeroOneAcc(ins, (IStructure)(goldState.structOutput), (IStructure)(currState.structOutput));
					//equalCheck(trueAcc, tacc);
				}
				
				
				
				//double sc1 = scoring(wv, (IInstance)ins, (IStructure)currState.structOutput) + loss;
				//double sc2 = factorGraph.computeScoreWithTable(wv.getDoubleArray(), (HwOutput)currState.structOutput) + loss;
				double sc3 = oldPredScore + factorGraph.computeScoreDiffWithTable(wv.getDoubleArray(), act, (HwOutput)currState.structOutput) + loss;
				//equalCheckThree(sc1, sc2, sc3);
				//equalCheck(sc2, sc3);
				//equalCheck(sc1, sc2);

				act.score = sc3;
				act.acc = trueAcc;
				act.accFrac = trAccFrac;
				double sc = (act.score);
				if (sc > bestSc) {
					bestSc = sc;
					bestAct = act;
					updateCnt++;
				}
				
				if (goldState != null) {
					// true accuracy
					if (trueAcc > bestTruAcc) {
						bestTruAcc = trueAcc;
						//bestTruAccFrac = trAccFrac;
					}
					if (trueAcc > cachedTruAcc) {
						cachedTruAcc = trueAcc;
					}
				}

			}
			
			currState.undoAction(act);
		}
		
		if (nonRepeatCnt > 0) {
			if (bestAct == null) {
				System.out.println(currState.structOutput.toString());
				System.out.println(actions.size());
				System.out.println("updateCnt = " + updateCnt);
				System.out.println("NonRepeatCnt = " + nonRepeatCnt);
				System.out.println("bestSc = " + bestSc);
				System.out.println("oldPredScore = " + oldPredScore);
				for (SearchAction a : actions) {
					System.out.println("aSc = " + a.score);
				}
				
			}
			assert (bestAct != null);
		} else {
			// OK, no action avaliable ...
		}
		
		
		// return
		bestTrueAcc[0] = bestTruAcc;
		//bestTrueAccFrac[0] = bestTruAccFrac;
		return bestAct;
	}
	
	public AbstractLossFunction getLossFunc() {
		return lossFunc;
	}
	
	public AbstractActionGenerator getActionGener() {
		return actionGener;
	}

	public String getInitGenerName() {
		String initName = initGenerator.getType().toString().toLowerCase();
		return initName;
	}
	
	public RandomStateGenerator getInitGenerator() {
		return initGenerator;
	}
	
	public FactorGraphType getFactorGraphType() {
		return fgType;
	}
	
	public ZobristKeys getZobKeys() {
		return zobKeys;
	}
	
	public AbstractFeaturizer getFeaturizer() {
		return featurizer;
	}


/*
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
	
	public static double computeZeroOneLosss(IInstance ins, IStructure goldStructure,  IStructure structure) {
		AbstractOutput goldLabeledSeq = (AbstractOutput) goldStructure;
		double loss = 0;
		for (int i = 0; i < goldLabeledSeq.size(); i++) {
			if (((AbstractOutput) structure).getOutput(i) != goldLabeledSeq.getOutput(i)) {
				loss += 1.0;
			}
		}
		return loss;
	}
	//// ====LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS ====================================================
	

	
	public static double computeZeroOneAccIncreamentally(SearchAction action, AbstractInstance ins, IStructure goldStructure,  IStructure structure) {
		AbstractOutput goldLabeledSeq = (AbstractOutput) goldStructure;
		double accChange = 0;
		int goldValue = goldLabeledSeq.getOutput(action.getSlotIdx());
		if (action.getOldVal() == goldValue) { // old is correct
			accChange -= 1.0;
		}
		if (action.getNewVal() == goldValue) {
			accChange += 1.0;
		}
		return accChange;
	}
	
	public static double computeZeroOneAcc(AbstractInstance ins, IStructure goldStructure,  IStructure structure) {
		AbstractOutput goldLabeledSeq = (AbstractOutput) goldStructure;
		double acc = 0;
		for (int i = 0; i < goldLabeledSeq.size(); i++) {
			if (((AbstractOutput) structure).getOutput(i) == goldLabeledSeq.getOutput(i)) {
				acc += 1.0;
			}
		}
		return acc;
	}
*/	
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
/*
	public double getLossDouble(AbstractInstance ins, IStructure goldStructure,  IStructure structure) {
		double dl = computeZeroOneLosss((IInstance)ins, goldStructure, structure);
		return (dl);
	}

	public float getLossFloat(AbstractInstance ins, IStructure goldStructure,  IStructure structure) {
		double dl = computeZeroOneLosss((IInstance)ins, goldStructure, structure);
		return ((float)dl);
	}
*/
	public double getLossDouble(AbstractInstance ins, IStructure goldStructure,  IStructure structure) {
		LossScore lsc = lossFunc.computeZeroOneLosss((IInstance)ins, goldStructure, structure);
		return (lsc.getVal());
	}

	public float getLossFloat(AbstractInstance ins, IStructure goldStructure,  IStructure structure) {
		LossScore lsc = lossFunc.computeZeroOneLosss((IInstance)ins, goldStructure, structure);
		return ((float)lsc.getVal());
	}
	
	public double computeZeroOneAcc(AbstractInstance ins, IStructure goldStructure,  IStructure structure) {
		LossScore accSc = lossFunc.computeZeroOneAcc(ins, goldStructure, structure);
		return accSc.getVal();
	}
	
	public void setEvalInferencer(EInferencer cEvalInfr) {
		cachedEvalInferencer = cEvalInfr;
	}
	
	public EInferencer getEvalInferencer() {
		return cachedEvalInferencer;
	}
	
	public void setRestart(int newRestart) {
		randInitSize = newRestart;
	}
	
	public double getCachedTrueAccuracy() {
		return cachedTruAcc;
	}
	
	public static List<SearchState> SetToList(Set<SearchState> stateSet) {
		ArrayList<SearchState> stateList = new ArrayList<SearchState>();
		for (SearchState s : stateSet) {
			stateList.add(s);
		}
		return stateList;
	}

}
