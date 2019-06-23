package elearning;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import berkeleyentity.MyTimeCounter;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearning.einfer.ESearchInferencer;
import elearning.einfer.SearchStateScoringFunction;
import elearnnew.SamplingELearning;
import general.AbstractFeaturizer;
import general.AbstractInstance;
import general.AbstractLossFunction;
import general.AbstractOutput;
import init.RandomStateGenerator;
import search.GreedySearcher;
import search.SearchResult;
import search.SearchState;
import search.SearchTrajectory;
import search.loss.GoldPredPair;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;

public class EfuncInferenceJuly2017 {
	
	///////////////////////////////////////////////
	//// Part 2, apply speedup in testing only
	///////////////////////////////////////////////

	
	// test evaluation function speedup

	public static void testEvaluationSpeedupJuly2017(List<HwInstance> tstInsts,
                                                     WeightVector cost_weight, 
                                                     GreedySearcher gsearcher,
                                                     int restartNum,
                                                     String namePrefix,
                                                     ////////////////////////////////////
                                                     boolean useEval,
                                                     AbstractFeaturizer efeaturizer,
		                                             AbstractRegressionLearner regLrner,
		                                             EInferencer einfr) {

		TrajectoryPloter ploter = new TrajectoryPloter(restartNum, 32767, gsearcher.getLossFunc());
		
		List<GoldPredPair> results = new ArrayList<GoldPredPair>();
		
		
		int[] bestRank = new int[restartNum];
		Arrays.fill(bestRank, 0);

		
		
		
		for (int i = 0; i < tstInsts.size(); i++) {
			
			HwInstance inst = tstInsts.get(i);
			HwOutput gold = inst.getGoldOutput();

			SearchResult infrRe =  searchWithRestartsMaybeEval(gsearcher, cost_weight,    restartNum, inst, gold, false,    useEval,efeaturizer,regLrner,(ESearchInferencer)einfr);
			//checkResult(infrRe);
			//SamplingELearning.checkRestartScores(infrRe);
			
			int bestIdx = infrRe.bestRank;
			bestRank[bestIdx]++;
			
			GoldPredPair re = new GoldPredPair((AbstractInstance)inst, (AbstractOutput)gold, (AbstractOutput)(infrRe.predState.structOutput));
			results.add(re);

			//////////////////////////////////
			if (gsearcher.getLossFunc().getClass().getSimpleName().equals("SearchLossHamming")) {
				EFunctionLearning.normalizeHammingLoss(infrRe, inst, gsearcher.getLossFunc()); //System.out.println("Normalize accuracy...");
			}
			
			ploter.addOneResult(infrRe);
		}
		
		

		// local optima analysis
		SamplingELearning.printLocalOptimaMeasure(bestRank, tstInsts.size(), restartNum);
		

		// naming prefix
		String useEfunc = "noEfunc";
		if (useEval) {
			useEfunc = "withEfunc";
		}
		String initTyp = gsearcher.getInitGenerName();
		
		String finalPrefix = namePrefix + "_" + initTyp +  "_" + useEfunc + "_" + "restr" + String.valueOf(restartNum);
		ploter.plotToFile(finalPrefix);
		
		AbstractLossFunction lossfunc = gsearcher.getLossFunc();
		double lossFuncAcc = lossfunc.computeMacroFromResults(results);
		System.out.println("LossFunction computed accuracy = " + lossFuncAcc);
		System.out.println("");
	}
	
	////////////////////////////
	////////////////////////////
	//// Run search inference
	////////////////////////////
	////////////////////////////
	

	
	// e_wght is for evaluation function weight
	public static SearchResult searchWithRestartsMaybeEval(GreedySearcher gsearcher, 
			                                               WeightVector cost_wv, 
			                                               
			                                               int restartNum, 
			                                               AbstractInstance input, 
			                                               AbstractOutput gold, 
			                                               boolean doLossAug,
			                                               
			                                               boolean useEval,
			                                               AbstractFeaturizer efeaturizer,
			                                               AbstractRegressionLearner regLrner,
			                                               ESearchInferencer einfr) {
	
		RandomStateGenerator initGenerator = gsearcher.getInitGenerator();
		
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
		List<SearchState> sortedInitStates = SetToListRndOrder(initStates);
		
		List<SearchTrajectory> e_trajs = new ArrayList<SearchTrajectory>();
		List<SearchTrajectory> trajs = new ArrayList<SearchTrajectory>();
		List<Double> predSc = new ArrayList<Double>();
		float bestScore = Float.NEGATIVE_INFINITY;
		SearchState pred = null;
		double bestTruAcc = Double.NEGATIVE_INFINITY;


		
		
		
		//// eval function
		SearchStateScoringFunction evalFunc = getInitializedEvalFunc();
		einfr.setEvalScoringFunc(evalFunc);
		
		
		//// about eval function training
		ArrayList<RegressionInstance> regrDataIter = new ArrayList<RegressionInstance>();

		
		
		SearchState y_start = getNonRepeatedState(0, sortedInitStates);
		
		int bestRestart = -1;
		int restartCnt = 0;
		
		
		// time counter array
		ArrayList<Long> iterTimes = new ArrayList<Long>();
		ArrayList<Long> accuTimes = new ArrayList<Long>();
		
		MyTimeCounter trTimer = new MyTimeCounter("instance time counter");
		trTimer.start();
		
		long accTimeConsum = 0;
		
		for (int rst = 0; rst < restartNum; rst++) {
			
			long startTime = trTimer.getMilSecondSnapShot();
			
			restartCnt++;

			// do base search ...
			SearchResult bret = gsearcher.doHillClimbing(cost_wv, x, y_start, goldState, Integer.MAX_VALUE, doLossAug);
			SearchState y_end = bret.predState;
			
			if (bret.predState.score > bestScore) {
				bestScore = bret.predState.score;
				pred = bret.predState;
				bestRestart = restartCnt - 1;
				
			}
			if (bret.accuarcy > bestTruAcc) {
				bestTruAcc = bret.accuarcy;
			}
			
			trajs.add(bret.getUniqueTraj());
			predSc.add((double)bret.predState.score);
			
			
			if (rst == (restartNum - 1)) {
				
				long endTime = trTimer.getMilSecondSnapShot();
				
				////////////////////
				// About time //////
				////////////////////
				long timSpend = endTime - startTime; // time spend in this iteration
				accTimeConsum += timSpend; // time accumulation
				
				// store it
				iterTimes.add(timSpend);
				accuTimes.add(accTimeConsum);
				
				
				break; // no need to re-train evaluation function ...
			}
			
			////////////////////////////////
			////////////////////////////////
			////////////////////////////////
			
			/// Apply evaluation function here
			if (useEval) {
				
				if (rst == 0) {
					e_trajs.add(new SearchTrajectory()); // put a empty trajectory (the first base-search do no have meta-search)
				}
				
				
				SearchResult cResult = bret;
						
				// re-train evaluation function
				boolean needUpdate = false;
				if (evalFunc == null) {
					needUpdate = true;
				} else {
					double evalValue = evalFunc.getScoring(x, efeaturizer.featurize(x, y_start.structOutput));
					if (evalValue != cResult.predScore) {
						needUpdate = true;
					}
				}
				if (needUpdate) { // update W_e or 

					// a regression instance: y
					SearchTrajectory traj = cResult.getUniqueTraj(); // should be just one
					List<SearchState> states = traj.getStateList();
					//sumSteps += states.size();
					for (int d =  0; d < states.size(); d++) {
						
						SearchState dstate = states.get(d);

						// one regression training data point
						// featurize data point
						HashMap<Integer, Double> phi_e = efeaturizer.featurize(x, dstate.structOutput);
						double cost_value = cResult.predScore;// dstate.score;
						
						// aggregate data
						RegressionInstance dp = new RegressionInstance(phi_e, cost_value);
						regrDataIter.add(dp);
						
						//break;
					}
					
					// do re-train here
					evalFunc = regLrner.regressionTrain(regrDataIter, efeaturizer.getFeatLen(), -1);
					einfr.setEvalScoringFunc(evalFunc);
				}
				
				
				// do eval-inference starting from y_end
				//SearchState y_restart = einfr.generateOneInitState(x, gold, y_end);
				SearchResult e_result = einfr.generateOneInitStateWithTraj(x, gold, y_end);
				SearchState y_restart = e_result.predState;
				e_trajs.add(e_result.getUniqueTraj());

				// re-assign y_start
				if (!y_restart.isEqualOutput(y_start)) {
					y_start = y_restart;
				} else {
					y_start = getNonRepeatedState(rst + 1, sortedInitStates);
				}
				
			} else { // no eval function
				y_start = getNonRepeatedState(rst + 1, sortedInitStates);  // random pick a new initial state
			}
			
			
			long endTime = trTimer.getMilSecondSnapShot();
			
			
			////////////////////
			// About time //////
			////////////////////
			long timSpend = endTime - startTime; // time spend in this iteration
			accTimeConsum += timSpend; // time accumulation
			
			// store it
			iterTimes.add(timSpend);
			accuTimes.add(accTimeConsum);
		}
		

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
		
		
		finalRe.iterTime = iterTimes;
		finalRe.iterAccumTime = accuTimes;
		
		return finalRe;
	}
	
	
	public static double getStateScore(SearchStateScoringFunction evalFunc, AbstractFeaturizer efeaturizer, AbstractInstance x, AbstractOutput output) {
		HashMap<Integer, Double> phi_e = efeaturizer.featurize(x, output);
		double sc = evalFunc.getScoring(x, phi_e);
		return sc;
	}
	
	public static SearchStateScoringFunction getInitializedEvalFunc() {
		return null;
	}

	public static SearchState getNonRepeatedState(int rst, List<SearchState> genList) { // this generated init states without repeat
		return genList.get(rst);
	}
	
	public static List<SearchState> SetToListRndOrder(Set<SearchState> stateSet) {
		ArrayList<SearchState> stateList = new ArrayList<SearchState>();
		for (SearchState s : stateSet) {
			stateList.add(s);
		}
		return stateList;
	}
}
