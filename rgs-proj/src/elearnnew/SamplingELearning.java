package elearnnew;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearning.AbstractRegressionLearner;
import elearning.EInferencer;
import elearning.RegressionInstance;
import elearning.einfer.ESamplingInferencer;
import elearning.einfer.SearchStateScoringFunction;
import elearning.einfer.XgbSearchStateOneInstanceScorer;
import elearning.einfer.XgbSearchStateSingleInstanceScorer;
import general.AbstractFeaturizer;
import general.AbstractInstance;
import general.AbstractOutput;
import general.FactorGraphBuilder.FactorGraphType;
import init.RandomStateGenerator;
import ml.dmlc.xgboost4j.java.Booster;
import search.GreedySearcher;
import search.SearchResult;
import search.SearchState;
import search.SearchTrajectory;
import search.ZobristKeys;
import sequence.hw.HwInstance;

public class SamplingELearning {

	public static EInferencer learnEFunction(RandomStateGenerator randomGenr,
			List<HwInstance> instances,
			AbstractFeaturizer efeaturizer,
			FactorGraphType fgt,
			ZobristKeys abkeys,

			WeightVector cost_weight, 
			GreedySearcher gsearcher, 

			AbstractRegressionLearner regressionTrainer,
			int iteration,
			boolean applyInstWght){
		
		//SearchStateScoringFunction emd = trainBatchXgb(randomGenr, instances, cost_weight, gsearcher, regressionTrainer, iteration);
		SearchStateScoringFunction emd = trainSingleXgb(randomGenr, instances, cost_weight, gsearcher, regressionTrainer, iteration);
		
		EInferencer einfr = new ESamplingInferencer(randomGenr, cost_weight, gsearcher.getFeaturizer(), emd, efeaturizer, 100);
		
		return einfr;
	}
	
	public static WeightVector aggregateWeights(List<WeightVector> e_ws_iter, int wlen) {
		
		if (e_ws_iter.size() == 0) {
			WeightVector empt = (new WeightVector(wlen));
			for (int j = 0; j < wlen; j++) {
				empt.setElement(j, 0);
			}
			return empt;
		} else {
			return e_ws_iter.get(e_ws_iter.size() - 1);
		}

	}
	
	public static SearchStateScoringFunction trainBatchXgb(RandomStateGenerator randomGenr,
			List<HwInstance> trainInstances,//AbstractInstance ainst,
			 
			 WeightVector cost_weight, 
			 GreedySearcher gsearcher, 

			 AbstractRegressionLearner regressionTrainer,
			 int iteration) {
		
		AbstractFeaturizer cfeaturizer = gsearcher.getFeaturizer();
		
		
		ArrayList<RegressionInstance> regrDataIter = new ArrayList<RegressionInstance>();
		
		for (AbstractInstance ainst : trainInstances) {
			for (int iter = 0; iter < iteration; iter++) {

				AbstractOutput y_start = (randomGenr.generateSingleRandomInitState(ainst).structOutput);

				SearchResult cResult = gsearcher.runSearchGivenInitState(cost_weight, ainst, y_start, null, false);
				//AbstractOutput y_real = cResult.predState.structOutput;

				// a regression instance: y
				SearchTrajectory traj = cResult.getUniqueTraj(); // should be just one
				List<SearchState> states = traj.getStateList();

				for (int d = 0; d < states.size(); d++) {

					SearchState dstate = states.get(d);

					// one regression training data point
					// featurize data point
					HashMap<Integer, Double> phi_e = cfeaturizer.featurize(ainst, dstate.structOutput);
					double cost_value = cResult.predScore;// dstate.score;

					// aggregate data

					double instWght = 1;
					RegressionInstance dp = new RegressionInstance(phi_e, cost_value);
					regrDataIter.add(dp);
				}

			}
		}
		
		System.out.println("Regression Data Size: " + regrDataIter.size());
		
		// train to get new e_function
		SearchStateScoringFunction regModel = regressionTrainer.regressionTrain(regrDataIter, cfeaturizer.getFeatLen(), 0);

		return regModel;
	}
	
/*
	public static void workOnOneInstance(RandomStateGenerator randomGenr,
										 AbstractInstance ainst,
										 
										 WeightVector cost_weight, 
										 GreedySearcher gsearcher, 

										 AbstractRegressionLearner regressionTrainer,
										 int iteration) {
		
		AbstractFeaturizer cfeaturizer = gsearcher.getFeaturizer();
		
		ArrayList<RegressionInstance> regrDataIter = new ArrayList<RegressionInstance>();
		for (int iter = 0; iter < iteration; iter++) {

			AbstractOutput y_start = (randomGenr.generateSingleRandomInitState(ainst).structOutput);

			SearchResult cResult = gsearcher.runSearchGivenInitState(cost_weight, ainst, y_start, null, false);
			//AbstractOutput y_real = cResult.predState.structOutput;

			// a regression instance: y
			SearchTrajectory traj = cResult.getUniqueTraj(); // should be just one
			List<SearchState> states = traj.getStateList();

			for (int d = 0; d < states.size(); d++) {

				SearchState dstate = states.get(d);

				// one regression training data point
				// featurize data point
				HashMap<Integer, Double> phi_e = cfeaturizer.featurize(ainst, dstate.structOutput);
				double cost_value = cResult.predScore;// dstate.score;
		
				// aggregate data

				double instWght = 1;
				RegressionInstance dp = new RegressionInstance(phi_e, cost_value);
				regrDataIter.add(dp);
				
				break;
			}

		}
		
		// train to get new e_function
		SearchStateScoringFunction regModel = regressionTrainer.regressionTrain(regrDataIter, cfeaturizer.getFeatLen(), 0);
		WeightVector e_weight = (WeightVector)regModel.getModel();
		
		
		int restart = 20;
		int eSampling = 100;
		testOneInstance(randomGenr, ainst, cost_weight, gsearcher, e_weight, restart, eSampling);
		
	}
	
	public static void testOneInstance(RandomStateGenerator randomGenr,
									   AbstractInstance ainst,
			 
									   WeightVector cost_weight, 
									   GreedySearcher gsearcher, 

									   WeightVector e_weight,
									   
									   int originRestart,
									   int eSampling) {
		

		
		HashSet<SearchState> istates = randomGenr.generateRandomInitState(ainst, originRestart);
		
		System.out.println("====================");
		// random pick some instance
		int i = 0;
		for (SearchState initState : istates) {
			i++;
			
			////////// original greedy search ////////// 
			
			AbstractOutput y_start = initState.structOutput;
			double initCost1 = gsearcher.scoring(cost_weight, (IInstance)ainst, (IStructure)y_start);
			//System.out.println("i---" + y_start.toString());
			
			SearchResult cResult1 = gsearcher.runSearchGivenInitState(cost_weight, ainst, y_start, null, false);
			SearchState y_real_baseline = cResult1.predState;
			
			
			////////// search with evaluation /////////
			
			AbstractOutput y_end = samplingYEnd(randomGenr, ainst, cost_weight, gsearcher, e_weight, eSampling).structOutput;
			double initCost2 = gsearcher.scoring(cost_weight, (IInstance)ainst, (IStructure)y_end);
					
			SearchResult cResult2 = gsearcher.runSearchGivenInitState(cost_weight, ainst, y_end, null, false);
			SearchState y_real_our = cResult2.predState;

			
			System.out.println(i + "  " + initCost1 + "--->" + cResult1.predScore + " " + initCost2 + "--->" + cResult2.predScore);
			//System.out.println(y_start.toString());
		}
		System.out.println("----------------------");

	}

	public static SearchState samplingYEnd(RandomStateGenerator randomGenr,
			   AbstractInstance ainst,
				 
			   WeightVector cost_weight, 
			   GreedySearcher gsearcher, 

			   WeightVector e_weight,
			   int eSampling) {
		
		SearchState bestYEnd = null;
		double bestEval = -Double.MAX_VALUE;

		
		AbstractFeaturizer cfeaturizer = gsearcher.getFeaturizer();
		
		HashSet<SearchState> jstates = randomGenr.generateRandomInitState(ainst, eSampling);
		for (SearchState initState : jstates) {

			HashMap<Integer, Double> phi_e = cfeaturizer.featurize(ainst, initState.structOutput);
			double eval_value = GreedySearcher.myDotProduct(phi_e, e_weight);

			if (eval_value > bestEval) {
				bestEval = eval_value;
				bestYEnd = initState;
			}
		}
		
		
		assert(bestYEnd != null);
		return bestYEnd;
	}
*/
	
	
	
	public static SearchStateScoringFunction trainSingleXgb(RandomStateGenerator randomGenr,
			List<HwInstance> trainInstances,
			 
			 WeightVector cost_weight, 
			 GreedySearcher gsearcher, 

			 AbstractRegressionLearner regressionTrainer,
			 int iteration) {
		
		HashMap<AbstractInstance, Booster> instModels = new HashMap<AbstractInstance, Booster>();
		
		for (AbstractInstance inst : trainInstances) {
			Booster instanceBooster = workOnOneInstance(randomGenr, inst, cost_weight, gsearcher, regressionTrainer, iteration);
			instModels.put(inst, instanceBooster);
		}

		XgbSearchStateSingleInstanceScorer instScr = new XgbSearchStateSingleInstanceScorer(instModels);
		return instScr;
	}
	
	
	//// All workOnOneInstance
	public static SearchStateScoringFunction trainOneInstanceXgb(RandomStateGenerator randomGenr,
			AbstractInstance inst,
			 
			 WeightVector cost_weight, 
			 GreedySearcher gsearcher, 

			 AbstractRegressionLearner regressionTrainer,
			 int iteration) {

		Booster instanceBooster = workOnOneInstance(randomGenr, inst, cost_weight, gsearcher, regressionTrainer, iteration);
		XgbSearchStateOneInstanceScorer instScr = new XgbSearchStateOneInstanceScorer(inst, instanceBooster);
		return instScr;
	}
	
	
	
	
	public static Booster workOnOneInstance(RandomStateGenerator randomGenr,
			AbstractInstance ainst,

			WeightVector cost_weight, 
			GreedySearcher gsearcher, 

			AbstractRegressionLearner regressionTrainer,
			int iteration) {

		AbstractFeaturizer cfeaturizer = gsearcher.getFeaturizer();

		ArrayList<RegressionInstance> regrDataIter = new ArrayList<RegressionInstance>();
		for (int iter = 0; iter < iteration; iter++) {

			AbstractOutput y_start = (randomGenr.generateSingleRandomInitState(ainst).structOutput);

			SearchResult cResult = gsearcher.runSearchGivenInitState(cost_weight, ainst, y_start, null, false);

			// a regression instance: y
			SearchTrajectory traj = cResult.getUniqueTraj(); // should be just one
			List<SearchState> states = traj.getStateList();

			for (int d = 0; d < states.size(); d++) {

				SearchState dstate = states.get(d);

				// one regression training data point
				// featurize data point
				HashMap<Integer, Double> phi_e = cfeaturizer.featurize(ainst, dstate.structOutput);
				double cost_value = cResult.predScore;// dstate.score;

				// aggregate data

				double instWght = 1;
				RegressionInstance dp = new RegressionInstance(phi_e, cost_value);
				regrDataIter.add(dp);

				//break;
			}

		}

		// train to get new e_function
		SearchStateScoringFunction regModel = regressionTrainer.regressionTrain(regrDataIter, cfeaturizer.getFeatLen(), 0);
		Booster bostr = (Booster)regModel.getModel();
		
		return bostr;
	}
	
/*
	public static void exploreLocalOptimal(RandomStateGenerator randomGenr,
			List<HwInstance> instances,

			WeightVector cost_weight, 
			GreedySearcher gsearcher,
			int restarts,
			EInferencer einfer){
		
		
		int[] bestRank = new int[restarts];


		for (AbstractInstance ainst : instances) {
			
			SearchResult tmpRe = gsearcher.runSearchWithRestarts(cost_weight, einfer, 20, ainst, null, false);
			checkRestartScores(tmpRe);
			
			
			HashSet<SearchState> istates = randomGenr.generateRandomInitState(ainst, restarts);
			
			int bestIdx = 0;
			double bestCost = -Double.MAX_VALUE;
			
			int i = 0;
			for (SearchState initState : istates) {
				i++;
				
				AbstractOutput y_start = initState.structOutput;
				AbstractOutput y_end = y_start;
				
				///////////////////////////////////////////////////////////////////////
				if (einfer != null) {
					SearchState y_end_state = einfer.generateOneInitState(ainst, null, initState);
					y_end = y_end_state.structOutput; // replace it with evaluation function picked init state
				}
				///////////////////////////////////////////////////////////////////////
						
				//double initCost1 = gsearcher.scoring(cost_weight, (IInstance)ainst, (IStructure)y_start);
				
				//SearchResult cResult1 = gsearcher.runSearchGivenInitState(cost_weight, ainst, y_start, null, false);
				SearchResult cResult1 = gsearcher.runSearchGivenInitState(cost_weight, ainst, y_end, null, false);
				SearchState y_real_baseline = cResult1.predState;
				double y_real_cost = cResult1.predScore;
				
				if (y_real_cost > bestCost) {
					bestCost = y_real_cost;
					bestIdx = i - 1;
				}

			}
			
			bestRank[bestIdx]++;
		}
		
		int    firstBest = -1;
		double firstRate = -1;
		
		double expectedRestarts = 0;
		double expectedRestarts2 = 0;
		
		double deno = (double)instances.size();
		for (int j = 0; j < restarts; j++) {
			double num = (double)bestRank[j];
			double per = num / deno;
			System.out.println(j + ": " + bestRank[j]+ " " + per);
			
			expectedRestarts += (num * ((double)(j + 1)));
			expectedRestarts2 += (per * ((double)(j + 1)));
			if (j == 0) {
				firstBest = bestRank[j];
				firstRate = per;
			}
		}
		
		expectedRestarts /= deno;
		
		//System.out.println("---TestFirstBestRate---");
		System.out.println("----LocalOptima----");
		System.out.println("TestFirstBestRate " + firstBest + " / " + instances.size() + " = " + firstRate);
		System.out.println("TestExpectedRestart " + expectedRestarts);
		//System.out.println("TestExpectedRestart2 " + expectedRestarts2);
		//System.out.println("" + firstRate);
		System.out.println("-------------------");
	}
*/
	
	public static void exploreLocalOptimal(RandomStateGenerator randomGenr,
			List<HwInstance> instances,

			WeightVector cost_weight, 
			GreedySearcher gsearcher,
			int restarts,
			EInferencer einfer){
		
		
		int[] bestRank = new int[restarts];
		Arrays.fill(bestRank, 0);


		for (AbstractInstance ainst : instances) {
			
			SearchResult tmpRe = gsearcher.runSearchWithRestarts(cost_weight, einfer, restarts, ainst, null, false);
			checkRestartScores(tmpRe);
			
			int bestIdx = tmpRe.bestRank;
			bestRank[bestIdx]++;
		}
		
		printLocalOptimaMeasure(bestRank, instances.size(), restarts);

	}
	
	public static void printLocalOptimaMeasure(int[] bestRank, int totalInstances, int restarts) {
		int    firstBest = -1;
		double firstRate = -1;
		
		double expectedRestarts = 0;
		double expectedRestarts2 = 0;
		
		double accumExs = 0;
		double accumPer = 0;
		
		double deno = (double)totalInstances;
		for (int j = 0; j < restarts; j++) {
			double num = (double)bestRank[j];
			double per = num / deno;
			
			accumExs += num;
			accumPer = accumExs / deno;
			System.out.println(j + ": " + bestRank[j]+ " " + per + " " + accumPer);
			
			expectedRestarts += (num * ((double)(j + 1)));
			expectedRestarts2 += (per * ((double)(j + 1)));
			if (j == 0) {
				firstBest = bestRank[j];
				firstRate = per;
			}
		}
		
		expectedRestarts /= deno;
		
		System.out.println("----LocalOptima----");
		System.out.println("TestFirstBestRate " + firstBest + " / " + totalInstances + " = " + firstRate);
		System.out.println("TestExpectedRestart " + expectedRestarts);
		System.out.println("-------------------");
	}
	
	public static void checkRestartScores(SearchResult sResult) {
		
		assert (sResult.trajectories.size() == sResult.restartPredScores.size());
		
		double bestSc = -Double.MAX_VALUE;
		int myBestRk = -2;
		for (int i = 0 ; i < sResult.trajectories.size(); i++) {
			
			List<SearchState> states = sResult.trajectories.get(i).getStateList();
			double s1 = states.get(states.size() - 1).score;
			double s2 = sResult.restartPredScores.get(i);
			assert (s1 == s2);
			
			if (sResult.restartPredScores.get(i) > bestSc) {
				bestSc = sResult.restartPredScores.get(i);
				myBestRk = i;
			}
		}
		
		assert (sResult.bestRank == myBestRk);
		
	}
	
}
