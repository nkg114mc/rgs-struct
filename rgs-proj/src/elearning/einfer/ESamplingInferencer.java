package elearning.einfer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearning.EInferencer;
import general.AbstractFeaturizer;
import general.AbstractInstance;
import general.AbstractOutput;
import init.RandomStateGenerator;
import search.GreedySearcher;
import search.SearchState;

public class ESamplingInferencer extends EInferencer {
	
	RandomStateGenerator randomGenr;
	
	WeightVector cost_weight;
	AbstractFeaturizer cfeaturizer;
	
	SearchStateScoringFunction stateScorer;
	AbstractFeaturizer efeaturizer;
	int eSampling;
	
	public ESamplingInferencer(RandomStateGenerator initGener,
			
			WeightVector cwght,
			AbstractFeaturizer cfeatr,
			
			SearchStateScoringFunction md,
			AbstractFeaturizer efeatr,
			int ecnt) {
		
		randomGenr = initGener;
		
		cost_weight = cwght;
		cfeaturizer = cfeatr;
		
		stateScorer = md;
		efeaturizer = efeatr;
		eSampling = ecnt;
		
	}
	
	public class StateSortEntry {
		public SearchState state;
		public double score;
	}
	
	public static class StateSortEntryCmp implements Comparator<StateSortEntry>  {
		@Override
		public int compare(StateSortEntry o1, StateSortEntry o2) {
			if (o1.score > o2.score) {
				return -1;
			} else if (o1.score < o2.score) {
				return 1;
			}
			return 0;
		}
	}

	@Override
	public SearchState generateOneInitState(AbstractInstance input, AbstractOutput gold, SearchState originalInit) {
		
		//SearchState y_end_state = yEndSampling(input, eSampling);
		//return y_end_state;
		
		return originalInit;
	}

	

	@Override
	public List<SearchState> generateMultiInitStates(AbstractInstance inst, AbstractOutput gold, List<SearchState> originalInitStates) {
		
		ArrayList<StateSortEntry> ilist = new ArrayList<StateSortEntry>();
		for (SearchState s : originalInitStates) {
			
			HashMap<Integer, Double> phie = efeaturizer.featurize(inst, s.structOutput);
			double e_val = stateScorer.getScoring(inst, phie);
			
			StateSortEntry sentry = new StateSortEntry();
			sentry.state = s; sentry.score = e_val;
			ilist.add(sentry);
		}
		
		Collections.sort(ilist, new StateSortEntryCmp());
		
		ArrayList<SearchState> jstates = new ArrayList<SearchState>();
		for (int k = 0; k < ilist.size(); k++) {
			jstates.add(ilist.get(k).state);
		}
		
		return jstates;
	}
	

	
	////////////////////////////////////
	////////////////////////////////////
	////////////////////////////////////
	
	
	
	public SearchState yEndSampling(AbstractInstance ainst, int eSamplingCnt) {
		
		SearchState bestYEnd = null;
		double bestEval = -Double.MAX_VALUE;
		
		HashSet<SearchState> jstates = randomGenr.generateRandomInitState(ainst, eSamplingCnt);
		//HashSet<SearchState> jstates = pickTopInits(ainst, eSamplingCnt);
		for (SearchState initState : jstates) {

			HashMap<Integer, Double> phi_e = efeaturizer.featurize(ainst, initState.structOutput);
			double eval_value = stateScorer.getScoring(ainst, phi_e);

			if (eval_value > bestEval) {
				bestEval = eval_value;
				bestYEnd = initState;
			}
		}
		
		assert(bestYEnd != null);
		return bestYEnd;
	}
	
	public HashSet<SearchState> pickTopInits(AbstractInstance ainst, int eSamplingCnt) {
		HashSet<SearchState> istates = randomGenr.generateRandomInitState(ainst, eSamplingCnt * 10);
		
		ArrayList<StateSortEntry> ilist = new ArrayList<StateSortEntry>();
		for (SearchState s : istates) {
			
			HashMap<Integer, Double> phic = cfeaturizer.featurize(ainst, s.structOutput);
			double predSc = GreedySearcher.myDotProduct(phic, cost_weight);
			
			StateSortEntry sentry = new StateSortEntry();
			sentry.state = s; sentry.score = predSc;
			ilist.add(sentry);
		}
		
		Collections.sort(ilist, new StateSortEntryCmp());
		
		HashSet<SearchState> jstates = new HashSet<SearchState>();
		for (int k = 0; k < eSamplingCnt; k++) {
			jstates.add(ilist.get(k).state);
		}
		return jstates;
	}



	@Override
	public void setEvalScoringFunc(SearchStateScoringFunction efunc) {
		stateScorer = efunc;
	}





}
