package elearning.einfer;

import java.util.List;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearning.EInferencer;
import general.AbstractFeaturizer;
import general.AbstractInstance;
import general.AbstractOutput;
import search.GreedySearcher;
import search.SearchResult;
import search.SearchState;


public class ELinearSearchInferencer extends EInferencer {
	
	public GreedySearcher esearcher;
	public WeightVector e_weight;
	
	
	public AbstractFeaturizer efeaturizer;
	public boolean considerInstWeight;
	
	public ELinearSearchInferencer(GreedySearcher esr, WeightVector e_w) {
		esearcher = esr;
		e_weight = e_w;
	}

	@Override
	public SearchState generateOneInitState(AbstractInstance x, AbstractOutput gold, SearchState originalInit) {
		
		// gold state
		SearchState goldState = new SearchState(gold);
		
		SearchResult e_result = esearcher.doHillClimbing(e_weight, x, originalInit, goldState, Integer.MAX_VALUE, false);
		SearchState y_end_state = e_result.predState; // this is just y_end
		return y_end_state;
	}

	@Override
	public List<SearchState> generateMultiInitStates(AbstractInstance inst, AbstractOutput gold, List<SearchState> originalInitStates) {
		return originalInitStates;
	}

	@Override
	public void setEvalScoringFunc(SearchStateScoringFunction efunc) {
		// TODO Auto-generated method stub
	}

}
