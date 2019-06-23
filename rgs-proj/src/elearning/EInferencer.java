package elearning;

import java.util.List;
import elearning.einfer.SearchStateScoringFunction;
import general.AbstractInstance;
import general.AbstractOutput;
import search.SearchState;

public abstract class EInferencer {
	
	public abstract SearchState generateOneInitState(AbstractInstance inst, AbstractOutput gold, SearchState originalInit);
	
	public abstract List<SearchState> generateMultiInitStates(AbstractInstance inst, AbstractOutput gold, List<SearchState> originalInitStates);// SearchState originalInit);
	
	public abstract void setEvalScoringFunc(SearchStateScoringFunction efunc);

}
