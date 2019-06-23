package search;

import java.util.ArrayList;
import java.util.List;

public class SearchTrajectory {
	
	List<SearchState> trajStates;
	
	public SearchTrajectory() {
		trajStates = new ArrayList<SearchState>();
	}
	
	public void concatenateState(SearchState state) {
		trajStates.add(state.getCopy());
	}
	
	public List<SearchState> getStateList() {
		return trajStates;
	}

}
