package search;

import java.util.ArrayList;
import java.util.List;

import general.AbstractActionGenerator;
import general.AbstractInstance;

public class SeachActionGenerator extends AbstractActionGenerator {

	@Override
	public List<SearchAction> genAllAction(AbstractInstance instace, SearchState currState) {
		List<SearchAction> actions = new ArrayList<SearchAction>();

		for (int i = 0; i < currState.structOutput.size(); i++) {
			for (int j = 0; j < currState.structOutput.tagSize(); j++) {
				int oldv = currState.structOutput.getOutput(i);
				if (oldv != j) {
					SearchAction act = new SearchAction(i, j, oldv);
					actions.add(act);
				}
			}
		}

		return actions;
	}

}
