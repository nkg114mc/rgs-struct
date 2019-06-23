package general;

import java.util.List;

import search.SearchAction;
import search.SearchState;

public abstract class AbstractActionGenerator {

	public abstract List<SearchAction> genAllAction(AbstractInstance instace, SearchState currState);
}
