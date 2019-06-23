package multilabel.learning.heuristic;

import multilabel.instance.Example;
import multilabel.learning.search.OldSearchState;

/**
 * Base class of all cost functions
 * @author machao
 *
 */
public class HeuristicFunction {

	public double getCost(OldSearchState newState, Example exmp) {
		return -999;
	}

}
