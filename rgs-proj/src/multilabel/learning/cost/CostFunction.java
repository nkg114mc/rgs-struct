package multilabel.learning.cost;

import multilabel.pruner.UMassRankLib;
import multilabel.instance.Example;
import multilabel.learning.search.OldSearchState;

/**
 * Base class of all cost functions
 * @author machao
 *
 */
public class CostFunction {

	public void loadModel(String modelFileName) {
	}
	
	public double getCost(OldSearchState newState, Example exmp) {
		return -999;
	}

}
