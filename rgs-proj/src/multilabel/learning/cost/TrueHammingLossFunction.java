package multilabel.learning.cost;

import multilabel.evaluation.LossFunction;
import multilabel.learning.StructOutput;
import multilabel.learning.search.OldSearchState;
import multilabel.instance.Example;

/**
 * This cost function is using ground truth accuracy as cost value,
 * Only used in training, or evaluating generation loss, etc
 * 
 * DO NOT USE IN TESTING!
 * 
 * @author machao
 *
 */
public class TrueHammingLossFunction extends CostFunction {

	public TrueHammingLossFunction() {

	}
	
	public double getCost(OldSearchState state, Example ex) {
		StructOutput gold = ex.getGroundTruthOutput();
		double sc = LossFunction.computeHammingAccuracy(state.getOutput(), gold);
		return sc;
	}

}
