package multilabel.learning.cost;

import java.util.Random;

import multilabel.learning.StructOutput;
import multilabel.learning.search.OldSearchState;

/**
 * For cost function unit test only
 * @author machao
 *
 */
public class RandomCostFunction extends CostFunction {
	
	double[] rndSc = new double[10000];
	
	public RandomCostFunction() {
		Random rnd = new Random();
		for (int i = 0; i < rndSc.length; i++) {
			rndSc[i] = rnd.nextDouble();
		}
	}
	
	
	public void loadModel() {
		
	}
	
	public double getCost(OldSearchState state) {
		double sc = 0;
		StructOutput output = state.getOutput();
		for (int i = 0; i < output.size(); i++) {
			if (output.getValue(i) > 0) {
				sc += rndSc[i];
			}
		}
		return sc;
	}
	
	public double getCostFast() {
		return 0;
	}

}
