package elearning.einfer;

import java.util.HashMap;

import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import general.AbstractInstance;
import search.GreedySearcher;

public class LinearSearchStateScorer extends SearchStateScoringFunction {
	
	WeightVector weight;
	
	public LinearSearchStateScorer(WeightVector wght) {
		weight = wght;
	}

	@Override
	public double getScoring(AbstractInstance ainst, HashMap<Integer, Double> phi) {
		double dot_prod = GreedySearcher.myDotProduct(phi, weight);
		return dot_prod;
	}

	@Override
	public Object getModel() {
		return weight;
	}

}
