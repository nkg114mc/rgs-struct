package elearning.einfer;

import java.util.HashMap;

import general.AbstractInstance;
import general.AbstractOutput;

public abstract class SearchStateScoringFunction {
	
	public abstract double getScoring(AbstractInstance inst, HashMap<Integer, Double> feature);
	
	public abstract Object getModel();
	
	//public abstract double getScoreOneExample(AbstractInstance inst, AbstractOutput output);
	
}
