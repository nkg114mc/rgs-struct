package general;

import search.SearchAction;
import sequence.hw.HwOutput;

public abstract class AbstractFactorGraph {
	
	public abstract void updateScoreTable(double[] weights);
	
	public abstract double computeScoreWithTable(double[] weights, HwOutput output);

	public abstract double computeScoreDiffWithTable(double[] weights, SearchAction action, HwOutput output);

	public abstract double computeScore(double[] weights, HwOutput output);
	public abstract double getCachedScore();

}
