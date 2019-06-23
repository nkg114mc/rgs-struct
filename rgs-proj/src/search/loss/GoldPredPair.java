package search.loss;

import general.AbstractInstance;
import general.AbstractOutput;

public class GoldPredPair {

	public AbstractInstance inst;
	public AbstractOutput gold;
	public AbstractOutput pred;
	
	public GoldPredPair(AbstractInstance ins, AbstractOutput g, AbstractOutput p) {
		inst = ins;
		gold = g;
		pred = p;
	}

}
