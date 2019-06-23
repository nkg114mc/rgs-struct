package init;

import java.io.Serializable;
import java.util.HashSet;
import java.util.List;

import experiment.RndLocalSearchExperiment.InitType;
import general.AbstractInstance;
import search.SearchState;
import sequence.hw.HwInstance;

public abstract class RandomStateGenerator implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 3074693378512957539L;

	public abstract HashSet<SearchState> generateRandomInitState(AbstractInstance inst, int stateNum);
	
	public abstract SearchState generateSingleRandomInitState(AbstractInstance inst);
	
	public abstract InitType getType();
	
	public abstract void testPerformance(List<HwInstance> insts, String[] alphabet);
	
}