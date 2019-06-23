package berkeleyentity.randsearch;

import java.util.HashSet;
import java.util.Random;

import general.AbstractInstance;
import init.UniformRndGenerator;
import search.SearchState;
import sequence.hw.HwOutput;

public class AceCorefSamplingRndGenerator extends UniformRndGenerator {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1270283446498442230L;

	boolean useGold;
	
	public AceCorefSamplingRndGenerator(Random rnd, boolean gd) {
		super(rnd);
		useGold = gd;
	}

	public HashSet<SearchState> generateRandomInitState(AbstractInstance inst, int stateNum) {
		
		AceCorefInstance x = (AceCorefInstance) inst;
		HashSet<SearchState> rndSet = new HashSet<SearchState>();
		
		// repeat
		int cnt = 0;
		while (true) {
			
			// get a uniform state
			HwOutput output = new HwOutput(inst.size(), x.alphabet);
			output.output[0] = 0;
			for (int j = 1; j < output.size(); j++) {
				int[] domain = x.getDomainGivenIndex(j);
				int vidx = getValueIndexUniformly(domain.length, random);// purely uniform
				output.output[j] = domain[vidx];
			}
			
			SearchState s = new SearchState(output);
			if (!rndSet.contains(s)) {
				rndSet.add(s);
				cnt++;
			}
			
			if (cnt >= stateNum) {
				break;
			}
		}
		
		return rndSet;
	}
}