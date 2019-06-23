package init;

import java.util.HashSet;
import java.util.List;
import java.util.Random;

import experiment.RndLocalSearchExperiment.InitType;
import general.AbstractInstance;
import search.SearchState;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;

public class UniformRndGenerator extends RandomStateGenerator {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8620446125579424391L;
	protected Random random;

	public UniformRndGenerator(Random rnd) {
		random = rnd;
		System.out.println("Create uniform distribution initializer.");
	}
	
	public HashSet<SearchState> generateRandomInitState(AbstractInstance inst, int stateNum) {
		
		HwInstance x = (HwInstance) inst;
		HashSet<SearchState> rndSet = new HashSet<SearchState>();
		
		// repeat
		int cnt = 0;
		while (true) {
			
			// get a uniform state
			HwOutput output = new HwOutput(inst.size(), x.alphabet);
			for (int j = 0; j < output.size(); j++) {
				output.output[j] =  getValueIndexUniformly(inst.domainSize(), random);// purely uniform
			}
			
			SearchState s = new SearchState(output);
			if (!rndSet.contains(s)) {
				rndSet.add(s);
				//System.out.println(s.structOutput.toString());
				cnt++;
			}
			
			if (cnt >= stateNum) {
				break;
			}
		}
		
		//System.out.println("Generate " + rndSet.size() + " initial states.");
		
		//System.out.println("done init...");
		return rndSet;
	}
	
	public static int getValueIndexUniformly(int domainSz, Random rnd) {
		return (rnd.nextInt(domainSz));
	}

	@Override
	public SearchState generateSingleRandomInitState(AbstractInstance inst) {
		HashSet<SearchState> sset = generateRandomInitState(inst,1);
		SearchState result = null;
		for (SearchState s : sset) {
			result = s;
			break;
		}
		return result;
	}
	
	@Override
	public InitType getType() {
		return InitType.UNIFORM_INIT;
	}

	@Override
	public void testPerformance(List<HwInstance> insts, String[] alphabet) {
		throw new RuntimeException("Can not test for uniform model...");
		
	}
	

}