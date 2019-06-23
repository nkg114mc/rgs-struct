package init;

import java.util.HashSet;
import java.util.List;
import java.util.Random;

import experiment.RndLocalSearchExperiment.InitType;
import general.AbstractInstance;
import general.AbstractOutput;
import multilabel.instance.Label;
import search.SearchState;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;

public class MultiLabelAllZeroGenerator extends RandomStateGenerator {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1597974283328828258L;
	private int labelSize;
	private int domainSize;
	private Random random;
	
	public MultiLabelAllZeroGenerator(int seqsz, int dmsz, Random rnd) {
		labelSize = seqsz;
		domainSize = dmsz;
		random = rnd;
		if (dmsz != 2) {
			throw new RuntimeException("Dimension size != " + dmsz);
		}
		System.out.println("Create multi-label logistic initializer.");
	}
	
	public MultiLabelAllZeroGenerator(int seqsz, int dmsz) {
		labelSize = seqsz;
		domainSize = dmsz;
		random = new Random();
		if (dmsz != 2) {
			throw new RuntimeException("Dimension size != " + dmsz);
		}
		System.out.println("Create multi-label logistic initializer.");
	}
	
	public HashSet<SearchState> generateRandomInitState(AbstractInstance inst, int stateNum) {
		HashSet<SearchState> genStates = new HashSet<SearchState>();
		for (int i = 0; i < stateNum; i++) {
			AbstractOutput rndout = sampleOneOutput();
			genStates.add(new SearchState(rndout));
		}
		return genStates;
	}
	
	public SearchState generateSingleRandomInitState(AbstractInstance inst) {
		HashSet<SearchState> sset = generateRandomInitState(inst,1);
		SearchState result = null;
		for (SearchState s : sset) {
			result = s;
			break;
		}
		return result;
	}
	
	public AbstractOutput sampleOneOutput() {
		
		HwOutput output = new HwOutput(labelSize, Label.MULTI_LABEL_DOMAIN);
		for (int i = 0; i < output.size(); i++) {
			int sampledIdx = 0;
			output.setOutput(i, sampledIdx);
		}
		
		return output;
	}

	@Override
	public InitType getType() {
		return InitType.ALLZERO_INIT;
	}

	@Override
	public void testPerformance(List<HwInstance> insts, String[] alphabet) {
		// do nothing
	}
	
}
