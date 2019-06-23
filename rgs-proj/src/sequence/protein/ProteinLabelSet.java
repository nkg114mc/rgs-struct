package sequence.protein;

import general.AbstractLabelSet;

public class ProteinLabelSet extends AbstractLabelSet {
	
	private static final String[] ALPHABET = {
			"0","1","2"
	};

	public String[] getLabels() {
		return ALPHABET;
	}
}
