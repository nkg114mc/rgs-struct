package sequence.nettalk;

import general.AbstractLabelSet;

public class NtkStressLabelSet  extends AbstractLabelSet {
	
	private static final String[] STRESSTAGS = {"00","01","02","03","04"};

	public String[] getLabels() {
		return STRESSTAGS;
	}

}
