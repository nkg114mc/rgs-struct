package multilabel.instance;

public class Label {
	public int originIndex; // this shows the label's name (usualy an index is enough)
	public int value;
	//public int truthValue; // unknown in testing
	
	public double rankScore;
	public boolean isPruned;
	
	public Label(int index, int initValue) {
		originIndex = index;
		value = initValue;
		
		rankScore = 0;
		isPruned = false;
	}
	
	public double getDoubleVal() {
		return ((double)value);
	}
	
	public int getValue() {
		return value;
	}
	
	public int getIndex() {
		return originIndex; 
	}
	
	public boolean isBinary() {
		if (value == 0 || value == 1) {
			return true;
		}
		return false;
	}
	
	public static final String[] MULTI_LABEL_DOMAIN = { "0", "1" };
}
