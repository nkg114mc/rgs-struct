package search;

import search.loss.LossScore;

public class SearchAction {
	
	int slotIndex;
	int newValue;
	int oldValue;
	//int actType;
	
	///
	public double score;
	public double acc;
	public LossScore accFrac;

	
	public int getSlotIdx() {
		return slotIndex;
	}
	
	public int getNewVal() {
		return newValue;
	}
	
	public int getOldVal() {
		return oldValue;
	}
	
	public SearchAction(int idx, int nv, int ov) {
		slotIndex = idx;
		newValue = nv;
		oldValue = ov;
	}

}
