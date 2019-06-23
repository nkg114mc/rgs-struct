package search;

import general.AbstractOutput;
import search.loss.LossScore;

public class SearchState {
	
	public AbstractOutput structOutput;
	
	public float score; // -cost
	public double trueAcc; // true accuracy
	public LossScore trueAccFrac; // the 
	
	public SearchState(AbstractOutput stropt) {
		structOutput = stropt;
	}
	
	public SearchState getCopy() {
		AbstractOutput newOtpt = structOutput.copyFrom(structOutput);
		SearchState newState = new SearchState(newOtpt);
		newState.score = score;
		newState.trueAcc = trueAcc;
		newState.trueAccFrac = trueAccFrac;
		return newState;
	}
	
	public int hashCode() {
		return structOutput.hashCode();
	}
	
	public void doAction(SearchAction action) { // one original state
		action.oldValue = structOutput.getOutput(action.slotIndex);
		structOutput.setOutput(action.slotIndex, action.newValue);
	}
	
	public void undoAction(SearchAction action) {
		structOutput.setOutput(action.slotIndex, action.oldValue);
	}
	
	public SearchState doActionNewState(SearchAction act) {
		SearchState newstate = this.getCopy();
		newstate.doAction(act);
		return newstate;
	}
	
	public boolean isEqualOutput(SearchState anotherState) {
		boolean isequal = true;
		AbstractOutput anotherOutput = anotherState.structOutput;
		if (anotherOutput.size() != structOutput.size()) {
			throw new RuntimeException("Not a equvalent length: " + anotherOutput.size() + "!=" + structOutput.size());
		}
		for (int i = 0; i < structOutput.size(); i++) {
			if (anotherOutput.getOutput(i) != structOutput.getOutput(i)) {
				isequal = false;
				break;
			}
		}
		return isequal;
	}
}
/*
  // Note: This copy does not copy the feature vector!
	def copyState(src: SearchState) : SearchState = {

		val outputCopy = new Array[Int](src.size()); 
		Array.copy(src.output, 0, outputCopy , 0, src.size());

		val newState = new SearchState(outputCopy);
		newState.cachedPredScore = src.cachedPredScore;
		newState.cachedTrueLoss = src.cachedTrueLoss;
		newState.cachedFeatureVector = null;//src.cachedFeatureVector;

		val muskCopy = new Array[Int](src.slotMusk.length);
		Array.copy(src.slotMusk, 0, muskCopy, 0, src.slotMusk.length);
		newState.slotMusk = muskCopy;

		newState;
	}
*/
