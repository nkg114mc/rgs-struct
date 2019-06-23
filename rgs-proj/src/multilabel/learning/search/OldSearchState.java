package multilabel.learning.search;

import java.util.ArrayList;

import multilabel.learning.StructOutput;
import multilabel.instance.Label;
import multilabel.instance.OldWeightVector;

public class OldSearchState {
	
	private StructOutput output = null;
	
	public double predScore = -Double.MAX_VALUE;
	public double trueAccuracy = - Double.MAX_VALUE;
	
	public OldWeightVector fv = null;
	
	
	public OldSearchState(int outputSize) {
		output = new StructOutput(outputSize);
	}
	
	public StructOutput getOutput() {
		return output;
	}
	
	public OldSearchState(StructOutput out) {
		output = out;
	}
	
	public int size() {
		return output.size();
	}
	
	
	public OldSearchState getSelfCopy() {
		StructOutput outputNew = new StructOutput(output.size());
		StructOutput.copyOutput(output, outputNew);
		
		OldSearchState stateCopy = new OldSearchState(outputNew);
		stateCopy.predScore = this.predScore;
		stateCopy.trueAccuracy = this.trueAccuracy;
		return stateCopy;
	}
	
	public static OldSearchState getAllZeroState(int size) {
		StructOutput output = new StructOutput(size);
		for (int i = 0; i < size; i++) {
			output.setValue(i, 0);
		}
		return (new OldSearchState(output));
	}
	
	public static OldSearchState getAllOneState(int size) {
		StructOutput output = new StructOutput(size);
		for (int i = 0; i < size; i++) {
			output.setValue(i, 1);
		}
		return (new OldSearchState(output));
	}
	
	public void print() {
		System.out.println("Output: " + output.toString());
		System.out.println("PredScore: " + predScore);
		System.out.println("TrueScore: " + trueAccuracy);
		
	}
}
