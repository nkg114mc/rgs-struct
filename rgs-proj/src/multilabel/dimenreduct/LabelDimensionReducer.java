package multilabel.dimenreduct;

import java.util.ArrayList;

import multilabel.learning.StructOutput;
import multilabel.data.DatasetReader;
import multilabel.instance.Example;
import multilabel.instance.Label;

public class LabelDimensionReducer  {
	
	public ArrayList<Label> encodeExample(Example ex) {
		return null;
	}

	public StructOutput decodeOutput(Example ex, StructOutput lowResult) {
		return null;
	}
	
	public static void doDimensionReduction(ArrayList<Example> exs, LabelDimensionReducer lreducer) {
		for (int i = 0; i < exs.size(); i++) {
			Example ex = exs.get(i);
			ex.encodeToLowDimension(lreducer);
			ex.predictOutput = null;//ex.getGroundTruthOutput();
		}
		
		System.out.println("Low dimension = " + exs.get(0).labelDim());
		DatasetReader.countSparsity(exs);
	}
	
	public static void doDrReconstruct(ArrayList<Example> exs, LabelDimensionReducer lreducer) {
		// reconstruct
		for (int i = 0; i < exs.size(); i++) {
			Example ex = exs.get(i);
			StructOutput lowResult = ex.predictOutput;
			ex.predictOutput = lreducer.decodeOutput(ex, lowResult);
			// recover example labels
			ex.decodeToFullDimension(lreducer);
		}
		
		System.out.println("Full dimension = " + exs.get(0).labelDim());
	}



}
