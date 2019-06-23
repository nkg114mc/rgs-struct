package multilabel.pruner;

import java.util.ArrayList;

import multilabel.instance.Example;
import multilabel.instance.OldWeightVector;

public class PrunerFeaturizer {
	
	
	public static void featurizeOneLabel(Example exmp, int i) {
		int nlabel = exmp.labelDim();
		int featDim = exmp.featDim();
		/*
		if (i) {
			
		}
		
		WeightVector wv = new WeightVector();
		
		for (int i = 0; i < nlabel; i++) {
			
			for (int j = 0; j < featDim; j++) {
				int idx = i * featDim + ;
				
				wv.put(, val);
			}
			
			
			
		}*/
	}
	
	public static OldWeightVector[] featurizeAll(Example exmp) {
		int nlabel = exmp.labelDim();
		int featDim = exmp.featDim();
		
		ArrayList<OldWeightVector> allwv = new ArrayList<OldWeightVector>();
		
		ArrayList<Double> feats = exmp.getFeat();
		for (int i = 0; i < nlabel; i++) {
			
			OldWeightVector wv = new OldWeightVector();
			wv.setMaxLength(nlabel * featDim);
			
			for (int j = 0; j < featDim; j++) {
				int idx = i * featDim + j;
				double val = feats.get(j);
				wv.put(idx, val);
			}
			
			allwv.add(wv);
		}
		
		return (allwv.toArray(new OldWeightVector[0]));
	}

}
