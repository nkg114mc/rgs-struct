package essvm;

import java.util.HashMap;
import java.util.HashSet;

import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import general.AbstractInstance;
import general.AbstractOutput;
import multilabel.MultiLabelFeaturizer;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;

public class MultiLabelFactorizedFeatureGenerator {

	MultiLabelFeaturizer mlfeaturizer;
	
	public MultiLabelFactorizedFeatureGenerator(MultiLabelFeaturizer mlfzr) {
		mlfeaturizer = mlfzr;
	}
	
	/////////////////////////////////////
	/////////////////////////////////////
	/////////////////////////////////////

	// use factor value diff to represent two feature vector diff
	// for multi-label only
	public MultiLabelFactorizedFeature getFactorizedFeatureVectorDiff(AbstractInstance xi, AbstractOutput yi1, AbstractOutput yi2) {
		
		HwInstance x = (HwInstance)xi; 
		HwOutput y1 = (HwOutput)yi1;
		HwOutput y2 = (HwOutput)yi2; 
		
		MultiLabelFactorizedFeature mlfacfeat = new MultiLabelFactorizedFeature();
		mlfacfeat.unaryValues = new HashMap<Integer, Float>();
		mlfacfeat.pairValues = new HashMap<Integer, Float>();
		

		// indexes that two y are different
		HashSet<Integer> diffIndexes = new HashSet<Integer>();
		
		// unary features
		for (int i = 0; i < x.size(); i++) {
			if (y1.getOutput(i) != y2.getOutput(i)) {
				if ( y1.getOutput(i)  != y2.getOutput(i) ) {
					diffIndexes.add(i);
					int sign = y1.getOutput(i) - y2.getOutput(i);
					mlfacfeat.unaryValues.put(i, (float)sign);
				}
			}
		}
		
		// pairwise features
		if (mlfeaturizer.considerPairs) {
			for (int i = 0; i < y1.size(); i++) {
				if (diffIndexes.contains(i)) {
					diffIndexes.remove(i); // remove from diff-set
				}
				
				if (y1.getOutput(i) == y2.getOutput(i)) { // i is the same
					for (Integer j2 : diffIndexes) {
						int j = j2.intValue();
						if (i != j) {
							if ( (y1.getOutput(j) != y2.getOutput(j)) ) {
								int idx11 = mlfeaturizer.getLabelPairFeatIndex(i, j, y1.getOutput(i), y1.getOutput(j));
								int idx22 = mlfeaturizer.getLabelPairFeatIndex(i, j, y2.getOutput(i), y2.getOutput(j));
								mlfacfeat.pairValues.put(idx11, 1.0f);
								mlfacfeat.pairValues.put(idx22, -1.0f);
							}
						}
					}
					
				} else { // i is different
					for (int j = (i + 1); j < y1.size(); j++) {
						if (i != j) {
							//if ( (y1.getOutput(j) != y2.getOutput(j)) ) {
								int idx11 = mlfeaturizer.getLabelPairFeatIndex(i, j, y1.getOutput(i), y1.getOutput(j));
								int idx22 = mlfeaturizer.getLabelPairFeatIndex(i, j, y2.getOutput(i), y2.getOutput(j));
								mlfacfeat.pairValues.put(idx11, 1.0f);
								mlfacfeat.pairValues.put(idx22, -1.0f);
							//}
						}
					}
					
				}
			}
		}

		
		return mlfacfeat;
		
	}
	
	public boolean getConsiderPair() {
		return mlfeaturizer.considerPairs;
	}
	
	public float scoringUnary(HashMap<Integer, Double> commonMlFeatMap, int yIdx, WeightVector w) {
	
		float rest = 0;
		for (Integer j2 : commonMlFeatMap.keySet()) {
			int idx = mlfeaturizer.getCompatibleFeatIndex(yIdx, j2.intValue());
			float v = commonMlFeatMap.get(j2).floatValue();
			rest += (w.get(idx) * v);
		}
		
		return rest;
	}

}
