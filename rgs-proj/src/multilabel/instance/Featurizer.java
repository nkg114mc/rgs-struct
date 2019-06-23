package multilabel.instance;

import java.util.ArrayList;
import java.util.Arrays;

import multilabel.learning.StructOutput;

public class Featurizer {
	
	// return a for bit indicator vector
	public int[] labelPairFeature(int lablel, int labelr) {
		int[] result = new int[4];
		Arrays.fill(result, 0);
		
		// set indicator
		if ((lablel == 0) && (labelr == 0)) {
			result[0] = 1;
		} else if ((lablel == 0) && (labelr == 1)) {
			result[1] = 1;			
		} else if ((lablel == 1) && (labelr == 0)) {
			result[2] = 1;			
		} else if ((lablel == 1) && (labelr == 1)) {
			result[3] = 1;			
		}
		
		return result;
	}
	
	// 0,0 -> 0
	// 0,1 -> 1
	// 1,0 -> 2
	// 1,1 -> 3
	public static int getIndicatorIndex(int i, int j) {
		return (i * 2 + j);
	}
	
	public static int computePairIndex(int i, int j, int nlabel) {
		int m = nlabel - 1;
		int idx = ((m + (m - i + 1)) * i ) / 2 + (j - i - 1);
		return idx;
	}
	
	public OldWeightVector getFeatureVector(Example ex, StructOutput structOut) {
		
		assert (ex.labelDim() == structOut.size());
		
		OldWeightVector fv = new OldWeightVector();
		
		int nlabel = ex.labelDim();
		int featDim = ex.featDim();
		ArrayList<Double> feats = ex.getFeat();
		int totalCnt = 0;
		
		// genCompatibilityFeatures (Unary)
		for (int i = 0; i < nlabel; i++) {
			totalCnt += feats.size();
			if (structOut.getValue(i) > 0) { // non-zero
				for (int j = 0; j < featDim; j++) {
					int idx = i * featDim + j;
					double val = feats.get(j);
					fv.put(idx, val);
				}
			}
		}
		int unaryCnt = totalCnt;

		
		// genRelaventFeatures
		int pairCnt = 0;
		for (int k = 0; k < (nlabel - 1); k++) {
			for (int k2 = (k + 1); k2 < nlabel; k2++) {
				
				int computedIdx = computePairIndex(k, k2, nlabel);
				if (pairCnt != computedIdx) {
					throw new RuntimeException("Index inequal " + pairCnt + " != " + computedIdx);
				}/* else {
					System.out.println("Index equal " + pairCnt + " == " + computedIdx);
				}*/
				
				pairCnt++;
				int ll = structOut.getValue(k);
				int lr = structOut.getValue(k2);
				int[] indicator = labelPairFeature(ll, lr);
				totalCnt += indicator.length;
				
				for (int j = 0; j < indicator.length; j++) {
					int idx = unaryCnt + (pairCnt - 1) * indicator.length + j;
					//if (indicator[j] > 0) {
						fv.put(idx, indicator[j]);
					//}
				}
			}
		}
		
		int count = nlabel * featDim + 4 * (((nlabel - 1) * nlabel) / 2);
		if (count != totalCnt) {
			throw new RuntimeException(count + " != " + totalCnt);
		}
		
		fv.setMaxLength(totalCnt);
		//System.out.println(fv.maxLength);

		return (fv);		
	}
	
	public static int getFeatureDimension(int featDim, int nlabel) {
		int count = nlabel * featDim + 4 * (((nlabel - 1) * nlabel) / 2);
		return count;
	}

}
