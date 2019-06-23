package multilabel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

import edu.berkeley.nlp.futile.fig.basic.Indexer;
import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.util.FeatureVectorBuffer;
import edu.illinois.cs.cogcomp.sl.util.IFeatureVector;
import general.AbstractFeaturizer;
import general.AbstractInstance;
import general.AbstractOutput;
import multilabel.instance.Example;
import multilabel.instance.OldWeightVector;
import multilabel.learning.StructOutput;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;

public class MultiLabelFeaturizer  extends AbstractFeaturizer {

	private static final long serialVersionUID = 2962044096338556470L;
	
	public int labelCnt = -1;
	public int singleFeatLen = -1;
	
	public boolean considerPairs = false;
	//public boolean considerTriplets = false;
	//public boolean considerQuadruples = false;
	public Indexer<String> featureIndexer = null;
	
	private int[][]     cachedCompitableFeatIndexs;
	private int[][][][] cachedPairwiseFeatIndexs;
	
	public MultiLabelFeaturizer(int nlabel, int sigflen, boolean pairwise, boolean trip, boolean quad) {
		labelCnt = nlabel;
		singleFeatLen = sigflen;
		considerPairs = pairwise;
		//considerTriplets = trip;
		//considerQuadruples = quad;
		System.out.println("Featurizer order 1 = " + singleFeatLen);
		System.out.println("Featurizer order 2 = " + considerPairs);
		initFeatIndexer();
	}
	
	private void initFeatIndexer() {
		
		featureIndexer = new Indexer<String>();
		
		
		cachedCompitableFeatIndexs = new int[labelCnt][singleFeatLen];
		
		// consistency feature
		for (int yi = 0; yi < labelCnt; yi++) {
			for (int xi = 0; xi < singleFeatLen; xi++) {
				String compitbleFeatName = getCompatibleFeatName(yi, xi);
				int idx = featureIndexer.getIndex(compitbleFeatName);
				cachedCompitableFeatIndexs[yi][xi] = idx;
			}
		}
		
		// pair feature
		if (considerPairs) {
			
			cachedPairwiseFeatIndexs = new int[labelCnt][labelCnt][2][2];
			
			for (int y1 = 0; y1 < labelCnt; y1++) {
				for (int y2 = 0; y2 < labelCnt; y2++) {
					if (y1 != y2) {
						int idx00 = featureIndexer.getIndex(getLabelPairFeatName(y1, y2, 0, 0));
						int idx01 = featureIndexer.getIndex(getLabelPairFeatName(y1, y2, 0, 1));
						int idx10 = featureIndexer.getIndex(getLabelPairFeatName(y1, y2, 1, 0));
						int idx11 = featureIndexer.getIndex(getLabelPairFeatName(y1, y2, 1, 1));
						
						cachedPairwiseFeatIndexs[y1][y2][0][0] = idx00;
						cachedPairwiseFeatIndexs[y1][y2][0][1] = idx01;
						cachedPairwiseFeatIndexs[y1][y2][1][0] = idx10;
						cachedPairwiseFeatIndexs[y1][y2][1][1] = idx11;
					}
				}
			}
		}
/*		
		// triple feature
		if (considerTriplets) {
			for (int y1 = 0; y1 < alphabet.length; y1++) {
				for (int y2 = 0; y2 < alphabet.length; y2++) {
					for (int y3 = 0; y3 < alphabet.length; y3++) {
						String tripleFeatName = getTenaryFeatName(alphabet, y1, y2, y3);
						featureIndexer.getIndex(tripleFeatName);
					}
				}
			}
		}
		
		// quad feature
		if (considerQuadruples) {
			for (int y1 = 0; y1 < alphabet.length; y1++) {
				for (int y2 = 0; y2 < alphabet.length; y2++) {
					for (int y3 = 0; y3 < alphabet.length; y3++) {
						for (int y4 = 0; y4 < alphabet.length; y4++) {
							String quadFeatName = getQuadFeatName(alphabet, y1, y2, y3, y4);
							featureIndexer.getIndex(quadFeatName);
						}
					}
				}
			}
		}
*/
		
		int oldLen = getFeatureDimension(singleFeatLen, labelCnt, considerPairs);
		if (oldLen != featureIndexer.size()) {
			throw new RuntimeException("Feature length inequal " + oldLen + " != " + featureIndexer.size());
		}
		
		System.out.println("Feature length = " + featureIndexer.size());
		
		
	}
	
	public String getCompatibleFeatName(int lbIdx, int singlefIdx) {
		return ("Comp:lb-" + String.valueOf(lbIdx)+ "-idx-" + String.valueOf(singlefIdx));
	}
	
	public String getLabelPairFeatName(int lb1Idx, int lb2Idx, int lb1Val, int lb2Val) {
		if (lb1Idx < lb2Idx) {
			return ("Pairw:lb1-" + String.valueOf(lb1Idx)+ "=" + String.valueOf(lb1Val) + "&" + "lb2-" + String.valueOf(lb2Idx)+"=" + String.valueOf(lb2Val));
		} else if (lb1Idx > lb2Idx) {
			return ("Pairw:lb1-" + String.valueOf(lb2Idx)+ "=" + String.valueOf(lb2Val) + "&" + "lb2-" + String.valueOf(lb1Idx)+"=" + String.valueOf(lb1Val));
		} else {
			throw new RuntimeException("lb1 = " + lb1Idx + " lb2 = " + lb2Idx);
		}
	}
	
	public int getCompatibleFeatIndexSlow(int lbIdx, int singlefIdx) {
		String unaryfn = getCompatibleFeatName(lbIdx, singlefIdx);
		int idx = getIndex(unaryfn);
		return (idx);
	}
	
	public int getLabelPairFeatIndexSlow(int lb1Idx, int lb2Idx, int lb1Val, int lb2Val) {
		String pairfn = getLabelPairFeatName(lb1Idx, lb2Idx, lb1Val, lb2Val);
		int idx2 = getIndex(pairfn);
		return idx2;
	}
	
	public int getCompatibleFeatIndex(int lbIdx, int singlefIdx) {
		return cachedCompitableFeatIndexs[lbIdx][singlefIdx];
	}
	
	public int getLabelPairFeatIndex(int lb1Idx, int lb2Idx, int lb1Val, int lb2Val) {
		return cachedPairwiseFeatIndexs[lb1Idx][lb2Idx][lb1Val][lb2Val];
	}

	@Override
	public IFeatureVector getFeatureVector(IInstance xi, IStructure yi) {
		FeatureVectorBuffer fv = new FeatureVectorBuffer();
		HwInstance x = (HwInstance) xi;
		HwOutput y = (HwOutput) yi;
		
		HashMap<Integer, Double> feats = featurize(x, y);
		for (int idx : feats.keySet()) {
			fv.addFeature(idx, feats.get(idx));			
		}
		return fv.toFeatureVector();
	}
	
	@Override
	public IFeatureVector getFeatureVectorDiff(IInstance xi, IStructure yi1, IStructure yi2) {
		FeatureVectorBuffer fv = new FeatureVectorBuffer();
		HwInstance x = (HwInstance) xi;
		HwOutput y1 = (HwOutput) yi1;
		HwOutput y2 = (HwOutput) yi2;
		
		HashMap<Integer, Double> featDiff = featurizeDiff(x, y1, y2);
		for (int idx : featDiff.keySet()) {
			fv.addFeature(idx, featDiff.get(idx));			
		}
		return fv.toFeatureVector();
	}

	
	public static HashMap<Integer, Double> arrToMap(double[] f) {
		HashMap<Integer, Double> feat = new HashMap<Integer, Double>();
		for (int i = 0; i < f.length; i++) {
			if (f[i] != 0) {
				feat.put(i, f[i]);
			}
		}
		return feat;
	}
	
	@Override
	public HashMap<Integer, Double> featurize(AbstractInstance xi, AbstractOutput yi) {
		
		HwInstance x = (HwInstance)xi; 
		HwOutput y = (HwOutput)yi; 
		
		
		HashMap<Integer, Double> sparseValues = new HashMap<Integer, Double>();
		
		// all unary multi-label feature are the same
		double[] commonMlFeat = x.getUnaryFeats(0);
		HashMap<Integer, Double> commonMlFeatMap = arrToMap(commonMlFeat);
		
		// unary features
		for (int i = 0; i < x.size(); i++) {
			double[] feat = commonMlFeat;//x.getUnaryFeats(i);
			if (y.getOutput(i) > 0) {
				/*
				for (int j = 0; j < feat.length; j++) {
					int idx = getCompatibleFeatIndex(i, j);
					if (feat[j] != 0) {
						sparseValues.put(idx, feat[j]);
					}
				} // else all 0
				*/
				for (Integer j2 : commonMlFeatMap.keySet()) {
					int idx = getCompatibleFeatIndex(i, j2.intValue());
					sparseValues.put(idx, feat[j2.intValue()]);
				}
			}
		}
		
		// pairwise features
		if (considerPairs) {
			for (int i = 0; i < y.size(); i++) {
				for (int j = (i + 1); j < y.size(); j++) {
					if (i != j) {
						int idx2 = getLabelPairFeatIndex(i, j, y.getOutput(i), y.getOutput(j));
						sparseValues.put(idx2, 1.0);
					}
				}
			}
		}
		/*
		if (considerPairs) {
				for (int i = 0; i < y.size(); i++) {
					for (int j = 0; j < y.size(); j++) {
						if (i != j) {
							int idx2 = getLabelPairFeatIndex(i, j, y.getOutput(i), y.getOutput(j));
							
							//chechIdx(idx2, pairfn);
							sparseValues.put(idx2, 1.0);
						}
					}
				}
		}
		*/
		
		
		
/*
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


		// genRelaventFeatures
		int pairCnt = 0;
		for (int k = 0; k < (nlabel - 1); k++) {
			for (int k2 = (k + 1); k2 < nlabel; k2++) {

				int computedIdx = computePairIndex(k, k2, nlabel);
				if (pairCnt != computedIdx) {
					throw new RuntimeException("Index inequal " + pairCnt + " != " + computedIdx);
				}
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
*/
		
		
		return sparseValues;
	}
	

	
	public HashMap<Integer, Double> featurizeDiff(AbstractInstance xi, AbstractOutput yi1, AbstractOutput yi2) {
		
		HwInstance x = (HwInstance)xi; 
		HwOutput y1 = (HwOutput)yi1;
		HwOutput y2 = (HwOutput)yi2; 
		
		
		HashMap<Integer, Double> sparseValues = new HashMap<Integer, Double>();
		
		// all unary multi-label feature are the same
		double[] commonMlFeat = x.getUnaryFeats(0);
		HashMap<Integer, Double> commonMlFeatMap = arrToMap(commonMlFeat);
		
		// indexes that two y are different
		HashSet<Integer> diffIndexes = new HashSet<Integer>();
		
		// unary features
		for (int i = 0; i < x.size(); i++) {
			double[] feat = commonMlFeat;//x.getUnaryFeats(i);
			if (y1.getOutput(i) != y2.getOutput(i)) {
				int sign = 1;
				if ((y1.getOutput(i) > 0) && (y2.getOutput(i) == 0)) { // y1[i] == 1 && y2[i] == 0
					sign = 1;
					diffIndexes.add(i);
				} else if ((y1.getOutput(i) == 0) && (y2.getOutput(i) > 0)) {// y1[i] == 0 && y2[i] == 1
					sign = -1;
					diffIndexes.add(i);
				}
				for (Integer j2 : commonMlFeatMap.keySet()) {
					int idx = getCompatibleFeatIndex(i, j2.intValue());
					double vlu = (double)(feat[j2.intValue()] * sign);
					sparseValues.put(idx, vlu);
				}
			}
		}
		
		// pairwise features
		if (considerPairs) {
			for (int i = 0; i < y1.size(); i++) {
				if (diffIndexes.contains(i)) {
					diffIndexes.remove(i); // remove from diff-set
				}
				
				if (y1.getOutput(i) == y2.getOutput(i)) { // i is the same
					for (Integer j2 : diffIndexes) {//int j = (i + 1); j < y1.size(); j++) {
						int j = j2.intValue();
						if (i != j) {
							if ( (y1.getOutput(j) != y2.getOutput(j)) ) {
								int idx11 = getLabelPairFeatIndex(i, j, y1.getOutput(i), y1.getOutput(j));
								int idx22 = getLabelPairFeatIndex(i, j, y2.getOutput(i), y2.getOutput(j));
								sparseValues.put(idx11, 1.0);
								sparseValues.put(idx22, -1.0);
							}
						}
					}
					
				} else { // i is different
					for (int j = (i + 1); j < y1.size(); j++) {
						if (i != j) {
							//if ( (y1.getOutput(j) != y2.getOutput(j)) ) {
								int idx11 = getLabelPairFeatIndex(i, j, y1.getOutput(i), y1.getOutput(j));
								int idx22 = getLabelPairFeatIndex(i, j, y2.getOutput(i), y2.getOutput(j));
								sparseValues.put(idx11, 1.0);
								sparseValues.put(idx22, -1.0);
							//}
						}
					}
					
				}
			}
		}
		
		/*
		if (considerPairs) {
			for (int i = 0; i < y1.size(); i++) {
				for (int j = (i + 1); j < y1.size(); j++) {
					if (i != j) {
						if ((y1.getOutput(i) != y2.getOutput(i)) || (y1.getOutput(j) != y2.getOutput(j)) ) { // y1[i] != y2[i]
							int idx11 = getLabelPairFeatIndex(i, j, y1.getOutput(i), y1.getOutput(j));
							int idx22 = getLabelPairFeatIndex(i, j, y2.getOutput(i), y2.getOutput(j));
							sparseValues.put(idx11, 1.0);
							sparseValues.put(idx22, -1.0);
						}
					}
				}
			}
		}*/
		
		// check correctness
		//HashMap<Integer, Double> compDiff = computeDiff(xi, yi1, yi2);
		//checkFvDiff(sparseValues, compDiff);
		
		
		return sparseValues;
	}
	
	private HashMap<Integer, Double> computeDiff(AbstractInstance xi, AbstractOutput yi1, AbstractOutput yi2) {
		HashMap<Integer, Double> fv1 = featurize(xi, yi1);
		HashMap<Integer, Double> fv2 = featurize(xi, yi2);
		
		HashSet<Integer> indexes = new HashSet<Integer>();
		indexes.addAll(fv1.keySet());
		indexes.addAll(fv2.keySet());
		
		HashMap<Integer, Double> fdiff = new HashMap<Integer, Double>();
		for (Integer id : indexes) {
			double v1 = fv1.getOrDefault(id.intValue(), 0.0);
			double v2 = fv2.getOrDefault(id.intValue(), 0.0);
			double df = v1 - v2;
			if (df != 0) {
				fdiff.put(id.intValue(), df);
			}
		}
		
		return fdiff;
	}
	
	private void checkFvDiff(HashMap<Integer, Double> diff, HashMap<Integer, Double> compdiff) {
		if (diff.size() != compdiff.size()) {
			throw new RuntimeException("Diff and Compdiff has different size: " + diff.size() +" != "+ compdiff.size());
		}
		HashSet<Integer> indexes = new HashSet<Integer>();
		indexes.addAll(diff.keySet());
		indexes.addAll(compdiff.keySet());
		for (Integer id : indexes) {
			Double v1 = diff.get(id.intValue());
			Double v2 = compdiff.get(id.intValue());
			if (v1 == null || v2 == null || v1.doubleValue() != v2.doubleValue()) {
				throw new RuntimeException("Diff and Compdiff has different value at "+id+": " + v1 +" != "+ v2);
			}
		}
	}
	
	
	private void chechIdx(int idx, String idxStr) {
		if (idx < 0) {
			throw new RuntimeException(idxStr + " idx = " + idx);
		}
	}
	
	public int getIndex(String featName) {
		return featureIndexer.indexOf(featName);
	}
	

	
	/////////////////////////////////////
	/////////////////////////////////////
	/////////////////////////////////////
	
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

	public static int getFeatureDimension(int featDim, int nlabel, boolean includePair) {
		int count = nlabel * featDim;// + 4 * (((nlabel - 1) * nlabel) / 2);
		if (includePair) {
			count += 4 * (((nlabel - 1) * nlabel) / 2);
		}
		return count;
	}

	@Override
	public int getFeatLen() {
		return featureIndexer.size();
	}
	

}
