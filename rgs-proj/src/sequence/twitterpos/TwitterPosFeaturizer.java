package sequence.twitterpos;

import java.util.Arrays;
import java.util.HashMap;

import edu.berkeley.nlp.futile.fig.basic.Indexer;
import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.util.FeatureVectorBuffer;
import edu.illinois.cs.cogcomp.sl.util.IFeatureVector;
import general.AbstractFeaturizer;
import general.AbstractInstance;
import general.AbstractOutput;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;

public class TwitterPosFeaturizer extends AbstractFeaturizer { 
	
	private static final long serialVersionUID = 2600327382486754166L;
	
	public static final int LSTMStateLen = 200;
	
	public String[] alphabet = null;
	public int singleFeatLen = -1;
	
	public boolean considerPairs = false;
	public boolean considerTriplets = false;
	public boolean considerQuadruples = false;
	public Indexer<String> featureIndexer = null;
	
	private int[][]     unaryIndex;
	private int[][]     pairIndex;
	private int[][][]   tenaryIndex;
	private int[][][][] quadIndex;
	
	public TwitterPosFeaturizer(String[] albt, int sigflen, boolean pairwise, boolean trip, boolean quad) {
		alphabet = albt;
		singleFeatLen = sigflen;
		considerPairs = pairwise;
		considerTriplets = trip;
		considerQuadruples = quad;
		System.out.println("Featurizer order 1 = " + singleFeatLen);
		System.out.println("Featurizer order 2 = " + considerPairs);
		System.out.println("Featurizer order 3 = " + considerTriplets);
		System.out.println("Featurizer order 4 = " + considerQuadruples);
		initFeatIndexer();
	}
	public static String getLabelName(String[] albt, int yIdx) {
		return albt[yIdx];
	}
	
	public static String getUnaryFeatName(String[] albt, int xIdx, int yIdx) {
		String compitbleFeatName = "Unary:" + getLabelName(albt, yIdx) + "-" + "Dim" + xIdx;
		return compitbleFeatName;
	}
	
	public static String getPairwiseFeatName(String[] albt, int y1Idx, int y2Idx) {
		String g2FeatName = "Pairwise:" + getLabelName(albt, y1Idx) + "-" + getLabelName(albt, y2Idx);
		return g2FeatName;
		
	}
	
	public static String getTenaryFeatName(String[] albt, int y1Idx, int y2Idx, int y3Idx) {
		String g3FeatName = "Triple:" + getLabelName(albt, y1Idx) + "-" + getLabelName(albt, y2Idx) + "-" + getLabelName(albt, y3Idx);
		return g3FeatName;
		
	}
	
	public static String getQuadFeatName(String[] albt, int y1Idx, int y2Idx, int y3Idx, int y4Idx) {
		String g4FeatName = "Quad:" + getLabelName(albt, y1Idx) + "-" + getLabelName(albt, y2Idx) + "-" + getLabelName(albt, y3Idx) + "-" + getLabelName(albt, y4Idx);
		return g4FeatName;
		
	}
	
	/////////////////////
	
	public int getUnaryIndex(int xIdx, int yIdx) {
		return unaryIndex[xIdx][yIdx];
	}
	
	public int getPairIndex(int y1Idx, int y2Idx) {
		return pairIndex[y1Idx][y2Idx];
	}
	
	public int getTenaryIndex(int y1Idx, int y2Idx, int y3Idx) {
		return tenaryIndex[y1Idx][y2Idx][y3Idx];
	}
	
	public int getQuadIndex(int y1Idx, int y2Idx, int y3Idx, int y4Idx) {
		return quadIndex[y1Idx][y2Idx][y3Idx][y4Idx];
	}
	
	
	
	private void initFeatIndexer() {
		
		featureIndexer = new Indexer<String>();
		
		// consistency feature
		unaryIndex = new int[singleFeatLen][alphabet.length];
		for (int yi = 0; yi < alphabet.length; yi++) {
			for (int xi = 0; xi < singleFeatLen; xi++) {
				String compitbleFeatName = getUnaryFeatName(alphabet, xi, yi);
				int idx = featureIndexer.getIndex(compitbleFeatName);
				unaryIndex[xi][yi] = idx;
			}
		}
		
		// pair feature
		if (considerPairs) {
			pairIndex = new int[alphabet.length][alphabet.length];
			for (int y1 = 0; y1 < alphabet.length; y1++) {
				for (int y2 = 0; y2 < alphabet.length; y2++) {
					String pairFeatName = getPairwiseFeatName(alphabet, y1, y2);
					int idx = featureIndexer.getIndex(pairFeatName);
					pairIndex[y1][y2] = idx;
				}
			}
		}
		
		// triple feature
		if (considerTriplets) {
			tenaryIndex = new int[alphabet.length][alphabet.length][alphabet.length];
			for (int y1 = 0; y1 < alphabet.length; y1++) {
				for (int y2 = 0; y2 < alphabet.length; y2++) {
					for (int y3 = 0; y3 < alphabet.length; y3++) {
						String tripleFeatName = getTenaryFeatName(alphabet, y1, y2, y3);
						int idx = featureIndexer.getIndex(tripleFeatName);
						tenaryIndex[y1][y2][y3] = idx;
					}
				}
			}
		}
		
		// quad feature
		if (considerQuadruples) {
			quadIndex = new int[alphabet.length][alphabet.length][alphabet.length][alphabet.length];;
			for (int y1 = 0; y1 < alphabet.length; y1++) {
				for (int y2 = 0; y2 < alphabet.length; y2++) {
					for (int y3 = 0; y3 < alphabet.length; y3++) {
						for (int y4 = 0; y4 < alphabet.length; y4++) {
							String quadFeatName = getQuadFeatName(alphabet, y1, y2, y3, y4);
							int idx = featureIndexer.getIndex(quadFeatName);
							quadIndex[y1][y2][y3][y4] = idx;
						}
					}
				}
			}
		}
		
		System.out.println("Feature length = " + featureIndexer.size());
		
	}

	public int getFeatLen() {
		return featureIndexer.size();
	}

	/**
	 * This function returns a feature vector \Phi(x,y) based on an instance-structure pair.
	 * 
	 * @return Feature Vector \Phi(x,y), where x is the input instance and y is the
	 *         output structure
	 */

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
	
	
	public int getIndex(String featName) {
		return featureIndexer.indexOf(featName);
	}
	
	// phi(x,y)
	@Override
	public HashMap<Integer, Double> featurize(AbstractInstance xi, AbstractOutput yi) {
		
		HwInstance x = (HwInstance)xi; 
		HwOutput y = (HwOutput)yi; 
		
		//int unaryLen = 0;
		//int pairLen = 0;
		//int tripLen = 0;
		//int quadLen = 0;
		HashMap<Integer, Double> sparseValues = new HashMap<Integer, Double>();
		

		// unary features
		double[][] unaryFeats = new double[alphabet.length][];
		for (int j = 0; j < alphabet.length; j++) {
			unaryFeats[j] = new double[singleFeatLen];
			Arrays.fill(unaryFeats[j], 0);
		}

		int[] tags = y.output;
		int len = tags.length;
		for (int i = 0; i < len; i++) {
			double[] ft = (x.letterSegs.get(i).getFeatArr());
			int value = tags[i];
			addVector(unaryFeats[value], ft);
		}
		
		//int beginIdx = 0;
		for (int i = 0; i < unaryFeats.length; i++) {
			for (int j = 0; j < unaryFeats[i].length; j++) {
				if (unaryFeats[i][j] > 0) {
					//String unaryfn = getUnaryFeatName(alphabet, j, i);
					//int idx = getIndex(unaryfn) ;
					int idx = getUnaryIndex(j, i);
					sparseValues.put(idx, unaryFeats[i][j]);
				}
			}
		}
		
		// pairwise features
		if (considerPairs) {
			if (x.size() >= 2) { // at least two slots to consider pair
				for (int i = 0; i < (len - 1); i++) {
					//String pairfn = getPairwiseFeatName(alphabet, tags[i], tags[i + 1]);
					//int idx2 = getIndex(pairfn) ;
					int idx2 = getPairIndex(tags[i], tags[i + 1]);
					increaseMap(sparseValues, idx2, 1);
				}
			}
		}
		
		
		// triplet features
		if (considerTriplets) {
			if (x.size() >= 3) { // at least two slots to consider pair
				for (int i = 0; i < (len - 2); i++) {
					//String triplefn = getTenaryFeatName(alphabet, tags[i], tags[i + 1], tags[i + 2]);
					//int idx3 = getIndex(triplefn) ;
					int idx3 = getTenaryIndex(tags[i], tags[i + 1], tags[i + 2]);
					increaseMap(sparseValues, idx3, 1);
				}
			}
		}
		
		// quad features
		if (considerQuadruples) {
			if (x.size() >= 4) { // at least two slots to consider pair
				for (int i = 0; i < (len - 3); i++) {
					//String quadfn = getQuadFeatName(alphabet, tags[i], tags[i + 1], tags[i + 2], tags[i + 3]);
					//int idx4 = getIndex(quadfn) ;
					int idx4 = getQuadIndex(tags[i], tags[i + 1], tags[i + 2], tags[i + 3]);
					increaseMap(sparseValues, idx4, 1);
				}
			}
		}

		
		return sparseValues;
	}
	
	public static void increaseMap(HashMap<Integer, Double> vmap, int idx, double increaser) {
		if (vmap.containsKey(idx)) {
			double existv = vmap.get(idx);
			vmap.put(idx, existv + increaser);
		} else {
			vmap.put(idx, increaser);
		}
	}
	

	public static void addVector(double[] v1, double[] v2) {
		if (v1.length != v2.length) {
			System.out.println("Vector length inconsistent: " + v1.length + "!=" + v2.length);
		}
		for (int i = 0; i < v1.length; i++) {
			v1[i] += v2[i];
		}
	}
	
	@Override
	public IFeatureVector getFeatureVectorDiff(IInstance x, IStructure y1, IStructure y2) {
		IFeatureVector f1 = getFeatureVector(x, y1);
		IFeatureVector f2 = getFeatureVector(x, y2);		
		return f1.difference(f2);
	}

}
