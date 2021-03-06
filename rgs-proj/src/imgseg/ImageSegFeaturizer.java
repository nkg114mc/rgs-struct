package imgseg;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import edu.berkeley.nlp.futile.fig.basic.Indexer;
import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.util.FeatureVectorBuffer;
import edu.illinois.cs.cogcomp.sl.util.IFeatureVector;
import general.AbstractFeaturizer;
import general.AbstractInstance;
import general.AbstractOutput;
import sequence.hw.HwFeaturizer;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;
import sequence.hw.HwSegment;

public class ImageSegFeaturizer  extends AbstractFeaturizer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7227727232590596681L;
	
	public String[] alphabet = null;
	private final int singleFeatLen = 6;
	
	public boolean considerPairs = false;
	public boolean considerGlobal = false;
	public Indexer<String> featureIndexer = null;
	
	private int[][] unaryIndex;
	private int[][] pairIndex;
	private int[][] globalIndex;
	
	public ImageSegFeaturizer(String[] albt, boolean pairwise, boolean global) {
		alphabet = albt;
		considerPairs = pairwise;
		considerGlobal = global;
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
	
	public static String getGlobalFeatName(String[] albt, int yIdx, int yg) {
		String g3FeatName = "Global:" + String.valueOf(yg) + "-" + getLabelName(albt, yIdx);
		return g3FeatName;
	}
	
	
	private void initFeatIndexer() {
		
		featureIndexer = new Indexer<String>();
		
		unaryIndex = new int[alphabet.length][singleFeatLen];
		pairIndex = new int[alphabet.length][alphabet.length];
		globalIndex = new int[alphabet.length][2];
		
		// consistency feature
		for (int yi = 0; yi < alphabet.length; yi++) {
			for (int xi = 0; xi < singleFeatLen; xi++) {
				String compitbleFeatName = getUnaryFeatName(alphabet, xi, yi);
				int idx = featureIndexer.getIndex(compitbleFeatName);
				unaryIndex[yi][xi] = idx;
			}
		}
		
		// pair feature
		if (considerPairs) {
			for (int y1 = 0; y1 < alphabet.length; y1++) {
				for (int y2 = 0; y2 < alphabet.length; y2++) {
					String pairFeatName = getPairwiseFeatName(alphabet, y1, y2);
					int idx = featureIndexer.getIndex(pairFeatName);
					pairIndex[y1][y2] = idx;
				}
			}
		}
		
		// global feature
		if (considerGlobal) {
			for (int y1 = 0; y1 < alphabet.length; y1++) {
				for (int yg = 0; yg < 2; yg++) { // yg \in {0, 1}
					String tripleFeatName = getGlobalFeatName(alphabet, y1, yg);
					int idx = featureIndexer.getIndex(tripleFeatName);
					globalIndex[y1][yg] = idx;
				}
			}
		}

		System.out.println("Feature length = " + featureIndexer.size());
		
	}

	public int getFeatLen() {
		return featureIndexer.size();
	}
	public int getIndex(String featName) {
		return featureIndexer.indexOf(featName);
	}
	
	public int getUnaryIndex(int xIdx, int yIdx) {
		return unaryIndex[yIdx][xIdx];
	}
	public int getPairIndex(int y1Idx, int y2Idx) {
		return pairIndex[y1Idx][y2Idx];
	}
	public int getGlobalIndex(int yIdx, int yg) {
		return globalIndex[yIdx][yg];
	}

	@Override
	public HashMap<Integer, Double> featurize(AbstractInstance xi, AbstractOutput yi) {
		
		ImageInstance x = (ImageInstance)(xi);
		HwOutput y = (HwOutput)(yi);

		HashMap<Integer, Double> sparseValues = new HashMap<Integer, Double>();
		
		
		// super pixels
		ImageSuperPixel[] supixels = x.getSuPixArr();
		List<HwSegment> segs = x.letterSegs;
		
		assert (y.size() == segs.size());

		// unary features
		double[][] unaryFeats = new double[alphabet.length][];
		for (int j = 0; j < alphabet.length; j++) {
			unaryFeats[j] = new double[singleFeatLen];
			Arrays.fill(unaryFeats[j], 0);
		}

		int[] tags = y.output;
		for (int i = 0; i < tags.length; i++) {
			int value = tags[i];
			int supIdx = segs.get(i).index;
			double[] ft = x.getSuPix(supIdx).features[value];//(x.letterSegs.get(i).imgDblArr);
			HwFeaturizer.addVector(unaryFeats[value], ft);
		}
		
		for (int i = 0; i < unaryFeats.length; i++) {
			for (int j = 0; j < unaryFeats[i].length; j++) {
				if (unaryFeats[i][j] != 0) {
					String unaryfn = getUnaryFeatName(alphabet, j, i);
					int idx = getIndex(unaryfn) ;
					sparseValues.put(idx, unaryFeats[i][j]);
				}
			}
		}
		
		// pairwise features
		/////
		//// scane the entire adjacent list, only neighbours will be considered
		////
		if (considerPairs) {
			for (int i = 0; i < tags.length; i++) {
				int supIdx = segs.get(i).index;
				int[] neigbours = supixels[supIdx].neighours;
				for (int jdx = 0; jdx < neigbours.length; jdx++) {
					int j = neigbours[jdx];
					int segIdx = supixels[j].hwsegIndex;
					if (segIdx >= 0) { // not a "void" pixel
						//if (i < segIdx) {
							String pairfn = getPairwiseFeatName(alphabet, tags[i], tags[segIdx]);
							int idx2 = getIndex(pairfn) ;
							HwFeaturizer.increaseMap(sparseValues, idx2, 1);
						//}						
					}
				}
			}
		}
		
		
		// global features
		if (considerGlobal) {
			
			double[] labelCnt = new double[alphabet.length];
			Arrays.fill(labelCnt, 0);
			for (int i = 0; i < tags.length; i++) {
				labelCnt[tags[i]]++;
			}
			
			for (int jv = 0; jv < alphabet.length; jv++) {
				double cnt1 = labelCnt[jv];
				double cnt0 = ((double)tags.length) - labelCnt[jv];
				// yg = 1
				String globalfn1 = getGlobalFeatName(alphabet, jv, 1);
				int idxyg1 = getIndex(globalfn1);
				sparseValues.put(idxyg1, cnt1);
				// yg = 0
				String globalfn0 = getGlobalFeatName(alphabet, jv, 0);
				int idxyg0 = getIndex(globalfn0);
				sparseValues.put(idxyg0, cnt0);
			}
		}
		
		return sparseValues;
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
	
	
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	
	// return a single unary feature
	public HashMap<Integer, Double> featurizeUnary(AbstractInstance xi, AbstractOutput yi, int yIndex) {
		
		ImageInstance x = (ImageInstance)(xi);
		HwOutput y = (HwOutput)(yi);

		HashMap<Integer, Double> sparseValues = new HashMap<Integer, Double>();
		
		
		// super pixels
		//ImageSuperPixel[] supixels = x.getSuPixArr();
		List<HwSegment> segs = x.letterSegs;
		
		assert (y.size() == segs.size());

		// unary features
		int i = yIndex;
		int supIdx = segs.get(i).index;
		int value = y.output[i];
		double[] ft = x.getSuPix(supIdx).features[value];


		for (int j = 0; j < ft.length; j++) {
			if (ft[j] > 0) {
				String unaryfn = getUnaryFeatName(alphabet, j, value);
				int idx = getIndex(unaryfn) ;
				sparseValues.put(idx, ft[j]);
			}
		}
		
		return sparseValues;
	}
	
	public IFeatureVector getUnaryFeatureVector(IInstance xi, IStructure yi, int yIndex) {
		FeatureVectorBuffer fv = new FeatureVectorBuffer();
		HwInstance x = (HwInstance) xi;
		HwOutput y = (HwOutput) yi;
		
		HashMap<Integer, Double> feats = featurizeUnary(x, y, yIndex);
		for (int idx : feats.keySet()) {
			fv.addFeature(idx, feats.get(idx));			
		}
		return fv.toFeatureVector();
	}
	
	
	// return a single unary feature
	public HashMap<Integer, Double> featurizeSuperPixel(ImageSuperPixel supxl, int yValue) {

		HashMap<Integer, Double> sparseValues = new HashMap<Integer, Double>();

		// unary features
		int value = yValue;
		double[] ft = supxl.features[value];

		for (int j = 0; j < ft.length; j++) {
			if (ft[j] > 0) {
				String unaryfn = getUnaryFeatName(alphabet, j, value);
				int idx = getIndex(unaryfn) ;
				sparseValues.put(idx, ft[j]);
			}
		}
		
		return sparseValues;
	}
	
	public IFeatureVector getSuPixFeatureVector(ImageSuperPixel supxl, int yValue) {
		FeatureVectorBuffer fv = new FeatureVectorBuffer();
		HashMap<Integer, Double> feats = featurizeSuperPixel(supxl,yValue);
		for (int idx : feats.keySet()) {
			fv.addFeature(idx, feats.get(idx));			
		}
		return fv.toFeatureVector();
	}
	
	
	@Override
	public IFeatureVector getFeatureVectorDiff(IInstance x, IStructure y1, IStructure y2) {
		IFeatureVector f1 = getFeatureVector(x, y1);
		IFeatureVector f2 = getFeatureVector(x, y2);		
		return f1.difference(f2);
	}

}
