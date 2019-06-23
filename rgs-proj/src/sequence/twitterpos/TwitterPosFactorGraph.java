package sequence.twitterpos;

import java.util.ArrayList;
import java.util.List;

import general.AbstractFactorGraph;
import general.AbstractFeaturizer;
import general.AbstractInstance;
import search.SearchAction;
import sequence.hw.HwOutput;

public class TwitterPosFactorGraph extends AbstractFactorGraph {
	
	private TwitterPosExample instance;
	private int outputSize;
	private int domainSize;
	public String[] alphabet;
	
	private TwitterPosFeaturizer featurizer;
	
	private double[][] cachedUnaryScores;
	
	// related factor index
	private int[][] relatedUnaryFacIdx;
	private int[][] relatedPairFacIdx;
	private int[][] relatedTripFacIdx;
	private int[][] relatedQuadFacIdx;

	private double cachedScore = 0;
	
	
	public TwitterPosFactorGraph(AbstractInstance insti, AbstractFeaturizer ftzri) {
		init((TwitterPosExample)insti, (TwitterPosFeaturizer)ftzri);
	}
	
	public TwitterPosFactorGraph(TwitterPosExample inst, TwitterPosFeaturizer ftzr) {
		init(inst, ftzr);
	}
	
	private void init(TwitterPosExample inst, TwitterPosFeaturizer ftzr) {
		int i;
		outputSize = inst.size();
		alphabet = inst.alphabet;
		domainSize = alphabet.length;
		
		instance = inst;
		featurizer = ftzr;
		
		cachedUnaryScores = new double[outputSize][domainSize];
		for (i = 0; i < outputSize; i++) {
			cachedUnaryScores[i] = new double[domainSize];
		}
		
		// compute related
		computeRelatedFactors();

	}
	
	public void computeRelatedFactors() {
		relatedUnaryFacIdx = new int[outputSize][];
		relatedPairFacIdx = new int[outputSize][];
		relatedTripFacIdx = new int[outputSize][];
		relatedQuadFacIdx = new int[outputSize][];
		int i, j;
		for (i = 0; i < relatedUnaryFacIdx.length; i++) {
			relatedUnaryFacIdx[i] = new int[1];
			relatedUnaryFacIdx[i][0] = i;
		}
		for (i = 0; i < relatedPairFacIdx.length; i++) {
			ArrayList<Integer> facs = new ArrayList<Integer>();
			for (j = (i - 1); j <= i; j++) { // start
				int end = j + 1;
				if ((j >= 0) && (end <= (outputSize - 1))) {
					facs.add(j);
				}
			}
			relatedPairFacIdx[i] = toIntPrime(facs);
		}
		for (i = 0; i < relatedTripFacIdx.length; i++) {
			ArrayList<Integer> facs = new ArrayList<Integer>();
			for (j = (i - 2); j <= i; j++) { // start
				int end = j + 2;
				if ((j >= 0) && (end <= (outputSize - 1))) {
					facs.add(j);
				}
			}
			relatedTripFacIdx[i] = toIntPrime(facs);
		}
		for (i = 0; i < relatedQuadFacIdx.length; i++) {
			ArrayList<Integer> facs = new ArrayList<Integer>();
			for (j = (i - 3); j <= i; j++) { // start
				int end = j + 3;
				if ((j >= 0) && (end <= (outputSize - 1))) {
					facs.add(j);
				}
			}
			relatedQuadFacIdx[i] = toIntPrime(facs);
		}
	}
	
	private int[] toIntPrime(List<Integer> myList) {
		int[] arr = new int[myList.size()];
		for(int i = 0; i < myList.size(); i++) {
		    if (myList.get(i) != null) {
		        arr[i] = myList.get(i);
		    }
		}
		return arr;
	}
	
	public void updateScoreTable(double[] weights) {
		for (int i = 0; i < outputSize; i++) {
			for (int j = 0; j < domainSize; j++) {
				cachedUnaryScores[i][j] = 0;
				double[] ufeat = instance.getUnaryFeats(i);
				for (int k = 0; k < ufeat.length; k++) {
					//String unaryfn = HwFeaturizer.getUnaryFeatName(alphabet, k, j);
					//int idx = featurizer.getIndex(unaryfn);
					int idx = featurizer.getUnaryIndex(k, j);
					cachedUnaryScores[i][j] += (weights[idx] * ufeat[k]);
				}
			}
		}
	}
	
	public double computeScoreWithTable(double[] weights, HwOutput output) {
		
		double score = 0;
		for (int i = 0; i < outputSize; i++) {
			
			// unary
			score += cachedUnaryScores[i][output.getOutput(i)];
			
			// binary
			if (featurizer.considerPairs) {
				if (i < (outputSize - 1)) {
					int v1 = output.getOutput(i);
					int v2 = output.getOutput(i + 1);
					//int idx = featurizer.getIndex(HwFeaturizer.getPairwiseFeatName(alphabet, v1, v2));
					int idx = featurizer.getPairIndex(v1, v2);
					score += weights[idx];
				}
			}
			
			// ternary
			if (featurizer.considerTriplets) {
				if (i < (outputSize - 2)) {
					int v1 = output.getOutput(i);
					int v2 = output.getOutput(i + 1);
					int v3 = output.getOutput(i + 2);
					//int idx = featurizer.getIndex(HwFeaturizer.getTenaryFeatName(alphabet, v1, v2, v3));
					int idx = featurizer.getTenaryIndex(v1, v2, v3);
					score += weights[idx];
				}
			}
			
			// quatery
			if (featurizer.considerQuadruples) {
				if (i < (outputSize - 3)) {
					int v1 = output.getOutput(i);
					int v2 = output.getOutput(i + 1);
					int v3 = output.getOutput(i + 2);
					int v4 = output.getOutput(i + 3);
					//int idx = featurizer.getIndex(HwFeaturizer.getQuadFeatName(alphabet, v1, v2, v3, v4));
					int idx = featurizer.getQuadIndex(v1, v2, v3, v4);//.getIndex(HwFeaturizer.getQuadFeatName(alphabet, v1, v2, v3, v4));
					score += weights[idx];
				}
			}

		}
		
		return score;
	}
	
	// compute the different only!
	public double computeScoreDiffWithTable(double[] weights, SearchAction action, HwOutput output) {
		
		
		double scoreDiff = 0;
		
		int vIdx = action.getSlotIdx();
		int oldv = action.getOldVal();
		int newv = action.getNewVal();

		// store origin value
		int originValue = output.getOutput(vIdx);
		assert (newv == originValue);
		
		// unary
		scoreDiff -= cachedUnaryScores[vIdx][oldv];
		scoreDiff += cachedUnaryScores[vIdx][newv];

		// binary
		if (featurizer.considerPairs) {
			// change relative factor scores
			int[] relatedBinFacIdx = relatedPairFacIdx[vIdx];
			double sc1 = 0;
			output.setOutput(vIdx, oldv);
			for (int i = 0; i < relatedBinFacIdx.length; i++) {
				int j = relatedBinFacIdx[i];
				int v1 = output.getOutput(j);
				int v2 = output.getOutput(j + 1);
				//int idx1 = featurizer.getIndex(HwFeaturizer.getPairwiseFeatName(alphabet, v1, v2));
				int idx1 = featurizer.getPairIndex(v1, v2);
				sc1 -= weights[idx1];
			}
			output.setOutput(vIdx, newv);
			for (int i = 0; i < relatedBinFacIdx.length; i++) {
				int j = relatedBinFacIdx[i];
				int v1 = output.getOutput(j);
				int v2 = output.getOutput(j + 1);
				//int idx2 = featurizer.getIndex(HwFeaturizer.getPairwiseFeatName(alphabet, v1, v2));
				int idx2 = featurizer.getPairIndex(v1, v2);
				sc1 += weights[idx2];
			}
			scoreDiff += sc1;
		}

		// ternary
		if (featurizer.considerTriplets) {
			// change relative factor scores
			int[] relatedTriFacIdx = relatedTripFacIdx[vIdx];
			double sc1 = 0;
			output.setOutput(vIdx, oldv);
			for (int i = 0; i < relatedTriFacIdx.length; i++) {
				int j = relatedTriFacIdx[i];
				int v1 = output.getOutput(j);
				int v2 = output.getOutput(j + 1);
				int v3 = output.getOutput(j + 2);
				//int idx1 = featurizer.getIndex(HwFeaturizer.getTenaryFeatName(alphabet, v1, v2, v3));
				int idx1 = featurizer.getTenaryIndex(v1, v2, v3);
				sc1 -= weights[idx1];
			}
			output.setOutput(vIdx, newv);
			for (int i = 0; i < relatedTriFacIdx.length; i++) {
				int j = relatedTriFacIdx[i];
				int v1 = output.getOutput(j);
				int v2 = output.getOutput(j + 1);
				int v3 = output.getOutput(j + 2);
				//int idx2 = featurizer.getIndex(HwFeaturizer.getTenaryFeatName(alphabet, v1, v2, v3));
				int idx2 = featurizer.getTenaryIndex(v1, v2, v3);
				sc1 += weights[idx2];
			}
			scoreDiff += sc1;
		}

		// quatery
		if (featurizer.considerQuadruples) {
			// change relative factor scores
			int[] relatedQuaFacIdx = relatedQuadFacIdx[vIdx];
			double sc1 = 0;
			output.setOutput(vIdx, oldv);
			for (int i = 0; i < relatedQuaFacIdx.length; i++) {
				int j = relatedQuaFacIdx[i];
				int v1 = output.getOutput(j);
				int v2 = output.getOutput(j + 1);
				int v3 = output.getOutput(j + 2);
				int v4 = output.getOutput(j + 3);
				//int idx1 = featurizer.getIndex(HwFeaturizer.getQuadFeatName(alphabet, v1, v2, v3, v4));
				int idx1 = featurizer.getQuadIndex(v1, v2, v3, v4);
				sc1 -= weights[idx1];
			}
			output.setOutput(vIdx, newv);
			for (int i = 0; i < relatedQuaFacIdx.length; i++) {
				int j = relatedQuaFacIdx[i];
				int v1 = output.getOutput(j);
				int v2 = output.getOutput(j + 1);
				int v3 = output.getOutput(j + 2);
				int v4 = output.getOutput(j + 3);
				//int idx2 = featurizer.getIndex(HwFeaturizer.getQuadFeatName(alphabet, v1, v2, v3, v4));
				int idx2 = featurizer.getQuadIndex(v1, v2, v3, v4);
				sc1 += weights[idx2];
			}
			scoreDiff += sc1;
		}

		// set back
		output.setOutput(vIdx, originValue);
		return scoreDiff;
	}
	
	public double computeScore(double[] weights, HwOutput output) {
		
		double sc = 0;
		return sc;
	}
	
	public double getCachedScore() {
		return cachedScore;
	}

}
