package imgcnn;

import java.util.Arrays;
import java.util.List;

import edu.berkeley.nlp.futile.util.Counter;
import general.AbstractFactorGraph;
import general.AbstractFeaturizer;
import general.AbstractInstance;
import imgseg.ImageInstance;
import imgseg.ImageSuperPixel;
import search.SearchAction;
import sequence.hw.HwOutput;
import sequence.hw.HwSegment;

public class ImageCNNFactorGraph extends AbstractFactorGraph {
	
	private ImageInstance instance;
	private int outputSize;
	private int domainSize;
	public String[] alphabet;
	
	private ImageCNNFeaturizer featurizer;
	
	private double[][] cachedUnaryScores;

	private double cachedScore = 0;
	
	
	public ImageCNNFactorGraph(AbstractInstance insti, AbstractFeaturizer ftzri) {
		init((ImageInstance)insti, (ImageCNNFeaturizer)ftzri);
	}
	
	public ImageCNNFactorGraph(ImageInstance inst, ImageCNNFeaturizer ftzr) {
		init(inst, ftzr);
	}
	
	private void init(ImageInstance inst, ImageCNNFeaturizer ftzr) {
		outputSize = inst.size();
		alphabet = ftzr.alphabet;
		domainSize = alphabet.length;
		
		instance = inst;
		featurizer = ftzr;
		
		cachedUnaryScores = new double[outputSize][domainSize];
		for (int i = 0; i < outputSize; i++) {
			cachedUnaryScores[i] = new double[domainSize];
		}
	}

	@Override
	public void updateScoreTable(double[] weights) {
		
		// unary score table
		
		for (int i = 0; i < outputSize; i++) {
			for (int j = 0; j < domainSize; j++) {
				cachedUnaryScores[i][j] = 0;
				int supIdx = instance.segIdxToSupIdx(i);
				double[] ufeat = instance.getSuPix(supIdx).features[j];
				for (int k = 0; k < ufeat.length; k++) {
					String unaryfn = ImageCNNFeaturizer.getUnaryFeatName(alphabet, k, j);
					int idx = featurizer.getIndex(unaryfn);
					cachedUnaryScores[i][j] += (weights[idx] * ufeat[k]);
				}
			}
		}
		
		
		//global term table
		
		
	}

	@Override
	public double computeScoreWithTable(double[] weights, HwOutput output) {
		
		// super pixels
		ImageSuperPixel[] supixels = instance.getSuPixArr();
		List<HwSegment> segs = instance.letterSegs;
		
		int pairCnt1 = 0, pairCnt2 = 0;
		Counter<String> pairCntMap = new Counter<String>();
		double score = 0;
		
		double[] labelCnt = new double[alphabet.length];
		Arrays.fill(labelCnt, 0);
		
		for (int i = 0; i < outputSize; i++) {
			
			// unary
			score += cachedUnaryScores[i][output.getOutput(i)];
			
			// binary
			if (featurizer.considerPairs) {
				/*
				int supIdx = segs.get(i).index;
				int[] neigbours = supixels[supIdx].neighours;
				for (int jdx = 0; jdx < neigbours.length; jdx++) {
					int j = neigbours[jdx];
					int segIdx = supixels[j].hwsegIndex;
					if (segIdx >= 0) { // not a "void" pixel
						pairCnt1++;
						
						pairCntMap.incrementCount(normalizedPairStr(i, segIdx), 1.0);
						
						if (i < segIdx) { // no repeat pair
							String pairfn = ImageCNNFeaturizer.getPairwiseFeatName(alphabet, output.getOutput(i),output.getOutput(segIdx));
							int idx2 = featurizer.getIndex(pairfn);
							score += (weights[idx2]);
							pairCnt2++;
						}						
					}
				}
				*/
				int[] rightNbrs = instance.getRightNeighbours(i);
				for (int jdx = 0; jdx < rightNbrs.length; jdx++) {
					int j = rightNbrs[jdx];
					String pairfn = ImageCNNFeaturizer.getPairwiseFeatName(alphabet, output.getOutput(i),output.getOutput(j));
					int idx2 = featurizer.getIndex(pairfn);
					score += (weights[idx2]);
				}
			}
			
			// global count
			labelCnt[output.getOutput(i)]++;
		}
		
		// global
		if (featurizer.considerGlobal) {
			for (int jval = 0; jval < alphabet.length; jval++) {
				double cnt1 = labelCnt[jval];
				double cnt0 = (double)output.size() - labelCnt[jval];
				int idx1 = featurizer.getIndex(ImageCNNFeaturizer.getGlobalFeatName(alphabet, jval, 1));
				score += (cnt1 * weights[idx1]);
				int idx0 = featurizer.getIndex(ImageCNNFeaturizer.getGlobalFeatName(alphabet, jval, 0));
				score += (cnt0 * weights[idx0]);
			}
		}

		return score;
	}
	
	@Override
	public double computeScoreDiffWithTable(double[] weights, SearchAction action, HwOutput output) {
		
		//ImageSuperPixel[] supixels = instance.getSuPixArr();
		//List<HwSegment> segs = instance.letterSegs;
		
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
			double sc1 = 0;
			
			// right neighbors
			int[] rightNbrs = instance.getRightNeighbours(vIdx);
			for (int jdx = 0; jdx < rightNbrs.length; jdx++) {
				int j = rightNbrs[jdx];
				// minus old score
				int idxo = featurizer.getPairIndex(oldv, output.getOutput(j)); //getPairIdx(oldv, output.getOutput(j));
				sc1 -= (weights[idxo]);
				// plus new score
				int idxn = featurizer.getPairIndex(newv, output.getOutput(j));//getPairIdx(newv, output.getOutput(j));
				sc1 += (weights[idxn]);
			}
			
			// left neighbors
			int[] leftNbrs = instance.getLeftNeighbours(vIdx);
			for (int jdx = 0; jdx < leftNbrs.length; jdx++) {
				int j = leftNbrs[jdx];
				// minus old score
				int idxo = featurizer.getPairIndex(output.getOutput(j), oldv);//getPairIdx(output.getOutput(j), oldv);
				sc1 -= (weights[idxo]);
				// plus new score
				int idxn = featurizer.getPairIndex(output.getOutput(j), newv);//getPairIdx(output.getOutput(j), newv);
				sc1 += (weights[idxn]);
			}
			
			/*
			int supIdx = segs.get(vIdx).index;
			int[] neigbours = supixels[supIdx].neighours;
			for (int jdx = 0; jdx < neigbours.length; jdx++) {
				int j = neigbours[jdx];
				int segIdx = supixels[j].hwsegIndex;
				if (segIdx >= 0) { // not a "void" pixel
					// minus old score
					int idxo = getPairIdx(vIdx, segIdx, oldv, output.getOutput(segIdx));
					sc1 -= (weights[idxo]);
					// plus new score
					int idxn = getPairIdx(vIdx, segIdx, newv, output.getOutput(segIdx));
					sc1 += (weights[idxn]);					
				}
			}*/
			scoreDiff += sc1;
		}

		// global
		if (featurizer.considerGlobal) {
			double sc1 = 0;
			// old value
			int idxOld1 = featurizer.getGlobalIndex(oldv, 1);//featurizer.getIndex(ImageCNNFeaturizer.getGlobalFeatName(alphabet, oldv, 1)); // 1 -> 0
			sc1 -= weights[idxOld1];
			int idxOld0 = featurizer.getGlobalIndex(oldv, 0);//featurizer.getIndex(ImageCNNFeaturizer.getGlobalFeatName(alphabet, oldv, 0));
			sc1 += weights[idxOld0];
			// new value
			int idxNew1 = featurizer.getGlobalIndex(newv, 1);//featurizer.getIndex(ImageCNNFeaturizer.getGlobalFeatName(alphabet, newv, 1)); // 0 -> 1
			sc1 += weights[idxNew1];
			int idxNew0 = featurizer.getGlobalIndex(newv, 0);//featurizer.getIndex(ImageCNNFeaturizer.getGlobalFeatName(alphabet, newv, 0));
			sc1 -= weights[idxNew0];
			scoreDiff += sc1;
		}

		// set back
		//output.setOutput(vIdx, originValue);
		return scoreDiff;
	}

	@Override
	public double computeScore(double[] weights, HwOutput output) {
		return 0; // useless...
	}

	@Override
	public double getCachedScore() {
		return cachedScore;
	}

}
