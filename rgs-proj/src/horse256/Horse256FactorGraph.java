package horse256;

import java.util.Arrays;
import java.util.List;

import edu.berkeley.nlp.futile.util.Counter;
import general.AbstractFactorGraph;
import general.AbstractFeaturizer;
import general.AbstractInstance;
import imgseg.ImageSuperPixel;
import search.SearchAction;
import sequence.hw.HwOutput;
import sequence.hw.HwSegment;

public class Horse256FactorGraph extends AbstractFactorGraph {
	
	private Horse256Instance instance;
	private int outputSize;
	private int domainSize;
	public String[] alphabet;
	
	private Horse256Featurizer featurizer;
	
	private double[][] cachedUnaryScores;

	private double cachedScore = 0;
	
	
	public Horse256FactorGraph(AbstractInstance insti, AbstractFeaturizer ftzri) {
		init((Horse256Instance)insti, (Horse256Featurizer)ftzri);
	}
	
	public Horse256FactorGraph(Horse256Instance inst, Horse256Featurizer ftzr) {
		init(inst, ftzr);
	}
	
	private void init(Horse256Instance inst, Horse256Featurizer ftzr) {
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
				int supIdx = i; //instance.segIdxToSupIdx(i);
				double[] ufeat = instance.getSuPix(supIdx).features[j];
				for (int k = 0; k < ufeat.length; k++) {
					String unaryfn = Horse256Featurizer.getUnaryFeatName(alphabet, k, j);
					int idx = featurizer.getIndex(unaryfn);
					cachedUnaryScores[i][j] += (weights[idx] * ufeat[k]);
				}
			}
		}
		
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
			
			/*
			// binary
			if (featurizer.considerPairs) {
				int[] rightNbrs = instance.getRightNeighbours(i);
				for (int jdx = 0; jdx < rightNbrs.length; jdx++) {
					int j = rightNbrs[jdx];
					String pairfn = ImageSegFeaturizer.getPairwiseFeatName(alphabet, output.getOutput(i),output.getOutput(j));
					int idx2 = featurizer.getIndex(pairfn);
					score += (weights[idx2]);
				}
			}
			*/
			// global count
			labelCnt[output.getOutput(i)]++;
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
			
/*
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
*/
			
			scoreDiff += sc1;
		}

		return scoreDiff;
	}

	@Override
	public double computeScore(double[] weights, HwOutput output) {
		return 0;
	}

	@Override
	public double getCachedScore() {
		return cachedScore;
	}

}
