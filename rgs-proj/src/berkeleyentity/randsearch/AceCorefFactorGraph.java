package berkeleyentity.randsearch;

import general.AbstractFactorGraph;
import general.AbstractFeaturizer;
import general.AbstractInstance;
//import ims.hotcoref.oregonstate.HotCorefDocInstance;
//import ims.hotcoref.oregonstate.HotCorefFeaturizer;
import search.SearchAction;
import sequence.hw.HwOutput;

public class AceCorefFactorGraph extends AbstractFactorGraph {
	
	private AceCorefInstance instance;
	private AceCorefFeaturizer featurizer;
	
	private double[][] cachedUnaryScores;
	private double cachedScore = 0;
	
	
	public AceCorefFactorGraph(AbstractInstance insti, AbstractFeaturizer ftzri) {
		init((AceCorefInstance)insti, (AceCorefFeaturizer)ftzri);
	}
	
	private void init(AceCorefInstance inst, AceCorefFeaturizer ftzr) {
		
		int outputSize = inst.size();
		
		instance = inst;
		featurizer = ftzr;
		
		cachedUnaryScores = new double[outputSize][];
		for (int i = 0; i < outputSize; i++) {
			cachedUnaryScores[i] = new double[inst.getDomainGivenIndex(i).length];
		}

	}
	
	public static double scoreWithOneHotFeature(int[] feat, double[] w) {
		double sc = 0;
		for (int i = 0; i < feat.length; i++) {
			int fidx = feat[i];
			assert (fidx < w.length);
			sc += (w[fidx]);
		}
		return sc;
	}

	@Override
	public void updateScoreTable(double[] weights) {
		for (int i = 0; i < instance.size(); i++) {
			int[] domains = instance.getDomainGivenIndex(i);
			for (int j = 0; j < domains.length; j++) {
				int jvIdx = domains[j];
				int[] pairFeat = featurizer.getEdgeFeat(instance, i, jvIdx);  //instance.getMentPairFeature(i, jvIdx);
				cachedUnaryScores[i][jvIdx] = scoreWithOneHotFeature(pairFeat, weights);
			}
		}
	}

	@Override
	public double computeScoreWithTable(double[] weights, HwOutput output) {
		double score = 0;
		for (int i = 0; i < instance.size(); i++) {
			score += cachedUnaryScores[i][output.getOutput(i)];
		}
		return score;
	}

	@Override
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
		
		return scoreDiff;
	}

	@Override
	public double computeScore(double[] weights, HwOutput output) {
		throw (new RuntimeException("Not implemented!"));
		//return 0;
	}

	@Override
	public double getCachedScore() {
		throw (new RuntimeException("Not implemented!"));
		//return 0;
	}

}
