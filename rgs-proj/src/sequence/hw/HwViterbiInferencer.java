package sequence.hw;

import java.util.HashSet;
import java.util.List;
import java.util.Random;

import edu.illinois.cs.cogcomp.sl.core.AbstractFeatureGenerator;
import edu.illinois.cs.cogcomp.sl.core.AbstractInferenceSolver;
import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.util.IFeatureVector;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import general.AbstractInstance;
import general.AbstractOutput;
import search.SeachActionGenerator;
import search.SearchAction;
import search.SearchState;

public class HwViterbiInferencer extends AbstractInferenceSolver {


	private static final long serialVersionUID = 1L;
	AbstractFeatureGenerator featurizer;

	public HwViterbiInferencer(AbstractFeatureGenerator fzr) {
		featurizer = fzr;
	}

	@Override
	public Object clone(){
		return new HwViterbiInferencer(featurizer);
	}


	
	@Override
	public IStructure getLossAugmentedBestStructure(WeightVector wv, IInstance input, IStructure gold) {

		HwInstance x = (HwInstance) input;
		HwOutput ystar = (HwOutput) gold;

		int numOflabels = x.domainSize();
		int numOfTokens = x.size();

		float[][] dpTable = new float[numOfTokens][numOflabels];
		int[][] path = new int[numOfTokens][numOflabels];

		// Viterbi algorithm
		for (int j = 0; j < numOflabels; j++) {
			float priorScore = 0;
			float lossAug = 0;
			if (gold != null && j != ystar.getOutput(0)) lossAug = 1.0f;
			float zeroOrderScore = (computeUnaryScore(x.letterSegs.get(0).getFeatArr(), wv, j) + lossAug);
			dpTable[0][j] = priorScore + zeroOrderScore;   
			path[0][j] = -1;
		}

		for (int i = 1; i < numOfTokens; i++) {
			for (int j = 0; j < numOflabels; j++) {
				float lossAug = 0;
				if (gold != null && j != ystar.getOutput(0)) lossAug = 1.0f;
				float zeroOrderScore = (computeUnaryScore(x.letterSegs.get(i).getFeatArr(), wv, j) + lossAug);
				float bestScore = Float.NEGATIVE_INFINITY;
				for (int k = 0; k < numOflabels; k++) {
					float candidateScore = dpTable[i-1][k] + computeTransitionScore(wv, k, j);
					if (candidateScore > bestScore) {
						bestScore = candidateScore;
						path[i][j] = k;
					}
				}
				dpTable[i][j] = zeroOrderScore + bestScore;
			}
		}

		// find the best sequence   
		int[] tags = new int[numOfTokens];
		HwOutput pred = new HwOutput(numOfTokens, x.alphabet);

		int maxTag = 0;
		for (int i = 0; i < numOflabels; i++) {
			if (dpTable[numOfTokens - 1][i] > dpTable[numOfTokens - 1][maxTag]) 
				maxTag = i;
		}
		tags[numOfTokens - 1] = maxTag;

		// track back
		for (int i = (numOfTokens - 1); i >= 1; i--) { 
			tags[i - 1] = path[i][tags[i]];
		}
		for (int i = 0; i < numOfTokens; i++) {
			pred.setOutput(i, tags[i]);
		}

		//System.out.println("randsearch");
		return (IStructure) (pred);
	}
	
	public float computeUnaryScore(double[] unaryFeatArr, WeightVector wv, int j) {
		float re = 0;
		HwFeaturizer hwfzr = (HwFeaturizer)featurizer;
		for (int i = 0; i < unaryFeatArr.length; i++) {
			if (unaryFeatArr[i] != 0) {
				String fn = HwFeaturizer.getUnaryFeatName(hwfzr.alphabet, i, j);
				int idx = hwfzr.getIndex(fn);
				re += wv.get(idx);
			}
		}
		return re;
	}
	
	// from i to j
	public float computeTransitionScore(WeightVector wv, int i, int j) {
		float re = 0;
		HwFeaturizer hwfzr = (HwFeaturizer)featurizer;
		String fn2 = HwFeaturizer.getPairwiseFeatName(hwfzr.alphabet, i, j);
		int idx = hwfzr.getIndex(fn2);
		re += wv.get(idx);
		return re;
	}

	public float scoring(WeightVector wv, IInstance ins, IStructure gold,  IStructure pred) {
		IFeatureVector fv = featurizer.getFeatureVector(ins, pred);
		float product = wv.dotProduct(fv);
		float loss = 0;
		if (gold != null) loss = getLoss(ins, gold, pred);
		return (product + loss);
	}

	@Override
	public IStructure getBestStructure(WeightVector wv, IInstance input) throws Exception {
		return getLossAugmentedBestStructure(wv, input, null);
	}
	@Override
	public float getLoss(IInstance ins, IStructure goldStructure,  IStructure structure) {
		AbstractOutput goldLabeledSeq = (AbstractOutput) goldStructure;
		float loss = 0;
		for (int i = 0; i < goldLabeledSeq.size(); i++) {
			if (((AbstractOutput) structure).getOutput(i) != goldLabeledSeq.getOutput(i)) {
				loss += 1.0f;
			}
		}
		return loss;
	}

}
