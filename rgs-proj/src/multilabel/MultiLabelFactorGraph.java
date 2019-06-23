package multilabel;

import general.AbstractFactorGraph;
import general.AbstractFeaturizer;
import general.AbstractInstance;
import search.SearchAction;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;

public class MultiLabelFactorGraph  extends AbstractFactorGraph {

	private HwInstance instance;
	private MultiLabelFeaturizer featurizer;
	
	public int labelCnt = -1;
	public int singleFeatLen = -1;
	//public boolean considerPairs = false;
	//public boolean considerTriplets = false;
	//public boolean considerQuadruples = false;
	
	double[][] unaryTable;

	
	
	public MultiLabelFactorGraph(AbstractInstance insti, AbstractFeaturizer ftzri) {
		init((HwInstance)insti, (MultiLabelFeaturizer)ftzri);
	}
	
	public MultiLabelFactorGraph(HwInstance inst, MultiLabelFeaturizer ftzr) {
		init(inst, ftzr);
	}
	
	private void init(HwInstance inst, MultiLabelFeaturizer ftzr) {

		featurizer = ftzr;
		instance = inst;
		
		labelCnt = inst.size();
		singleFeatLen = ftzr.singleFeatLen;
		
		// unary table
		unaryTable = new double[inst.size()][2];
		for (int i = 0; i < inst.size(); i++) {
			unaryTable[i] = new double[2];
		}
		
	}

	@Override
	public void updateScoreTable(double[] weights) {
		
		for (int i = 0; i < labelCnt; i++) {
			unaryTable[i][0] = 0;
			unaryTable[i][1] = 0;
			double[] feat = instance.getUnaryFeats(i);
			for (int j = 0; j < feat.length; j++) {
				//String unaryfn = featurizer.getCompatibleFeatName(i, j);
				//int idx = featurizer.getIndex(unaryfn);
				int idx = featurizer.getCompatibleFeatIndex(i, j);
				unaryTable[i][1] += (weights[idx] * feat[j]);
			}
		}
		
		
	}

	@Override
	public double computeScoreWithTable(double[] weights, HwOutput output) {
		
		double score = 0;
		
		for (int i = 0; i < output.size(); i++) {
			if (output.getOutput(i) > 0) {
				score += unaryTable[i][1];
			}
		}
		
		if (featurizer.considerPairs) {
			for (int j1 = 0; j1 < output.size(); j1++) {
				for (int j2 = (j1 + 1); j2 < output.size(); j2++) {
					//int idx = featurizer.getIndex(featurizer.getLabelPairFeatName(j1, j2, output.getOutput(j1), output.getOutput(j2)));
					int idx = featurizer.getLabelPairFeatIndex(j1, j2, output.getOutput(j1), output.getOutput(j2));
					score += (weights[idx]);
				}
			}
		}

		return score;
	}

	@Override
	public double computeScoreDiffWithTable(double[] weights, SearchAction action, HwOutput output) {
		
		double scoreDiff = 0;
		
		int vIdx = action.getSlotIdx();
		int oldv = action.getOldVal();
		int newv = action.getNewVal();
		
		// unary
		scoreDiff -= unaryTable[vIdx][oldv];
		scoreDiff += unaryTable[vIdx][newv];
		
		if (featurizer.considerPairs) {
			// pairwise
			for (int j = 0; j < output.size(); j++) {
				if (j != vIdx) {
					// - oldValue
					//int idxOld = featurizer.getIndex(featurizer.getLabelPairFeatName(j, vIdx, output.getOutput(j), oldv));
					int idxOld = featurizer.getLabelPairFeatIndex(j, vIdx, output.getOutput(j), oldv);
					scoreDiff -= (weights[idxOld]);
					// + newValue
					//int idxNew = featurizer.getIndex(featurizer.getLabelPairFeatName(j, vIdx, output.getOutput(j), newv));
					int idxNew = featurizer.getLabelPairFeatIndex(j, vIdx, output.getOutput(j), newv);
					scoreDiff += (weights[idxNew]);
				}
			}
		}
		
		return scoreDiff;
	}

	@Override
	public double computeScore(double[] weights, HwOutput output) {
		return 0;
	}

	@Override
	public double getCachedScore() {
		return 0;
	}
	
}
