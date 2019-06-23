package multilabel.learning.inferencer;


import java.util.ArrayList;

import multilabel.instance.Example;
import multilabel.instance.Featurizer;
import multilabel.learning.cost.CostFunction;
import multilabel.learning.search.OldSearchState;

public class LinearCostFunction extends CostFunction {
	
	int featDim;
	int labelDim;
	double[][] unaryScores;
	double[][] binaryScores;
	
	int totalFeatLen;
	int totalUnaryLen;
	int totalBinaryLen;
	
	public LinearCostFunction() {
		featDim = -1;
		labelDim = -1;
		unaryScores = null;
		binaryScores = null;
		
		totalFeatLen = -1;
		totalUnaryLen = -1;
		totalBinaryLen = -1;
	}
	
	
	void init(Example ex) {
		featDim = ex.featDim();
		labelDim = ex.labelDim();
		
		int pairCnt = (labelDim * (labelDim - 1)) / 2;
		unaryScores = new double[labelDim][2];
		binaryScores = new double[pairCnt][4];
		
		totalUnaryLen = labelDim * featDim;
	}
	
	double computeDotProduct(ArrayList<Double> feats, multilabel.instance.OldWeightVector wv, int offset) {
		double result = 0;
		for (int i = 0; i < feats.size(); i++) {
			result += (feats.get(i) * wv.get(i + offset));
		}
		return result;
	}

	
	public void loadNewWeight(Example ex, multilabel.instance.OldWeightVector wv) {
		init(ex);
		
		// unary
		for (int i = 0; i < labelDim; i++) {
			// value = 0
			unaryScores[i][0] = 0;
			/////////////////////////////////////
			// value = 1
			unaryScores[i][1] = computeDotProduct(ex.getFeat(), wv, (i * labelDim));
		}
		
		// binary
		for (int k = 0; k < (labelDim - 1); k++) {
			for (int k2 = (k + 1); k2 < labelDim; k2++) {
				
				int computedIdx = Featurizer.computePairIndex(k, k2, labelDim);
				for (int valk = 0; valk < 2; valk++) {
					for (int valk2 = 0; valk2 < 2; valk2++) {
						int indicatorIdx = Featurizer.getIndicatorIndex(valk, valk2);
						int weightIdx = totalUnaryLen + (computedIdx) * 4 + indicatorIdx;
						
						binaryScores[computedIdx][indicatorIdx] = wv.get(weightIdx) * 1.0;
					}
				}
			}
		}
	}
	
	
	double computeDotProductUiuc(ArrayList<Double> feats, edu.illinois.cs.cogcomp.sl.util.WeightVector wv, int offset) {
		double result = 0;
		for (int i = 0; i < feats.size(); i++) {
			result += (feats.get(i) * wv.get(i + offset));
		}
		return result;
	}
	
	public void loadNewWeightUiuc(Example ex, edu.illinois.cs.cogcomp.sl.util.WeightVector wvUiuc) {
		init(ex);
		
		// unary
		for (int i = 0; i < labelDim; i++) {
			// value = 0
			unaryScores[i][0] = 0;
			/////////////////////////////////////
			// value = 1
			unaryScores[i][1] = computeDotProductUiuc(ex.getFeat(), wvUiuc, (i * featDim + 1));
			
			//System.out.println("unaryScores["+i+"][0] = " + unaryScores[i][0]);
			//System.out.println("unaryScores["+i+"][1] = " + unaryScores[i][1]);
		}
		
		// binary
		for (int k = 0; k < (labelDim - 1); k++) {
			for (int k2 = (k + 1); k2 < labelDim; k2++) {
				
				int computedIdx = Featurizer.computePairIndex(k, k2, labelDim);
				for (int valk = 0; valk < 2; valk++) {
					for (int valk2 = 0; valk2 < 2; valk2++) {
						int indicatorIdx = Featurizer.getIndicatorIndex(valk, valk2);
						int weightIdx = totalUnaryLen + (computedIdx) * 4 + indicatorIdx;
						
						binaryScores[computedIdx][indicatorIdx] = wvUiuc.get(weightIdx + 1) * 1.0;
						//System.out.println("binaryScores["+computedIdx+"]["+indicatorIdx+"] = " + binaryScores[computedIdx][indicatorIdx]);
					}
				}
			}
		}
	}
	/*
	void setScores(int offset) {
		
	}*/
	
	public double getUnaryScore(int i, int ival) {
		return unaryScores[i][ival];
	}
	
	public double getBinaryScore(int i, int j, int ival, int jval) {
		
		int pairIdx = Featurizer.computePairIndex(i, j, labelDim);
		int indicatorIdx = Featurizer.getIndicatorIndex(ival, jval);
		
		//System.out.println(pairIdx + " " + indicatorIdx);
		double sc = binaryScores[pairIdx][indicatorIdx];
		return sc;
	}

	
	public double getCost(OldSearchState newState, Example exmp) {
		
		double result = 0;
		
		for (int i = 0; i < labelDim; i++) {
			int ival = newState.getOutput().getValue(i);
			result += getUnaryScore(i, ival);
		}
		
		for (int k = 0; k < (labelDim - 1); k++) {
			for (int k2 = (k + 1); k2 < labelDim; k2++) {
				int valk1 = newState.getOutput().getValue(k);
				int valk2 = newState.getOutput().getValue(k2);
				result += getBinaryScore(k, k2, valk1, valk2);
			}
		}
		
		return result;
	}
	
}
