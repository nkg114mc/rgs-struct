package essvm;

import java.util.Arrays;
import java.util.HashMap;

import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import general.AbstractInstance;
import general.AbstractOutput;
import multilabel.MultiLabelFeaturizer;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;

public class MultiLabelFactorizedWeight {
	
	int unaryCnt;
	WeightVector originalWeight;
	HwInstance ins;
	MultiLabelFactorizedFeatureGenerator lgenr;
	
	public float[] unaryScores;

	public MultiLabelFactorizedWeight(HwInstance inst, WeightVector w, MultiLabelFactorizedFeatureGenerator lbffGener) {
		init(inst, w, lbffGener);
	}
	
	public void init(HwInstance inst, WeightVector w, MultiLabelFactorizedFeatureGenerator lbffGener) {
		originalWeight = w;
		ins = inst;
		lgenr = lbffGener;
		
		unaryScores = new float[inst.size()];
		Arrays.fill(unaryScores, 0);
		
		//////////////////////
		
		double[] commonMlFeat = inst.getUnaryFeats(0);
		HashMap<Integer, Double> commonMlFeatMap = MultiLabelFeaturizer.arrToMap(commonMlFeat);
		
		for (int i = 0; i < inst.size(); i++) {
			unaryScores[i] = lbffGener.scoringUnary(commonMlFeatMap, i, w);
		}
	}
	
	
	public float doProduct(MultiLabelFactorizedFeature mlff) {
		
		float res = 0;
		
		// unary score
		/*
		for (Integer uidx : mlff.unaryValues.keySet()) {
			res += (  (mlff.unaryValues.get(uidx).floatValue()) * unaryScores[uidx.intValue()]  );
		}
		*/
		double[] commonMlFeat = ins.getUnaryFeats(0);
		HashMap<Integer, Double> commonMlFeatMap = MultiLabelFeaturizer.arrToMap(commonMlFeat);
		
		for (Integer uidx : mlff.unaryValues.keySet()) {
			res += ( lgenr.scoringUnary(commonMlFeatMap, uidx, originalWeight) * mlff.unaryValues.get(uidx).floatValue() );
		}
		

		/*
		for (int i = 0; i < x.size(); i++) {
			double[] feat = commonMlFeat;//x.getUnaryFeats(i);
			if (y.getOutput(i) > 0) {
				for (Integer j2 : commonMlFeatMap.keySet()) {
					int idx = getCompatibleFeatIndex(i, j2.intValue());
					sparseValues.put(idx, feat[j2.intValue()]);
				}
			}
		}
		*/
		if (lgenr.getConsiderPair()) {
			// pair score
			for (Integer pidx : mlff.pairValues.keySet()) {
				float f2 = (mlff.pairValues.get(pidx).floatValue());
				float wi = originalWeight.get(pidx.intValue());
				res += ( f2 * wi );
			}
		}

		return res;
	}
	
	
	public float doProductDebug(MultiLabelFactorizedFeature mlff, AbstractInstance xi, AbstractOutput yi1, AbstractOutput yi2) {
		
		HwOutput y1 = (HwOutput)yi1; 
		HwOutput y2 = (HwOutput)yi2;
		
		float res = 0;
		
		// unary score
		double[] commonMlFeat = ins.getUnaryFeats(0);
		HashMap<Integer, Double> commonMlFeatMap = MultiLabelFeaturizer.arrToMap(commonMlFeat);
		
		for (Integer uidx : mlff.unaryValues.keySet()) {
			float sign = (float)(y1.getOutput(uidx) - y2.getOutput(uidx));
			for (Integer j2 : commonMlFeatMap.keySet()) {
				int idx = lgenr.mlfeaturizer.getCompatibleFeatIndex(uidx, j2.intValue());
				
				float f1 = commonMlFeatMap.get(j2).floatValue() * sign;
				float wj = originalWeight.get(idx);
				res += ( f1 * wj);
			}
		}
		
		if (lgenr.getConsiderPair()) {
			// pair score
			for (Integer pidx : mlff.pairValues.keySet()) {
				float f2 = (mlff.pairValues.get(pidx).floatValue());
				float wi = originalWeight.get(pidx.intValue());
				res += ( f2 * wi );
			}
		}

		return res;
	}
	
	
	public void addFactorizedFeatureVector(MultiLabelFactorizedFeature mlff, float multplyFactor) {
		// only update unary
		for (Integer uidx : mlff.unaryValues.keySet()) {
			unaryScores[uidx.intValue()] += (mlff.unaryValues.get(uidx).floatValue() * multplyFactor);
		}
	}
}
