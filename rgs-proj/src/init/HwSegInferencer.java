package init;

import java.util.ArrayList;
import java.util.List;

import edu.illinois.cs.cogcomp.sl.core.AbstractInferenceSolver;
import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.util.IFeatureVector;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import general.AbstractOutput;
import sequence.hw.HwFeaturizer;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;
import sequence.hw.HwSegment;

public class HwSegInferencer extends AbstractInferenceSolver {
	
	/**
	 * Only works for single segment instance
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	HwFeaturizer featurizer;
	public HwSegInferencer(HwFeaturizer fzr) {
		super(); 
		featurizer = fzr;
	}

	
	public IFeatureVector computeCachedSegFeatures(HwInstance inst, int yIdx) {
		if (inst.cachedFeatVec == null) {
			inst.cachedFeatVec = new IFeatureVector[inst.alphabet.length];
			for (int j = 0; j < inst.alphabet.length; j++) {
				HwOutput singley = new HwOutput(1, inst.alphabet);
				singley.output[0] = j;
				inst.cachedFeatVec[j] = featurizer.getFeatureVector(inst, singley);
			}
		}
		return inst.cachedFeatVec[yIdx];
	}

	public IStructure getLossAugmentedBestStructure(WeightVector wv, IInstance input, IStructure gold) {
		
		HwOutput goldLabeledSeq = (HwOutput) gold;
		HwInstance inst = (HwInstance) input;
		HwOutput pred = unaryInferenceOnly(wv, inst, goldLabeledSeq);//new HwOutput(inst.size());
		return (pred);
	}
	
	private HwOutput unaryInferenceOnly(WeightVector wv, HwInstance inst, HwOutput gold) {
		
		assert (inst.size() == 1);
		
		HwOutput pred = new HwOutput(inst.size(), inst.alphabet);
		
		float bestSc = Float.NEGATIVE_INFINITY;
		for (int j = 0; j < inst.alphabet.length; j++) {
			IFeatureVector fv = computeCachedSegFeatures(inst, j);
			float sc = wv.dotProduct(fv);

			float loss = 0;
			if (gold != null) {
				if (j != gold.output[0]) {
					loss = 1.0f;
				}
			}

			sc += loss;
			if (sc > bestSc) {
				bestSc = sc;
				pred.output[0] = j;
			}
		}
		
		return pred;
	}
	
/*	
	public static HwInstance getSingleSegInstance(HwSegment seg, String[] albt) {
		List<HwSegment> hwsegs = new ArrayList<HwSegment>();
		hwsegs.add(seg);
		HwInstance hwinst = new HwInstance(hwsegs, albt);
		return hwinst;
	}
*/

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
				loss += 1.0;
			}
		}
		return loss;
	}

}
