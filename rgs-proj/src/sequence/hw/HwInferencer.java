package sequence.hw;

import java.util.ArrayList;
import java.util.List;

import edu.illinois.cs.cogcomp.sl.core.AbstractInferenceSolver;
import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.util.IFeatureVector;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import general.AbstractOutput;

public class HwInferencer extends AbstractInferenceSolver {
	
	/**
	 */
	private static final long serialVersionUID = 1L;
	
	HwFeaturizer featurizer;
	public HwInferencer(HwFeaturizer fzr) {
		super(); 
		featurizer = fzr;
	}

	
	public void computeCachedSegFeatures(HwInstance inst) {
		
	}

	public IStructure getLossAugmentedBestStructure(WeightVector wv, IInstance input, IStructure gold) {
		
		HwOutput goldLabeledSeq = (HwOutput) gold;
		HwInstance inst = (HwInstance) input;
		HwOutput pred = unaryInferenceOnly(wv, inst, goldLabeledSeq);//new HwOutput(inst.size());
		return (pred);
	}
	
	private HwOutput unaryInferenceOnly(WeightVector wv, HwInstance inst, HwOutput gold) {
		
		HwOutput pred = new HwOutput(inst.size(), inst.alphabet);
		
		for (int i = 0; i < inst.size(); i++) {
			float bestSc = Float.NEGATIVE_INFINITY;
			HwInstance singlex = getSingleSegInstance(inst.letterSegs.get(i), inst.alphabet);
			for (int j = 0; j < inst.alphabet.length; j++) {
				HwOutput singley = new HwOutput(1, inst.alphabet);
				singley.output[0] = j; // value
				//System.out.println(featurizer);
				IFeatureVector fv = featurizer.getFeatureVector(singlex, singley);
				float sc = wv.dotProduct(fv);
				
				float loss = 0;
				if (gold != null) {
					if (j != gold.output[i]) {
						loss = 1.0f;
					}
				}
				
				sc += loss;
				//System.out.println(sc);
				if (sc > bestSc) {
					bestSc = sc;
					pred.output[i] = j;
				}
			}
		}
		
		return pred;
	}
	
	
	public static HwInstance getSingleSegInstance(HwSegment seg, String[] albt) {
		List<HwSegment> hwsegs = new ArrayList<HwSegment>();
		hwsegs.add(seg);
		HwInstance hwinst = new HwInstance(hwsegs, albt);
		return hwinst;
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
				loss += 1.0;
			}
		}
		return loss;
	}

}
