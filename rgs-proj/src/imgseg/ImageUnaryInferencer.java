package imgseg;

import edu.illinois.cs.cogcomp.sl.core.AbstractInferenceSolver;
import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.util.IFeatureVector;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import general.AbstractOutput;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;

public class ImageUnaryInferencer extends AbstractInferenceSolver {
	
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	ImageSegFeaturizer featurizer;
	public ImageUnaryInferencer(ImageSegFeaturizer fzr) {
		super(); 
		featurizer = fzr;
	}

	public IStructure getLossAugmentedBestStructure(WeightVector wv, IInstance input, IStructure gold) {
		
		HwOutput goldLabeledSeq = (HwOutput) gold;
		HwInstance inst = (HwInstance) input;
		HwOutput pred = unaryInferenceOnly(wv, inst, goldLabeledSeq);//new HwOutput(inst.size());
		return (pred);
	}
	
	private HwOutput unaryInferenceOnly(WeightVector wv, HwInstance inst, HwOutput gold) {
		
		String[] labelDomain = featurizer.alphabet;
		
		HwOutput pred = new HwOutput(inst.size(), labelDomain);
		HwOutput predCp = new HwOutput(inst.size(), labelDomain);//inst.alphabet);
		
		for (int i = 0; i < inst.size(); i++) {
			float bestSc = Float.NEGATIVE_INFINITY;
			for (int j = 0; j < labelDomain.length; j++) {
				predCp.setOutput(i, j);
				//System.out.println(featurizer);
				IFeatureVector fv = featurizer.getUnaryFeatureVector(inst, predCp, i);
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
