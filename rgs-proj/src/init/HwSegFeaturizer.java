package init;

import java.util.HashMap;

import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.util.IFeatureVector;
import general.AbstractFeaturizer;
import general.AbstractInstance;
import general.AbstractOutput;
import sequence.hw.HwFeaturizer;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;

public class HwSegFeaturizer extends AbstractFeaturizer { 
	
	private static final long serialVersionUID = 2600327382486754166L;
	
	public HwFeaturizer featurizer;
	
	public HwSegFeaturizer(String[] albt, int sigflen, boolean pairwise, boolean trip, boolean quad) {
		featurizer = new HwFeaturizer(albt, sigflen, false, false, false);
	}
	
	public int getFeatLen() {
		return featurizer.getFeatLen();
	}
	
	public HwFeaturizer getFeaturizer() {
		return featurizer;
	}

	@Override
	public IFeatureVector getFeatureVector(IInstance xi, IStructure yi) {
		HwInstance inst = (HwInstance)(xi);
		if (inst.cachedFeatVec == null) {
			inst.cachedFeatVec = new IFeatureVector[inst.alphabet.length];
			for (int j = 0; j < inst.alphabet.length; j++) {
				HwOutput singley = new HwOutput(1, inst.alphabet);
				singley.output[0] = j;
				inst.cachedFeatVec[j] = featurizer.getFeatureVector(inst, singley);
			}
		}
		HwOutput y = (HwOutput)yi;
		return inst.cachedFeatVec[y.getOutput(0)];
	}
	
	// phi(x,y)
	@Override
	public HashMap<Integer, Double> featurize(AbstractInstance xi, AbstractOutput yi) {
		
		HwInstance x = (HwInstance)xi; 
		HwOutput y = (HwOutput)yi; 
		
		HwInstance inst = (HwInstance)(xi);
		if (inst.cachedHashVec == null) {
			inst.cachedHashVec = new SegHashFeatVecWrapper[inst.alphabet.length];
			for (int j = 0; j < inst.alphabet.length; j++) {
				HwOutput singley = new HwOutput(1, inst.alphabet);
				singley.output[0] = j;
				SegHashFeatVecWrapper hashVec = new SegHashFeatVecWrapper(featurizer.featurize(inst, singley));
				inst.cachedHashVec[j] = hashVec;
			}
		}
		return inst.cachedHashVec[y.getOutput(0)].sparseFeature;
	}
	
	@Override
	public IFeatureVector getFeatureVectorDiff(IInstance x, IStructure y1, IStructure y2) {
		IFeatureVector f1 = getFeatureVector(x, y1);
		IFeatureVector f2 = getFeatureVector(x, y2);		
		return f1.difference(f2);
	}

}
