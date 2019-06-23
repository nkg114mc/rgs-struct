package multilabel.learning.structsvm;

import multilabel.learning.StructOutput;
import multilabel.instance.Featurizer;
import edu.illinois.cs.cogcomp.sl.core.AbstractFeatureGenerator;
import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.util.FeatureVectorBuffer;
import edu.illinois.cs.cogcomp.sl.util.IFeatureVector;

public class StructSvmFeatGen extends AbstractFeatureGenerator  {

	private static final long serialVersionUID = -2319479743180652851L;
	
	
	Featurizer featzer;
		
	public StructSvmFeatGen() {
		featzer = new Featurizer();
	}
	
	@Override
	public IFeatureVector getFeatureVector(IInstance xi, IStructure yi) {

		StrucSvmInstance x = (StrucSvmInstance) xi;
		StructOutput y = (StructOutput) yi;

		multilabel.instance.OldWeightVector fv = featzer.getFeatureVector(x.example, y);

		FeatureVectorBuffer fb = new FeatureVectorBuffer();
		for (int idx : fv.getKeys()) {
			fb.addFeature(idx + 1, fv.get(idx));
		}
		return fb.toFeatureVector();
	}

	@Override
	public IFeatureVector getFeatureVectorDiff(IInstance x, IStructure y1, IStructure y2) {
		IFeatureVector f1 = getFeatureVector(x, y1);
		IFeatureVector f2 = getFeatureVector(x, y2);		
		return f1.difference(f2);
	}
}
