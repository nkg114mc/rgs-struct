package general;

import java.util.HashMap;

import edu.illinois.cs.cogcomp.sl.core.AbstractFeatureGenerator;


public abstract class AbstractFeaturizer  extends AbstractFeatureGenerator {

	private static final long serialVersionUID = 8398488311686751004L;
	
	public abstract HashMap<Integer, Double> featurize(AbstractInstance x, AbstractOutput y);
	
	public abstract int getFeatLen();

}
