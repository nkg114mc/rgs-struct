package essvm;

import edu.illinois.cs.cogcomp.sl.util.IFeatureVector;

// work set items
public class SSVMConstraint {
	
	/*
	public double loss = Double.NEGATIVE_INFINITY;
	public double predDiff = Double.NEGATIVE_INFINITY;
	
	public double predScore = Double.NEGATIVE_INFINITY;
	public double goldScore = Double.NEGATIVE_INFINITY;
	
	public IFeatureVector goldFeatures = null; 
	public IFeatureVector predictedFeatures = null;
	*/
	
	int hashCd;
	
	public SSVMConstraint(int hscd) {
		hashCd = hscd;
	}
	
	//public int hashCode() {
	//	return hashCd;
	//}
	
	
	public double loss = Double.NEGATIVE_INFINITY;
	public IFeatureVector goldMinusPredFeatures = null;
	
}
