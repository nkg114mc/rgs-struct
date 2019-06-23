package essvm;

import edu.illinois.cs.cogcomp.sl.util.IFeatureVector;

public class SSVMUpdateItem {
	
	public double loss = Double.NEGATIVE_INFINITY;
	public double predDiff = Double.NEGATIVE_INFINITY;
	
	public double predScore = Double.NEGATIVE_INFINITY;
	public double goldScore = Double.NEGATIVE_INFINITY;
	
	public IFeatureVector goldFeatures = null; 
	public IFeatureVector predictedFeatures = null;
	public IFeatureVector update = null;
	
}
