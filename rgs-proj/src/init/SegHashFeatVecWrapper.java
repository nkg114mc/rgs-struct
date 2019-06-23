package init;

import java.util.HashMap;

public class SegHashFeatVecWrapper {
	public SegHashFeatVecWrapper(HashMap<Integer, Double> feat) {
		sparseFeature = feat;
	}
	public HashMap<Integer, Double> sparseFeature;
}
