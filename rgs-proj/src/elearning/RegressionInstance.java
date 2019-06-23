package elearning;

import java.util.Arrays;
import java.util.HashMap;

public class RegressionInstance {
	HashMap<Integer, Double> sparseFeat;
	double value;
	double weight;
	
	public RegressionInstance(HashMap<Integer, Double> feat, double v) {
		sparseFeat = feat;
		value = v;
		weight = 1;
	}
	
	public RegressionInstance(HashMap<Integer, Double> feat, double v, double dpWgt) {
		sparseFeat = feat;
		value = v;
		weight = dpWgt;
	}
	
	public String toArffSparseVecStr(int featLen) {
		
		StringBuilder sb = new StringBuilder();
		
		Integer[] idxs = sparseFeat.keySet().toArray(new Integer[0]);
		Arrays.sort(idxs);
		
		sb.append("{ ");
		
		for (int j = 0; j < idxs.length; j++) {
			int idx = idxs[j].intValue();
			double val = sparseFeat.get(idx);
			
			assert (!Double.isNaN(val));
		
			//System.out.print(idx + " " + val + ",");
			sb.append(idx + " " + val + ",");
		}
		sb.append(featLen + " " + value);
		
		sb.append(" }");
		
		// instance weight
		sb.append(", { " +weight  + " }");
		
		return sb.toString();
	}
}
