package multilabel.instance;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

// more useful in sparse format
public class OldWeightVector {
	
	int maxLength;
	HashMap<Integer, Double> valueMap;
	
	public OldWeightVector() {
		valueMap = new HashMap<Integer, Double>();
		maxLength = 0;
	}
	
	public Set<Integer> getKeys() {
		return valueMap.keySet();
	}
	
	public double get(int idx) {
		Double vd = valueMap.get(idx);
		if (vd == null) {
			return 0;
		}
		return vd.doubleValue();
	}
	
	public int size() {
		return (valueMap.size());
	}

	
	public void put(int idx, double val) {
		if (val != 0) {
			valueMap.put(idx, val);
		}
	}
	
	public boolean containsKey(int idx) {
		return (valueMap.containsKey(idx));
	}
	
	
	public String toSparseRanklibStr() {
		String result = "";
		ArrayList<Integer> indices = new ArrayList<Integer>(valueMap.keySet());
		Collections.sort(indices);
		for (int i = 0; i < indices.size(); i++) {
			int idx = indices.get(i);
			double val = valueMap.get(idx);
			
			if (!result.equals("")) {
				result += " ";
			}
			result += (String.valueOf(idx + 1) + ":" + String.valueOf(val));
		}
		return result;
	}
	
	public String toDenseRanklibStr() {
		return toDenseRanklibStr(maxLength);
	}
	
	public String toDenseRanklibStr(int length) {
		String result = "";
		ArrayList<Integer> indices = new ArrayList<Integer>(valueMap.keySet());
		Collections.sort(indices);
		HashSet<Integer> existIdx = new HashSet<Integer>(indices);
		for (int idx = 0; idx < length; idx++) {
			
			double val = 0;
			if (existIdx.contains(idx)) {
				val = valueMap.get(idx);
			}
			
			if (!result.equals("")) {
				result += " ";
			}
			result += (String.valueOf(idx + 1) + ":" + String.valueOf(val));
		}
		return result;
	}
	
	
	public int getMaxLength() {
		return maxLength;
	}
	public void setMaxLength(int l) {
		maxLength = l;
	}
}
