package sequence.hw;

import java.util.ArrayList;
import java.util.Arrays;

public class HwSegment {
	
	public int index;
	public String letter;
	
	// for logistic regression learning
	public int goldIndex;
	
	////////// feature
	private boolean isDense = true; // default
	// dense feature
	private double[] imgDblArr;
	// sparse feature
	private int      sparseMaxLen = -1;
	private int[]    nonzeroIdx = null;
	private double[] nonzeroVal = null;

	public HwSegment(String lineStr) {
		parse(lineStr);
		
		nonzeroIdx = null;
		nonzeroVal = null;
		sparseMaxLen = 0;
	}
	
	public HwSegment(int i, String feat, String c) {
		index = i;
		imgDblArr = getSegmentFeature(feat);
		letter = c;
		
		nonzeroIdx = null;
		nonzeroVal = null;
		sparseMaxLen = -1;
	}
	
	public HwSegment(int i, double[] featVec, String c) {
		this(i, featVec, c, true);
	}
	
	// for sparse feature construction
	public HwSegment(int i, double[] featVec, String c, boolean asDense) {
		isDense = asDense;
		if (asDense) {
			
			index = i;
			imgDblArr = featVec;
			letter = c;
			
			nonzeroIdx = null;
			nonzeroVal = null;
			sparseMaxLen = -1;
			
		} else {
			
			index = i;
			imgDblArr = null;
			letter = c;
			
			sparseMaxLen = featVec.length;
			
			ArrayList<Integer> featIdxs = new ArrayList<Integer>();
			ArrayList<Double> featVals = new ArrayList<Double>();
			for (int i2 = 0; i2 < featVec.length; i2++) {
				if (featVec[i2] != 0) { // non-zero
					featIdxs.add(i2);
					featVals.add(featVec[i2]);
				}
			}
			assert(featIdxs.size() == featVals.size());
			nonzeroIdx = new int[featIdxs.size()];
			nonzeroVal = new double[featVals.size()];
			for (int j = 0; j < nonzeroIdx.length; j++) {
				nonzeroIdx[j] = featIdxs.get(j).intValue();
				nonzeroVal[j] = featVals.get(j).doubleValue();
			}
		}
	}
	
	public double[] getFeatArr() {
		if (isDense) {
			assert(imgDblArr != null);
			return imgDblArr;
		} else {
			
			assert(sparseMaxLen > 0);
			double[] featArr = new double[sparseMaxLen];
			Arrays.fill(featArr, 0);
			
			assert(nonzeroIdx.length == nonzeroVal.length);
			for (int j = 0; j < nonzeroIdx.length; j++) {
				int id = nonzeroIdx[j];
				double v  = nonzeroVal[j];
				assert(v != 0);
				featArr[id] = v;
			}
			
			return featArr;
		}
	}
	
	/////////////////////////////////////////////
	/// Private 
	/////////////////////////////////////////////
	
	private void parse(String lineStr) {
		String[] arr = lineStr.split("\t");
		if (arr.length < 4) {
			throw new RuntimeException("Too few segments for line: " + lineStr);
		}
		index = Integer.parseInt(arr[0]);
		String imgFeature = arr[1].trim();
		imgDblArr = getSegmentFeature(imgFeature);
		letter = arr[2].trim().toLowerCase();
	}

	private double[] getSegmentFeature(String strFeat) {
		double[] result = new double[strFeat.length() - 2];
		for (int i = 2; i < strFeat.length(); i++) {
			result[i - 2] = Double.parseDouble(String.valueOf(strFeat.charAt(i)));
		}
		return result;
	}

}
