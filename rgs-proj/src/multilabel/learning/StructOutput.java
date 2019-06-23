package multilabel.learning;

import edu.illinois.cs.cogcomp.sl.core.IStructure;
import multilabel.learning.search.OldSearchState;

public class StructOutput implements IStructure {
	
	public static int[] PossibleValues = { 0 , 1 };
	
	
	int[] output = null;
	
	public StructOutput(int len) {
		output = new int[len];
	}
	
	public int size() {
		return output.length;
	}
	
	public void setAll(int[] values) {
		if (values.length != output.length) {
			throw new RuntimeException("Value arraies are not in the same length: " + values.length + " != " +output.length);
		}
		System.arraycopy(values, 0, output, 0, values.length);
	}
	
	public void setValue(int idx, int value) {
		output[idx] = value;
	}
	
	public int[] getAll() {
		return output;
	}
	
	public int getValue(int idx) {
		return output[idx];
	}
	
	
	public boolean isEqual(StructOutput that) {
		if (this.size() != that.size()) {
			throw new RuntimeException("Not equal length to compare!");//return false;
		}
		for (int i = 0; i < this.size(); i++) {
			if (this.getValue(i) != that.getValue(i)) {
				return false;
			}
		}
		return true;
	}
	
	public int getOneCount() {
		int cnt = 0;
		for (int i = 0; i < output.length; i++) {
			if (output[i] > 0) { cnt++; }
		}
		return cnt;
	}
	
	public static void copyOutput(StructOutput src, StructOutput des) {
		if (src.size() != des.size()) {
			throw new RuntimeException("Error on copy: Not in the same length: " + src.size() + " != " + des.size());
		}
		int[] srcArr = src.getAll();
		int[] desArr = des.getAll();
		System.arraycopy(srcArr, 0, desArr, 0, desArr.length);
	}
	
	public String toString() {
		String str = "(";
		for (int i = 0; i < output.length; i++) {
			if (i > 0) str += " ";
			str += String.valueOf(output[i]);
		}
		str += ")";
		return str;
	}
}