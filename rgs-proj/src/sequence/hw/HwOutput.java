package sequence.hw;

import java.util.HashMap;
import java.util.List;

import edu.illinois.cs.cogcomp.sl.core.IStructure;
import general.AbstractOutput;
import search.SearchState;
import search.ZobristKeys;

public class HwOutput implements AbstractOutput, IStructure {
	
	
	public static final ZobristKeys selfZobrist = new ZobristKeys(4096, 1024);
	public static int computeHashKey(HwOutput outp) {
		int[][] zobristHashKeys = selfZobrist.getZbKeys();
		int result = 0;
		for (int i = 0; i < outp.size(); i++) {
			int j = outp.getOutput(i);
			result = (result ^ zobristHashKeys[i][j]);
		}
		return result;
	}
	
	////////////////////////////////////////////
	
	public int[] output = null;
	
	private String[] alphabet;
	
	public HwOutput(int sz, String[] lbs) {
		output = new int[sz];
		alphabet = lbs;
	}

	public int size() {
		return output.length;
	}

	@Override
	public int tagSize() {
		return alphabet.length;
	}

	@Override
	public int getOutput(int slotIdx) {
		return output[slotIdx];
	}

	@Override
	public void setOutput(int slotIdx, int slotValue) {
		output[slotIdx] = slotValue;
	}

	@Override
	public AbstractOutput copyFrom(AbstractOutput srci) {
		HwOutput src = (HwOutput) srci;
		HwOutput newOut = new HwOutput(src.size(), alphabet);
		for (int i = 0; i < newOut.size(); i++) {
			newOut.setOutput(i, src.getOutput(i));
		}
		return newOut;
	}

	public String toString() {
		String str = "";
		for (int i = 0; i < output.length; i++) {
			str += alphabet[output[i]];
		}
		return str;
	}
	
	public int hashCode() {
		//String str = toString();
		int key = computeHashKey(this);
		return key;
	}
	
	public boolean equals(Object that) {
		HwOutput thatOut = (HwOutput) that;

		return false;
		
	}
	
	public boolean isEqual(AbstractOutput that) {
		HwOutput thatOut = (HwOutput) that;
		boolean same = true;
		for (int i = 0; i < thatOut.size(); i++) {
			if (this.getOutput(i) != thatOut.getOutput(i)) {
				same = false;
				break;
			}
		}
		return same;
	}
	
/*
	public static HwOutput getGoldOutput(HwInstance inst) {
		List<String> gltrs = inst.getLetterSeq();
		HashMap<String, Integer> LetterToIntMap = HwLabelSet.initMap(inst.alphabet);;
		
		//System.out.println(gltrs);
		HwOutput gst = new HwOutput(gltrs.size(), inst.alphabet);
		//System.out.println("size = " + gst.size());
		for (int i = 0; i < gltrs.size(); i++) {
			Integer vidx = LetterToIntMap.get(gltrs.get(i));
			if (vidx == null) {
				throw new RuntimeException("Can not find value whose name is " + gltrs.get(i));
			}
			gst.output[i] = vidx.intValue();
		}
		return gst;
	}
*/
}
