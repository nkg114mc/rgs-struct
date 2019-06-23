package sequence.hw;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.util.IFeatureVector;
import general.AbstractInstance;
import general.AbstractOutput;
import init.SegHashFeatVecWrapper;

public class HwInstance implements AbstractInstance, IInstance {
	
	
	private AbstractOutput predict;
	
	public List<HwSegment> letterSegs;
	public String[] alphabet;
	
	@Override
	public int size() {
		return letterSegs.size();
	}
	
	public HwInstance(List<HwSegment> segs, String[] albt) {
		letterSegs = segs; 
		alphabet = albt;
		//factorGraph = new HwFactorGraph(this);
	}
	
	public List<String> getLetterSeq() {
		List<String> ltrs = new ArrayList<String>();
		for (int i = 0; i < letterSegs.size(); i++) {
			ltrs.add(letterSegs.get(i).letter);
		}
		return ltrs;
	}
	
	public double[] getUnaryFeats(int xi) {
		return (letterSegs.get(xi).getFeatArr());
	}

	@Override
	public int domainSize() {
		return alphabet.length;
	}

	// for unary inference only
	public IFeatureVector[] cachedFeatVec = null;
	public SegHashFeatVecWrapper[] cachedHashVec = null;

	@Override
	public HwOutput getGoldOutput() {
		List<String> gltrs = this.getLetterSeq();
		HashMap<String, Integer> LetterToIntMap = HwLabelSet.initMap(this.alphabet);;
		HwOutput gst = new HwOutput(gltrs.size(), this.alphabet);
		for (int i = 0; i < gltrs.size(); i++) {
			Integer vidx = LetterToIntMap.get(gltrs.get(i));
			if (vidx == null) {
				throw new RuntimeException("Can not find value whose name is " + gltrs.get(i));
			}
			gst.output[i] = vidx.intValue();
		}
		return gst;
	}

	@Override
	public AbstractOutput getPredict() {
		return predict;
	}

	@Override
	public void setPredict(AbstractOutput opt) {
		predict = opt;
	}
}
