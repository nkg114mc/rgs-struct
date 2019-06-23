package sequence.twitterpos;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.util.IFeatureVector;
import general.AbstractInstance;
import general.AbstractOutput;
import init.SegHashFeatVecWrapper;
import sequence.hw.HwInstance;
import sequence.hw.HwSegment;

public class TwitterPosExample extends HwInstance {

	public TwitterPosExample(List<HwSegment> segs, String[] albt) {
		super(segs, albt);
	}

}