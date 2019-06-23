package essvm;

import java.util.ArrayList;
import java.util.List;

import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearnnew.SamplingELearning;
import general.AbstractInstance;
import general.AbstractLossFunction;
import general.AbstractOutput;
import search.GreedySearcher;
import search.SearchResult;
import search.SearchState;
import search.SearchTrajectory;
import search.loss.GoldPredPair;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;

public class TrainSnapshot {
	
	// c learning result
	public WeightVector c_weight;
	public String weightFilePath;
	
	// x
	public long time;
	public int  iter;
	
	// y
	public double trainAccuracy;

	
	public void setCostWeight(WeightVector w) {
		// copy
		c_weight = new WeightVector(w);
	}
	
	///////////////////////////////////////////////////////////
	
	public static double computeAcc(WeightVector wght, List<HwInstance> trInsts, GreedySearcher gsearcher, int restarts) {//, AbstractLossFunction lossfunc) {
		
		AbstractLossFunction gloss = gsearcher.getLossFunc();
		
		List<GoldPredPair> results = new ArrayList<GoldPredPair>();
		
		for (int i = 0; i < trInsts.size(); i++) {
			HwInstance inst = trInsts.get(i);
			HwOutput gold = inst.getGoldOutput();
			SearchResult infrRe = gsearcher.runSearchWithRestarts(wght, null, restarts, inst, gold, false);

			GoldPredPair re = new GoldPredPair((AbstractInstance)inst, (AbstractOutput)gold, (AbstractOutput)(infrRe.predState.structOutput));
			results.add(re);
		}
		
		double batchAcc = gloss.computeMacroFromResults(results);
		return batchAcc;
	}

}
