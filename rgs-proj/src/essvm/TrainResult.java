package essvm;

import java.util.ArrayList;

import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import multilabel.utils.WeightDumper;

public class TrainResult {
	
	public ArrayList<TrainSnapshot> snapshots;
	
	public long totalMilSeconds;
	
	public int iterNum;
	
	public WeightDumper wdumper = null; // default
	
	public TrainResult(WeightDumper dpr) {
		snapshots = new ArrayList<TrainSnapshot>();
		wdumper = dpr;
	}
	
	public void addSnapshot(int itr, long tim, WeightVector w) {
		TrainSnapshot ss = new TrainSnapshot();
		
		ss.iter = itr;
		ss.time = tim;
		if (wdumper != null) {
			ss.weightFilePath = wdumper.dumpWeight(itr, w);
		} else {
			ss.setCostWeight(w);
		}
		
		
		snapshots.add(ss);
	}

	
	public void printTrainResult() {
		System.out.println("Iteration:   " + iterNum);
		System.out.println("Time(milsec):" + totalMilSeconds);
	}
}
