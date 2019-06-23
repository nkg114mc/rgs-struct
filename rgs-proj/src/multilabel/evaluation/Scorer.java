package multilabel.evaluation;

import java.util.List;

import multilabel.instance.Example;
import multilabel.learning.StructOutput;
import sequence.hw.HwInstance;

public abstract class Scorer {

	/**
	 *  This is a base class for all scoring functions
	 *  Each scorer can return either "loss" or "accuracy", in the scale or 0 to 1
	 */
	
	public abstract String name();
	
	public abstract double getAccuracy(Example ex, StructOutput output);
	public abstract double getLoss(Example ex, StructOutput output);
	
	public abstract double getAccuracyBatch(List<Example> exs);
	public abstract double getAccuracyBatchHw(List<HwInstance> ins);
}
