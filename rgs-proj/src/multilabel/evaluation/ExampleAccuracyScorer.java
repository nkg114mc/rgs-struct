package multilabel.evaluation;

import java.util.List;

import general.AbstractOutput;
import multilabel.utils.UtilFunctions;
import sequence.hw.HwInstance;
import multilabel.learning.StructOutput;
import multilabel.instance.Example;

public class ExampleAccuracyScorer extends Scorer {

	@Override
	public double getAccuracy(Example ex, StructOutput output) {
		StructOutput groudtruth = UtilFunctions.getGoldOutputFromExmp(ex);
		int[] crrWng = LossFunction.computeTFPN(output, groudtruth);
		double enu = crrWng[0];
		double den = (crrWng[0] + crrWng[2] + crrWng[3]);
		double sc = enu / den;
		if (Double.isNaN(sc)) sc = 0; 
		return sc;
	}

	@Override
	public double getLoss(Example ex, StructOutput output) {
		return 0;
	}

	@Override
	public double getAccuracyBatch(List<Example> exs) {
		double totalacc = 0;
		double totalSize = 0;
		for (Example ex : exs) {
			totalSize += 1;
			totalacc += getAccuracy(ex, ex.predictOutput);
		}
		double sc = totalacc / totalSize;
		return sc;
	}

	@Override
	public String name() {
		return "ExampleAcc";
	}

	
	
	
	
	
	public double getAccuracyHw(HwInstance ex, AbstractOutput output) {
		AbstractOutput groudtruth = ex.getGoldOutput();
		int[] crrWng = LossFunction.computeTFPNhw(output, groudtruth);
		double enu = crrWng[0];
		double den = (crrWng[0] + crrWng[2] + crrWng[3]);
		double sc = enu / den;
		if (Double.isNaN(sc)) sc = 0; 
		return sc;
	}

	@Override
	public double getAccuracyBatchHw(List<HwInstance> ins) {
		double totalacc = 0;
		double totalSize = 0;
		for (HwInstance ex : ins) {
			totalSize += 1;
			totalacc += getAccuracyHw(ex, ex.getPredict());
		}
		double sc = totalacc / totalSize;
		return sc;
	}

}
