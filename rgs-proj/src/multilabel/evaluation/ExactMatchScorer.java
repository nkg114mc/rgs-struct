package multilabel.evaluation;

import java.util.List;

import general.AbstractOutput;
import multilabel.instance.Example;
import multilabel.learning.StructOutput;
import multilabel.utils.UtilFunctions;
import sequence.hw.HwInstance;


public class ExactMatchScorer extends Scorer {

	@Override
	public double getAccuracy(Example ex, StructOutput output) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double getLoss(Example ex, StructOutput output) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double getAccuracyBatch(List<Example> exs) {
		double totalCrr = 0;
		double totalSize = 0;
		for (Example ex : exs) {
			totalSize += 1;
			StructOutput groudtruth = UtilFunctions.getGoldOutputFromExmp(ex);
			if (ex.predictOutput.isEqual(groudtruth)) {
			//if (groudtruth.isEqual(groudtruth)) {
				totalCrr += 1;
			}
		}
		double sc = totalCrr / totalSize;
		return sc;
	}

	@Override
	public String name() {
		return "ExactMatch";
	}

	@Override
	public double getAccuracyBatchHw(List<HwInstance> ins) {
		double totalCrr = 0;
		double totalSize = 0;
		for (HwInstance ex : ins) {
			totalSize += 1;
			AbstractOutput predict = ex.getPredict();
			AbstractOutput groudtruth = ex.getGoldOutput();
			if (predict.isEqual(groudtruth)) {
				totalCrr += 1;
			}
		}
		double sc = totalCrr / totalSize;
		return sc;
	}

}
