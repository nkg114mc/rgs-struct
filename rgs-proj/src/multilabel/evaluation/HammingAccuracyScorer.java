package multilabel.evaluation;

import java.util.List;

import general.AbstractOutput;
import multilabel.instance.Example;
import multilabel.learning.StructOutput;
import multilabel.utils.UtilFunctions;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;

public class HammingAccuracyScorer extends Scorer {

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
		double totalWng = 0;
		double totalSize = 0;
		for (Example ex : exs) {
			totalSize += ((double)(ex.labelDim()));
			StructOutput groudtruth = UtilFunctions.getGoldOutputFromExmp(ex);
			//int[] crrwng = LossFunction.computeTFPN(groudtruth, groudtruth);//(ex.predictOutput, groudtruth);
			int[] crrwng = LossFunction.computeRightWrong(ex.predictOutput, groudtruth);
			totalCrr += crrwng[0];
			totalWng += crrwng[1];
		}
		
		double sc = totalCrr / totalSize;
		return sc;
	}

	@Override
	public String name() {
		return "HammingAccuracy";
	}

	@Override
	public double getAccuracyBatchHw(List<HwInstance> ins) {
		double totalCrr = 0;
		double totalWng = 0;
		double totalSize = 0;
		for (HwInstance ex : ins) {
			totalSize += ((double)(ex.size()));
			AbstractOutput groudtruth = ex.getGoldOutput();
			int[] crrwng = LossFunction.computeRightWrongHw(ex.getPredict(), groudtruth);
			totalCrr += crrwng[0];
			totalWng += crrwng[1];
		}
		
		double sc = totalCrr / totalSize;
		return sc;
	}

}
