package multilabel.evaluation;

import java.util.List;

import general.AbstractOutput;
import multilabel.instance.Example;
import multilabel.learning.StructOutput;
import multilabel.utils.UtilFunctions;
import sequence.hw.HwInstance;

public class ExampleF1Scorer extends Scorer {

	
	public double computeF1() {
		return 0;
	}
	
	@Override
	public double getAccuracy(Example ex, StructOutput output) {
		StructOutput groudtruth = UtilFunctions.getGoldOutputFromExmp(ex);
		int[] crrWng = LossFunction.computeTFPN(output, groudtruth);
		double tp = crrWng[0];
		double tn = crrWng[1];
		double fp = crrWng[2];
		double fn = crrWng[3];
		/*
		double pre = 0;
		if ((crrWng[0] + crrWng[1]) != 0) pre = crrWng[0] / (crrWng[0] + crrWng[1]);
		double rec = 0;
		if ((crrWng[0] + crrWng[3]) != 0) rec = crrWng[0] / (crrWng[0] + crrWng[3]);
		double sc = 2 / (1 / pre + 1 / rec);
		System.out.println(crrWng[0] + " " + crrWng[1] + " " + crrWng[2] + " " + crrWng[3]);
		System.out.println(pre + " " + rec);
		*/
		double sc = (2 * tp) / (tp + fp + tp + fn);
		if (Double.isNaN(sc)) sc = 0; 
		
		//System.out.println("sc = " + sc);
		return sc;
	}

	@Override
	public double getLoss(Example ex, StructOutput output) {
		return 0;
	}

	@Override
	public double getAccuracyBatch(List<Example> exs) {
		double totalf1 = 0;
		double totalSize = 0;
		
		for (Example ex : exs) {
			totalSize += 1;
			totalf1 += getAccuracy(ex, ex.predictOutput);
		}
		
		double sc = totalf1 / totalSize;
		return sc;
	}

	@Override
	public String name() {
		return "Example-F1";
	}


	
	
	
	
	public double getAccuracyHw(HwInstance ex, AbstractOutput output) {
		AbstractOutput groudtruth = ex.getGoldOutput();
		int[] crrWng = LossFunction.computeTFPNhw(output, groudtruth);
		double tp = crrWng[0];
		double tn = crrWng[1];
		double fp = crrWng[2];
		double fn = crrWng[3];

		double sc = (2 * tp) / (tp + fp + tp + fn);
		if (Double.isNaN(sc)) sc = 0; 
		
		//System.out.println("sc = " + sc);
		return sc;
	}
	
	@Override
	public double getAccuracyBatchHw(List<HwInstance> ins) {
		double totalf1 = 0;
		double totalSize = 0;
		
		for (HwInstance ex : ins) {
			totalSize += 1;
			totalf1 += getAccuracyHw(ex, ex.getPredict());
		}
		
		double sc = totalf1 / totalSize;
		return sc;
	}
	
	

}
