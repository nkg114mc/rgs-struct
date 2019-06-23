package multilabel.evaluation;

import general.AbstractOutput;
import multilabel.learning.StructOutput;

public class LossFunction {
	
	public static int[] computeTFPN(StructOutput pred, StructOutput truth) {
		int tpos = 0;
		int tneg = 0;
		int fpos = 0;
		int fneg = 0;
		for (int i = 0; i < pred.size(); i++) {
			if ((pred.getValue(i) > 0) && (truth.getValue(i) > 0)) {
				tpos++;
			} else if ((pred.getValue(i) == 0) && (truth.getValue(i) == 0)) {
				tneg++;
			} else if ((pred.getValue(i) > 0) && (truth.getValue(i) == 0)) {
				fpos++;
			} else if ((pred.getValue(i) == 0) && (truth.getValue(i) > 0)) {
				fneg++;
			}
		}

		// 0: true pos
		// 1: true neg
		// 2: false pos
		// 3: false neg
		int[] result = new int[4];
		result[0] = tpos;
		result[1] = tneg;
		result[2] = fpos;
		result[3] = fneg;
		return result;
	}
	
	public static int[] computeRightWrong(StructOutput pred, StructOutput truth) {
		int correct = 0;
		int wrong = 0;
		for (int i = 0; i < pred.size(); i++) {
			if (pred.getValue(i) == truth.getValue(i)) {
				correct++;
			} else {
				wrong++;
			}
		}
		int[] result = new int[2];
		result[0] = correct;
		result[1] = wrong;
		return result;
	}
	
	////////////////////////////////////////////
	
	public static int[] computeTFPNhw(AbstractOutput pred, AbstractOutput truth) {
		int tpos = 0;
		int tneg = 0;
		int fpos = 0;
		int fneg = 0;
		for (int i = 0; i < pred.size(); i++) {
			if ((pred.getOutput(i) > 0) && (truth.getOutput(i) > 0)) {
				tpos++;
			} else if ((pred.getOutput(i) == 0) && (truth.getOutput(i) == 0)) {
				tneg++;
			} else if ((pred.getOutput(i) > 0) && (truth.getOutput(i) == 0)) {
				fpos++;
			} else if ((pred.getOutput(i) == 0) && (truth.getOutput(i) > 0)) {
				fneg++;
			}
		}

		// 0: true pos
		// 1: true neg
		// 2: false pos
		// 3: false neg
		int[] result = new int[4];
		result[0] = tpos;
		result[1] = tneg;
		result[2] = fpos;
		result[3] = fneg;
		return result;
	}
	
	public static int[] computeRightWrongHw(AbstractOutput pred, AbstractOutput truth) {
		int correct = 0;
		int wrong = 0;
		for (int i = 0; i < pred.size(); i++) {
			if (pred.getOutput(i) == truth.getOutput(i)) {
				correct++;
			} else {
				wrong++;
			}
		}
		int[] result = new int[2];
		result[0] = correct;
		result[1] = wrong;
		return result;
	}
	
	
	public static double computeHammingLoss(StructOutput pred, StructOutput truth) {
		int[] crrWng = computeRightWrong(pred, truth);
		int correct = crrWng[0];
		int wrong = crrWng[1];
		int total = pred.size();
		double err = ((double)wrong) / ((double)total);
		return err;
	}
	
	public static double computeHammingAccuracy(StructOutput pred, StructOutput truth) {
		double acc = 1.0 - computeHammingLoss(pred, truth);
		return acc;
	}
	
	public static double computeCorrectCnt(StructOutput pred, StructOutput truth) {
		int[] crrWng = computeRightWrong(pred, truth);
		return ((double) (crrWng[0]));
	}

}
