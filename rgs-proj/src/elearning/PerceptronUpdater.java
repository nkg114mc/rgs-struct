package elearning;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class PerceptronUpdater {
	
	//private int len;
	private double[] weightSum;
	private double[] weight;
	private double updateCnt;
	
	public double learnRate;
	public double lambda;
	
	public void reset() {
		updateCnt = 0;
		Arrays.fill(weightSum, 0);
		Arrays.fill(weight, 0);
	}
	
	public PerceptronUpdater(int featLen, double lrate, double lbd) {
		
		learnRate = lrate;
		lambda = lbd;
		
		weightSum = new double[featLen];
		weight = new double[featLen];
		reset();
	}
	
	public void runQuickPerceptron(ArrayList<StarHatPair> datas, int featLen) {

		for (StarHatPair example : datas) {

			updateCnt += 1;
			if (updateCnt % 1000 == 0) System.out.println("Update " + updateCnt);

			HashMap<Integer, Double> featGold = example.phi_real;
			HashMap<Integer, Double> featPred = example.phi_end;

			updateWeight(weight, 
					featGold,
					featPred,
					learnRate,
					lambda);
			sumWeight(weightSum, weight);
		}

	}

	public void sumWeight(double[] sum, double[] w) {
		for (int i = 0; i < w.length; i++) {
			sum[i] += w[i];
		}
	}

	public void divdeNumber(double[] w, double deno) {
		for (int i = 0; i < w.length; i++) {
			w[i] = (w[i] / deno);
		}
	}

	public void updateWeight(double[] currentWeight,
							HashMap<Integer, Double> featGold,
							HashMap<Integer, Double> featPred,
							double eta,
							double lambda) {
		double[] gradient = new double[currentWeight.length];
		Arrays.fill(gradient, 0);
		for (Integer i : featGold.keySet()) {
			gradient[i] += (featGold.get(i.intValue()).doubleValue());
		}
		for (Integer j : featPred.keySet()) {
			gradient[j] += (featPred.get(j.intValue()).doubleValue());
		}

		// do L2 Regularization
		//var l1norm = getL1Norm(currentWeight);
		for (int i2 = 0; i2 < currentWeight.length; i2++) {
			//var regularizerNum: Double = Math.max(0, b);
			//var regularizerDen: Double = Math.max(0, b);
			//double reg = 1.0 - (eta * lambda);
			//double curWeightVal = currentWeight[i2] * reg;
			//currentWeight[i2] = curWeightVal + (gradient[i2] * eta);
			currentWeight[i2] += (gradient[i2] * eta);
		}
	}
	
	public double[] getCurAvg() {
		double[] tmpAvg = new double[weightSum.length];
        System.arraycopy(weightSum, 0, tmpAvg, 0, weightSum.length);
        if (updateCnt > 0) {
        	divdeNumber(tmpAvg, (double)updateCnt);
        }
        return tmpAvg;
	}
		
}