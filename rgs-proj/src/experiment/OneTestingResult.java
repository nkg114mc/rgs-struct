package experiment;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

public class OneTestingResult {

	public ArrayList<TestingAcc> testingScores = new ArrayList<TestingAcc>();
	
	public void addScore(TestingAcc sc) {
		testingScores.add(sc);
	}
	
	public int getSize() {
		return testingScores.size();
	}
	
	////////////////////////////////////////
	
	public static void computeAverage(List<OneTestingResult> results) {
		
		
		int n = 0;
		HashSet<Integer> sizeset = new HashSet<Integer>();
		for (int i = 0; i < results.size(); i++) {
			n = results.get(i).getSize();
			sizeset.add(results.get(i).getSize());
		}
		
		if (sizeset.size() != 1) {
			throw new RuntimeException("More than one size: " + sizeset.size());
		}
		
		/////////////////////////////////////////////////
		
		OneTestingResult avgResult = new OneTestingResult();
		
		for (int i = 0; i < n; i++) {
			String scName = "????";
			ArrayList<Double> scs = new ArrayList<Double>();
			HashSet<String> nms = new HashSet<String>();
			for (int j = 0; j < results.size(); j++) {
				OneTestingResult re = results.get(j);
				TestingAcc sc = re.testingScores.get(i);
				scs.add(sc.getVal());
				nms.add(sc.getStr());
				scName = sc.getStr();
			}
			
			if (nms.size() != 1) {
				throw new RuntimeException("Name inconsistent: " + nms.size());
			}
			
			
			double mean = computeMean(scs);
			double variance = computeVariance(scs);
			
			
			TestingAcc avgSc = new TestingAcc(scName, mean);
			avgSc.variance = variance;
			
			avgResult.testingScores.add(avgSc);
			
			String str = getStringDump(scName, mean, variance, scs);
			System.out.println("Score" + i + ", " + str);
		}
		
	}
	
	public static double computeMean(ArrayList<Double> scs) {
		double sum = 0;
		for (Double sc : scs) {
			sum += sc.doubleValue();
		}
		double den = (double)scs.size();
		double avg = sum / den;
		return avg;
	}
	
	public static double computeVariance(ArrayList<Double> scs) {
		double mean = computeMean(scs);
		double diffsum = 0;
		for (Double sc : scs) {
			diffsum += ((sc.doubleValue() - mean) * (sc.doubleValue() - mean));
		}
		double den = 1;
		if (scs.size() > 1) den = (double)(scs.size() - 1);
		double vari = diffsum / den;
		return vari;
	}
	
	public static String getStringDump(String nm, double mean, double variance, ArrayList<Double> scs) {
		String str = nm + ", " + mean + "," + variance + ",  " + String.valueOf(scs);
		return str;
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
