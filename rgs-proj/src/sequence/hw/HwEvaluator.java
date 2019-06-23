package sequence.hw;

import java.util.List;

public class HwEvaluator {

	
	public static void evaluateBatch(List<HwInstance> instances, List<HwOutput> golds, List<HwOutput> predicts ) {
	
		double total = 0;
		double crr = 0;
		for (int i = 0; i < instances.size(); i++) {
			HwOutput gld = golds.get(i);
			HwOutput prd = predicts.get(i);
			if (gld.size() != prd.size()) {
				throw new RuntimeException("Gold predict output size inconsistent! " + golds.get(i).size() + "!="+ predicts.get(i).size());
			}
			int n = golds.get(i).size();
			for (int j = 0; j < n; j++) {
				total += 1.0;
				if (gld.output[j] != prd.output[j]) {
					crr += 1.0;
				}
			}
		}
		
		double acc = crr / total;
		System.out.println("== Evaluation ==");
		System.out.println("Hamming Accuracy: " + crr + " / " + total + " = " + acc);
		
	}
	

}
