package experiment;

import java.util.List;

import general.AbstractOutput;

public class ExperimentResult {
	
	public String name;
	
	public String lossName;
	
	
	public double generationAcc;
	public double selectionAcc;
	public double overallAcc;
	
	public AbstractOutput bestCostOutput;
	public AbstractOutput bestLossOutput;
	
	public OneTestingResult testAcc = new OneTestingResult();
	public void addAcc(TestingAcc acc) {
		testAcc.addScore(acc);
	}
	public void addAccBatch(List<TestingAcc> acs) {
		for (int i = 0; i < acs.size(); i++) {
			testAcc.addScore(acs.get(i));
		}
	}
}