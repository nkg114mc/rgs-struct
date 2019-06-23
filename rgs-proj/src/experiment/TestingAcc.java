package experiment;

import search.loss.LossScore;

public class TestingAcc extends LossScore {

	double value;
	String name;
	
	public double variance = 0;
	
	public TestingAcc(String nm, double v) {
		value = v;
		name = nm;
	}
	
	public double getVal() {
		return value;
	}

	public LossScore addWith(LossScore sc2) {
		throw new RuntimeException("Should not use this method!");
	}

	public String getStr() {
		return name;
	}

	public LossScore getSelfCopy() {
		return new TestingAcc(name, value);
	}

}
