package search.loss;

public class LossScoreHamm extends LossScore {

	double correctCnt;
	public double totalCnt;
	
	public LossScoreHamm(double v, double t) {
		correctCnt = v;
		totalCnt = t;
	}
	
	@Override
	public double getVal() {
		return correctCnt;
	}

	@Override
	public LossScore addWith(LossScore sc) {
		LossScoreHamm sc2 = (LossScoreHamm)(sc);
		double sumCrr = correctCnt + sc2.getVal();
		if (sc2.totalCnt != totalCnt) {
			throw new RuntimeException("Total value of Hamming score is not same: " + sc2.totalCnt + " " + totalCnt);
		}
		
		return (new LossScoreHamm(sumCrr, sc2.totalCnt));
	}

	@Override
	public String getStr() {
		return String.valueOf(correctCnt);
	}

	@Override
	public LossScore getSelfCopy() {
		LossScoreHamm cp = new LossScoreHamm(correctCnt, totalCnt);
		return cp;
	}
}
