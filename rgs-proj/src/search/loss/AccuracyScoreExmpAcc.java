package search.loss;

public class AccuracyScoreExmpAcc extends LossScore {
	
	double tp;
	double tn;
	double fp;
	double fn;
	
	public AccuracyScoreExmpAcc(double trpos, double trneg, double fspos, double fsneg) {
		tp = trpos;
		tn = trneg;
		fp = fspos;
		fn = fsneg;
	}
	
	/*
	double pre = 0;
	if ((crrWng[0] + crrWng[1]) != 0) pre = crrWng[0] / (crrWng[0] + crrWng[1]);
	double rec = 0;
	if ((crrWng[0] + crrWng[3]) != 0) rec = crrWng[0] / (crrWng[0] + crrWng[3]);
	double sc = 2 / (1 / pre + 1 / rec);
	System.out.println(crrWng[0] + " " + crrWng[1] + " " + crrWng[2] + " " + crrWng[3]);
	System.out.println(pre + " " + rec);
	
	double tp = crrWng[0];
	double tn = crrWng[1];
	double fp = crrWng[2];
	double fn = crrWng[3];
	
	double enu = tp;
	double den = (tp + fp + fn);
	*/
	
	public double getTp() {
		return tp;
	}
	
	public double getTn() {
		return tn;
	}
	
	public double getFp() {
		return fp;
	}
	
	public double getFn() {
		return fn;
	}

	@Override
	public double getVal() {
		double sc = (tp) / (tp + fp + fn);
		return sc;
	}

	@Override
	public LossScore addWith(LossScore sc2) {
		AccuracyScoreExmpAcc expf1 = (AccuracyScoreExmpAcc)sc2;
		AccuracyScoreExmpAcc newSc = new AccuracyScoreExmpAcc(tp + expf1.getTp(), tn + expf1.getTn(), fp + expf1.getFp(), fn + expf1.getFn());
		return newSc;
	}
	
	@Override
	public String getStr() {
		return (tp + "," + fp + "," + tp + "," + fn);
	}

	@Override
	public LossScore getSelfCopy() {
		AccuracyScoreExmpAcc newSc = new AccuracyScoreExmpAcc(tp, tn, fp, fn);
		return newSc;
	}

}
