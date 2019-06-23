package search.loss;

public class LossScoreExmpF1 extends LossScore {
	
	double tp;
	double tn;
	double fp;
	double fn;
	
	public LossScoreExmpF1(double trpos, double trneg, double fspos, double fsneg) {
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
		//double sc = (2 * tp) / (tp + fp + tp + fn);
		double sc = (fp + fn) / (tp + fp + tp + fn); // 1 - f1
		return sc;
	}

	@Override
	public LossScore addWith(LossScore sc2) {
		LossScoreExmpF1 expf1 = (LossScoreExmpF1)sc2;
		LossScoreExmpF1 newSc = new LossScoreExmpF1(tp + expf1.getTp(), tn + expf1.getTn(), fp + expf1.getFp(), fn + expf1.getFn());
		return newSc;
	}
	
	@Override
	public String getStr() {
		return (tp + "," + fp + "," + tp + "," + fn);
	}
	
	@Override
	public LossScore getSelfCopy() {
		LossScoreExmpF1 newSc = new LossScoreExmpF1(tp, tn, fp, fn);
		return newSc;
	}

}
