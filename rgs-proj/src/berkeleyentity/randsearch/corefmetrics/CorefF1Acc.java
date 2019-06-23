package berkeleyentity.randsearch.corefmetrics;

import search.loss.LossScore;

public class CorefF1Acc extends LossScore {
	
	double preNum;
	double preDen;
	double recNum;
	double recDen;
	
	public CorefF1Acc(double pNum, double pDen, double rNum, double rDen) {
		preNum = pNum;
		preDen = pDen;
		recNum = rNum;
		recDen = rDen;
	}
	
	public double getPreNum() {
		return preNum;
	}
	
	public double getPreDen() {
		return preDen;
	}
	
	public double getRecNum() {
		return recNum;
	}
	
	public double getRecDen() {
		return recDen;
	}

	
	@Override
	public double getVal() {
		double sc = (2 * preNum * recNum) / (recNum * preDen + preNum * recDen);
		return sc;
	}

	@Override
	public LossScore addWith(LossScore sc2) {
		CorefF1Acc f1 = (CorefF1Acc)sc2;
		CorefF1Acc newSc = new CorefF1Acc(preNum + f1.getPreNum(),
				                          preDen + f1.getPreDen(),
				                          recNum + f1.getRecNum(),
				                          recDen + f1.getRecDen());
		return newSc;
	}
	

	@Override
	public String getStr() {
		return (preNum + " / " + preDen + " , " + recNum + " / " + recDen);
	}

	@Override
	public LossScore getSelfCopy() {
		CorefF1Acc newSc = new CorefF1Acc(preNum, preDen, recNum, recDen);
		return newSc;
	}

}
