package imgseg;

public class FracScore {
	
	private double num;
	private double den;
	
	public FracScore(double n, double d) {
		num = n;
		den = d;
	}
	
	public FracScore() {
		num = 0;
		den = 0;
	}
	
	public double getNum() {
		return num;
	}
	
	public double getDen() {
		return den;
	}
	
	public double getFrac() {
		if (den == 0) {
			return 0;
		} else {
			double ret = num / den;
			return ret;
		}
	}
	
	///////////////////////////////

	
	public void setNum(double n) {
		num = n;
	}
	
	public void setDen(double d) {
		den = d;
	}
	
	public void setNumDen(double n, double d) {
		num = n;
		den = d;
	}
	
	public void addNumDen(double n, double d) {
		num += n;
		den += d;
	}
	
	public static FracScore sumToGetNew(FracScore a, FracScore b) {
		FracScore sum = new FracScore();
		sum.setNumDen((a.getNum() + b.getNum()), (a.getDen() + b.getDen()));
		return sum;
	}
	
	/**
	 * Add b to s
	 * @param s
	 * @param b
	 * @return
	 */
	public static void sumTo(FracScore s, FracScore b) {
		double newNum = s.getNum() + b.getNum();
		double newDen = s.getDen() + b.getDen();
		s.setNumDen(newNum, newDen);
	}

}
