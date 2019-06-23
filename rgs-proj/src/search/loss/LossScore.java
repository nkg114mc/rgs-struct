package search.loss;

public abstract class LossScore {
	
	public String info = "";

	public abstract double getVal();
	
	public abstract LossScore addWith(LossScore sc2);
	
	public abstract String getStr();
	
	public abstract LossScore getSelfCopy();

}
