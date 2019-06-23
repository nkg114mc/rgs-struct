package general;

import sequence.hw.HwOutput;

public interface AbstractInstance{// extends IInstance {
	
	public abstract int size();
	
	public abstract int domainSize();
	
	public abstract HwOutput getGoldOutput();
	
	
	public abstract AbstractOutput getPredict();
	public abstract void setPredict(AbstractOutput opt);	

}
