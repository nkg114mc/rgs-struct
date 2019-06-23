package general;

public interface AbstractOutput{// extends IStructure {

	public abstract int size();
	
	public abstract int tagSize();
	
	public abstract int getOutput(int slotIdx);
	public abstract void setOutput(int slotIdx, int slotValue);

	public abstract AbstractOutput copyFrom(AbstractOutput src);
	
	public boolean isEqual(AbstractOutput output);
	
}
