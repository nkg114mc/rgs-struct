package imgcnn;

public class SuperPixelEdge {
	
	public int fromId;
	public int toId;
	public int borderLen; // shared border length
	public double[] directVec;
	public int directId;
	
	public SuperPixelEdge() {
		fromId = -1;
		toId = -1;
		borderLen = 0;
		directVec = null;
		directId = -1;
	}
	
	public SuperPixelEdge(int f, int t, int l, double[] dv, int di) {
		this();
		fromId = f;
		toId = t;
		borderLen = l;
		directVec = dv;
		directId = di;
	}
	
	public double getLengthDouble() {
		return (double)borderLen;
	}

}
