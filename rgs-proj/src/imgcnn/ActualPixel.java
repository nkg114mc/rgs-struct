package imgcnn;

public class ActualPixel {
	
	public int x;
	public int y;
	public int xy;
	
	public int gt_label;
	
	public int supix_idx;
	
	public int gtColorR;
	public int gtColorG;
	public int gtColorB;
	
	// about belonging super pixel
	public int[] adjSuIdx = null;
	public int majorAdjIdx = -1;
	public boolean isBound = false; 
	
	
	public static ActualPixel loadOnePixel(String line) {
		
		ActualPixel p = new ActualPixel();
		
		String[] arr = line.split("\\s+");
		
		p.x = Integer.parseInt(arr[0]);
		p.y = Integer.parseInt(arr[1]);
		
		p.supix_idx = Integer.parseInt(arr[2]);
		
		p.gtColorR = Integer.parseInt(arr[3]);
		p.gtColorG = Integer.parseInt(arr[4]);
		p.gtColorB = Integer.parseInt(arr[5]);
		
		p.gt_label = -1;

		return p;
	}
}
