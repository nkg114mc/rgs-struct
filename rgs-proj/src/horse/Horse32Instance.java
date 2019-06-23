package horse;

import java.util.List;

import sequence.hw.HwInstance;
import sequence.hw.HwSegment;

public class Horse32Instance extends HwInstance {

	String name;
	int width;
	int height;
	
	// file paths
	public String imgPath; // image
	public String local1Path; // local feature 1
	public String labelPath; // super pixel label
	
	public Horse32Instance(String nm, List<HwSegment> segs, String[] imgLabels) {
		super(segs, imgLabels);
		name = nm;
	}
	
	public Horse32Instance(List<HwSegment> segs, String[] albt) {
		super(segs, albt);
	}

	public void setWidthHeight(int w, int h) {
		if (w >= 0) width = w;
		if (h >= 0) height = h;
	}
	
	public String getName() {
		return name;
	}
	
	public int getSize() {
		return (width * height);
	}
	
	public int getHeight() {
		return height;
	}
	
	public int getWidth() {
		return width;
	}
	
	public int getGlobalIndex(int x, int y) {
		int idx = x * width + y;
		return idx;
	}

/*
	public static List<HwSegment> toHwSegs(ImageSuperPixel[] spixels, String[] imgLabels, boolean dropVoid) {
		List<HwSegment> segList = new ArrayList<HwSegment>();
		for (int i = 0; i < spixels.length; i++) {
			spixels[i].hwsegIndex = -1; // no corresponding segIdx
			if (dropVoid) {
				if (spixels[i].getLabel() <= 20) { // drop all "void"
					HwSegment seg = new HwSegment(i, (new double[0]), imgLabels[spixels[i].getLabel()]);
					segList.add(seg);
					spixels[i].hwsegIndex = segList.size() - 1;
				} else {
					spixels[i].hwsegIndex = -1;
				}
			} else {
				HwSegment seg = new HwSegment(i, (new double[0]), imgLabels[spixels[i].getLabel()]);
				segList.add(seg);
				spixels[i].hwsegIndex = segList.size() - 1;
			}
		}
		return segList;
	}
*/



}
