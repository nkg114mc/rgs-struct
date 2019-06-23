package horse256;

import java.util.ArrayList;
import java.util.List;

import imgseg.ImageSuperPixel;
import sequence.hw.HwInstance;
import sequence.hw.HwSegment;

public class Horse256Instance extends HwInstance {

	String name;
	int width;
	int height;
	
	// file paths
	public String imgPath; // image
	public String local1Path; // local feature 1
	public String labelPath; // super pixel label
	
	ImageSuperPixel[] superPixelArr;
	
	public ImageSuperPixel[] getSuPixArr() {
		return superPixelArr;
	}
	public ImageSuperPixel getSuPix(int i) {
		assert (superPixelArr[i] != null);
		return superPixelArr[i];
	}
	
	public void setSuperPixels(ImageSuperPixel[] spixels) {
		superPixelArr = spixels;
		
	}
	
	public void refreshHwSegs() {
		letterSegs = toHwSegsNEW(superPixelArr, HorseLabel.FOREBACK_GROUND_DOMAIN);
	}
	
	public Horse256Instance(String nm, List<HwSegment> segs, String[] imgLabels) {
		super(segs, imgLabels);
		name = nm;
	}
	
	public Horse256Instance(List<HwSegment> segs, String[] albt) {
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
		int idx = x  + y * height;
		return idx;
	}


	public static List<HwSegment> toHwSegsNEW(ImageSuperPixel[] spixels, String[] imgLabels) {
		List<HwSegment> segList = new ArrayList<HwSegment>();
		for (int i = 0; i < spixels.length; i++) {
			HwSegment seg = new HwSegment(i, (new double[0]), imgLabels[spixels[i].getLabel()]);
			segList.add(seg);
			spixels[i].hwsegIndex = i;
		}
		return segList;
	}




}
