package imgseg;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import sequence.hw.HwInstance;
import sequence.hw.HwSegment;

public class ImageInstance extends HwInstance {
	
	String name;
	ImageSuperPixel[] superPixelArr;
	int width;
	int height;
	
	// file paths
	public String imgPath; // image
	public String local1Path; // local feature 1
	public String local2Path; // local feature 2
	public String local3Path; // local feature 3
	public String local4Path; // local feature 4
	public String local5Path; // local feature 5
	public String local6Path; // global feature
	public String edgePath; //adject list of super pixels
	public String mapPath; // super pixel -> pixel
	// cnn file path
	public String cnnPath; // all-in-one (for cnn only)
	// ground truth
	public String gimgPath;  // pixel label
	public String gtPath;    // same as above, but in .bmp format)
	public String labelPath; // super pixel label
	
	private int[][] leftNbrs = null;
	private int[][] rightNbrs = null;	
	
	public ImageInstance(String nm, ImageSuperPixel[] spixels, String[] imgLabels, boolean dropVoid) {
		super(toHwSegs(spixels, imgLabels, dropVoid), imgLabels);
		superPixelArr = spixels;
		name = nm;
		
	}
	
	public ImageInstance(String nm, String[] imgLabels) {
		super(null, imgLabels);
		superPixelArr = null;
		name = nm;
		
	}

	public void setWidthHeight(int w, int h) {
		if (w >= 0) width = w;
		if (h >= 0) height = h;
	}
	
	public void setSuperPixels(ImageSuperPixel[] spixels, boolean dropVoid) {
		superPixelArr = spixels;
		letterSegs = toHwSegs(spixels, alphabet, dropVoid);
	}
	
	// recompute HwSegments
	public void refreshHwSegments(boolean dropVoid) {
		letterSegs = toHwSegs(superPixelArr, alphabet, dropVoid);
		computeLeftRightNeighbours();
	}
	
	public String getName() {
		return name;
	}
	
	public int getSize() {
		return superPixelArr.length;
	}
	
	public int getHeight() {
		return height;
	}
	
	public int getWidth() {
		return width;
	}
	
	public ImageSuperPixel[] getSuPixArr() {
		return superPixelArr;
	}
	public ImageSuperPixel getSuPix(int i) {
		assert (superPixelArr[i] != null);
		return superPixelArr[i];
	}
	
	public int segIdxToSupIdx(int segIdx) {
		HwSegment seg = letterSegs.get(segIdx);
		return seg.index;
	}
	
	public int supIdxTosegIdx(int supIdx) {
		ImageSuperPixel sp = getSuPix(supIdx);
		return sp.hwsegIndex;
	}

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
	
	public void computeLeftRightNeighbours() {
		leftNbrs = new int[letterSegs.size()][];
		rightNbrs = new int[letterSegs.size()][];
		
		ArrayList<ArrayList<Integer>> lnbrs = new ArrayList<ArrayList<Integer>>(letterSegs.size());
		ArrayList<ArrayList<Integer>> rnbrs = new ArrayList<ArrayList<Integer>>(letterSegs.size());
		for (int i = 0; i < letterSegs.size(); i++) {
			lnbrs.add(new ArrayList<Integer>());
			rnbrs.add(new ArrayList<Integer>());
		}
		
		for (int i = 0; i < letterSegs.size(); i++) {
			ArrayList<Integer> rnrs = rnbrs.get(i);
			
			int supIdx = letterSegs.get(i).index;
			int[] neigbours = getSuPix(supIdx).neighours;
			for (int jdx = 0; jdx < neigbours.length; jdx++) {
				int j = neigbours[jdx];
				int segIdx = getSuPix(j).hwsegIndex;
				if (segIdx >= 0) { // not a "void" pixel
					rnrs.add(segIdx); // i's left is segIdx
					lnbrs.get(segIdx).add(i); // segIdx's right is i					
				}
			}
		}
		
		
		for (int i = 0; i < letterSegs.size(); i++) {
			rightNbrs[i] = Arrays.stream(rnbrs.get(i).toArray(new Integer[0])).mapToInt(Integer::intValue).toArray();
			leftNbrs[i] =  Arrays.stream(lnbrs.get(i).toArray(new Integer[0])).mapToInt(Integer::intValue).toArray();
		}
		
	}
	
	public int[] getRightNeighbours(int i) {
		return rightNbrs[i];
	}
	public int[] getLeftNeighbours(int i) {
		return leftNbrs[i];
	}


}
