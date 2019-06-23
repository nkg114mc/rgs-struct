package imgseg;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class ImageSegLabel {
	
	public int value;
	public String name;

	public int colorRGB;
	public int r;
	public int g;
	public int b;
	public int originIdx;

	
	public ImageSegLabel(int v, int rgbInt, int rv, int gv, int bv, String nm, int idx) {
		value = v;
		name = nm;
		colorRGB = rgbInt;
		r = rv;
		g = gv;
		b = bv;
		originIdx = idx;
	}
	
	public String toString() {
		return ("[" + originIdx + "]:" + value + " " + name);
	}

	public static ImageSegLabel[] loadLabelFromFile(String path) {
		
		try {
			BufferedReader br = new BufferedReader(new FileReader(path));
			String line;
			ArrayList<ImageSegLabel> lbs = new ArrayList<ImageSegLabel>();
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (!line.equals("")) {
					ImageSegLabel lb = parseLine(line);
					lbs.add(lb);
				}
			}
			
			ImageSegLabel[] lbArr = lbs.toArray(new ImageSegLabel[0]);
			System.out.println("Load " + lbs.size() + " labels...");
			return lbArr;
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
		return null; // should not reach here
	}
	
	public static ImageSegLabel parseLine(String lineStr) {
		
		String[] tks = ImageDataReader.lineToTokens(lineStr);
		if (tks.length < 7) {
			throw new RuntimeException("Not enough elements in line:" + lineStr);
		}
		//Array(label, value, r, g, b, className, labelIdx)
		ImageSegLabel lable = new ImageSegLabel(Integer.parseInt(tks[0]),
				                                Integer.parseInt(tks[1]),
				                                Integer.parseInt(tks[2]),
				                                Integer.parseInt(tks[3]),
				                                Integer.parseInt(tks[4]),
				                                (tks[5]),
				                                Integer.parseInt(tks[6]));
		return lable;
	}
	
	public static String[] getStrLabelArr(ImageSegLabel[] labels, boolean includeVoidAndThen) {
		String[] lbStr = new String[labels.length];
		for (int i = 0; i < labels.length; i++) {
			lbStr[labels[i].originIdx] = labels[i].name;
		}
		
		if (!includeVoidAndThen) { // drop void label
			String[] lbStrNoVoid = new String[21];
			for (int i = 0; i < 21; i++) {
				lbStrNoVoid[i] = lbStr[i];
			}
			lbStr = lbStrNoVoid;
		}
		
		return lbStr;
	}
	
	public static void main(String[] args) {
		ImageSegLabel[] lbs = ImageSegLabel.loadLabelFromFile("../msrc/imageseg_label_color_map.txt");
		for (ImageSegLabel isl : lbs) {
			System.out.println(isl.toString());
		}
	}
}
