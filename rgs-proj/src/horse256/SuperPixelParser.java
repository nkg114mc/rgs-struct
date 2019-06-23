package horse256;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import imgcnn.ActualPixel;
import imgcnn.SuperPixelEdge;
import imgseg.ImageSuperPixel;

public class SuperPixelParser {

	BufferedReader br;
	private List<String> cachedTokens;
	
	public SuperPixelParser() {
		br = null;
		cachedTokens = new LinkedList<String>();
	}
	
	public void initFile(String fn) {
		try {
			br = new BufferedReader(new FileReader(fn));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private String getNextLine() {
		String line = null;
		try {
			line = br.readLine();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return line;
	}
	
	private String[] lineToTokens(String line) {
		StringBuilder sb = new StringBuilder("");
		for (int i = 0 ; i < line.length(); i++) {
			if (line.charAt(i) == '(' || line.charAt(i) == ')') {
				sb.append(' ');
				sb.append(line.charAt(i));
				sb.append(' ');
			} else {
				sb.append(line.charAt(i));
			}
		}
		return sb.toString().trim().split("\\s+");
	}
	
	// lexer
	public String getNextToken() {
		if (hasNextToken()) {
			String tk = new String(cachedTokens.get(0)); // head
			//System.err.println("token = " + tk);
			cachedTokens.remove(0); // pop head
			return tk;
		} else {
			return null;
		}
	}
	public boolean hasNextToken() {
		if (br == null) {
			return false;
		}
		try {
			if (cachedTokens.isEmpty()) {
				// input next line
				String line = null;
				String[] tks = null;
				while (true) {
					line = br.readLine();
					if (line == null) break;
					line = line.trim();
					if (!line.equals("")) {
						tks = lineToTokens(line);
						if (tks.length > 0) {
							break;
						}
					}
				}

				if (line == null) {
					return false;
				} else {
					for (int j = 0; j < tks.length; j++) {
						cachedTokens.add(tks[j]);
					}
					return true;
				}

			} else {
				return true;
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		return false;
	}
	
	public List<ImageSuperPixel> doParsing() {
	
		List<ImageSuperPixel> results = new ArrayList<ImageSuperPixel>();
		
		while (hasNextToken()) {
			
			String curTk = getNextToken();
			if (curTk.equals("(")) {
				String name = getNextToken();
				if (name.equals("supix")) {
					ImageSuperPixel sup = parseSuperPixel();
					results.add(sup);
				} else {
					throw new RuntimeException("Unknown head: " + name);
				}
			} else {
				throw new RuntimeException("Unknown head: " + curTk);
			}
			
		}
		
		return results;
	}
	
	public ImageSuperPixel parseSuperPixel() {
		
		int id = -1;
		List<ActualPixel> pixs = null;
		List<SuperPixelEdge> edgs = null;
		double[] centroid = null;
		
		double[] fv36 = null;
		double[] fv48 = null;
		double[] fv64 = null;
		double[] fv72 = null;
		
		while (hasNextToken()) {
			String curTk = getNextToken();
			if (curTk.equals("(")) {
				String hd = getNextToken();
				 if (hd.equals("id")) {
					 String v = parseValue();
					 id = Integer.parseInt(v);
					 System.out.println("id = " + id);
				 } else if (hd.equals("centroid")) {
					 centroid = parseDoubleVector();
				 } else if (hd.equals("featcnn32")) {
					 fv36 = parseDoubleVector();
				 } else if (hd.equals("featcnn48")) {
					 fv48 = parseDoubleVector();
				 } else if (hd.equals("featcnn64")) {
					 fv64 = parseDoubleVector();
				 } else if (hd.equals("featcnn72")) {
					 fv72 = parseDoubleVector();
				 } else if (hd.equals("pixels")) {
					 pixs = parsePixels();
				 } else if (hd.equals("edges")) {
					 edgs = parseEdges();
				 } else {
					 throw new RuntimeException("Unknown head: " + hd);
				 }
			} else if (curTk.equals(")")) {
				 // done
				 break;
			}
		}
		
		
		ImageSuperPixel imp = new ImageSuperPixel(id);
		imp.setAdjEdges(edgs);
		imp.setZMap(pixs);
		imp.centroid = centroid;
		// store features
		imp.features = new double[4][];
		imp.features[0] = fv36;
		imp.features[1] = fv48;
		imp.features[2] = fv64;
		imp.features[3] = fv72;
		
		System.out.print("Get one super pixel:");
		System.out.print(" pixels: " + imp.getPixCnt());
		System.out.print(" edges: " + imp.getAdjSupixCnt());
		System.out.print(" len(fv36) = " + imp.features[0].length);
		System.out.print(" len(fv48) = " + imp.features[1].length);
		System.out.print(" len(fv64) = " + imp.features[2].length);
		System.out.print(" len(fv72) = " + imp.features[3].length);
		System.out.println("");
		
		//imp.neighours
		return imp;
	}
	
	public List<SuperPixelEdge> parseEdges() {
		
		List<SuperPixelEdge> edgs = new ArrayList<SuperPixelEdge>();

		while (hasNextToken()) {
			String curTk = getNextToken();
			if (curTk.equals("(")) {
				String hd = getNextToken();
				 if (hd.equals("edge")) {
					 SuperPixelEdge edg = parseEdge();
					 edgs.add(edg);
				 } else {
					 throw new RuntimeException("Unknown edges head: " + hd);
				 }
			} else if (curTk.equals(")")) {
				 // done
				 break;
			 }
		}
		
		return edgs;
	}
	
	public List<ActualPixel> parsePixels() {
		
		List<ActualPixel> pixs = new ArrayList<ActualPixel>();

		while (hasNextToken()) {
			String curTk = getNextToken();
			if (curTk.equals("(")) {
				String hd = getNextToken();
				 if (hd.equals("pixel")) {
					 ActualPixel pix = parsePixel();
					 pixs.add(pix);
				 } else {
					 throw new RuntimeException("Unknown pixels head: " + hd);
				 }
			} else if (curTk.equals(")")) {
				 // done
				 break;
			}
		}
		
		return pixs;
	}
	
	public SuperPixelEdge parseEdge() {
		
		int fromIdx = -1;
		int toIdx = -1;
		int len = 0;
		int directId = -1;
		double[] directVec = null;
		
		while (hasNextToken()) {
			String curTk = getNextToken();
			if (curTk.equals("(")) {
				String hd = getNextToken();
				 if (hd.equals("fromid")) {
					 String v = parseValue();
					 fromIdx = Integer.parseInt(v);
				 } else if (hd.equals("toid")) {
					 String v = parseValue();
					 toIdx = Integer.parseInt(v);
				 } else if (hd.equals("len")) {
					 String v = parseValue();
					 len = Integer.parseInt(v);
				 } else if (hd.equals("directid")) {
					 String v = parseValue();
					 directId = Integer.parseInt(v);
				 } else if (hd.equals("directvec")) {
					 directVec = parseDoubleVector();
				 } else {
					 throw new RuntimeException("Unknown edge head: " + hd);
				 }
			} else if (curTk.equals(")")) {
				 // done
				 break;
			 }
		}
		
		SuperPixelEdge edge = new SuperPixelEdge(fromIdx, toIdx, len, directVec, directId);
		return edge;
	}
	
	public ActualPixel parsePixel() {
		
		int[] coords = null;
		int[] adjSuIdx = null;
		int majorAdjIdx = -1;
		boolean isbnd = false;
		
		while (hasNextToken()) {
			String curTk = getNextToken();
			if (curTk.equals("(")) {
				String hd = getNextToken();
				//xy (vec 5 5)) (isbound true) (adjsupixids (vec 1)) (
				if (hd.equals("xy")) {
					coords = parseIntVector();
				} else if (hd.equals("isbound")) {
					String v = parseValue();
					isbnd = Boolean.parseBoolean(v);
				} else if (hd.equals("adjsupixids")) {
					adjSuIdx = parseIntVector();
				} else if (hd.equals("majorsupixid")) {
					String v = parseValue();
					majorAdjIdx = Integer.parseInt(v);
				} else {
					 throw new RuntimeException("Unknown pixel head: " + hd);
				 }
			} else if (curTk.equals(")")) {
				 // done
				 break;
			 }
		}
		
		ActualPixel pixel = new ActualPixel();
		pixel.x = coords[0]; 
		pixel.y = coords[1];
		pixel.adjSuIdx = adjSuIdx;
		pixel.majorAdjIdx = majorAdjIdx;
		pixel.isBound = isbnd;
		
		return pixel;
	}
	
	public String parseValue() {
		String v = getNextToken();
		String end = getNextToken();
		if (end.equals(")")) {
			return v;
		} else {
			throw new RuntimeException("Value does not ends: " + v);
		}
	}
	
	public double[] parseDoubleVector() {
		List<String> strv = parseStingVector();
		if (strv != null) {
			double[] dv = new double[strv.size()];
			for (int i = 0; i < strv.size(); i++) {
				dv[i] = Double.parseDouble(strv.get(i));
			}
			return dv;
		} else {
			return null;
		}
	}
	public int[] parseIntVector() {
		List<String> strv = parseStingVector();
		if (strv != null) {
			int[] intv = new int[strv.size()];
			for (int i = 0; i < strv.size(); i++) {
				intv[i] = Integer.parseInt(strv.get(i));
			}
			return intv;
		} else {
			return null;
		}
	}
	public List<String> parseStingVector() {
		String v = getNextToken();
		if (v.equals("none")) {
			// this bracket is from the parent
			String end = getNextToken();
			if (end.equals(")")) {
				return null;
			} else {
				throw new RuntimeException("Property not ends: " + end);
			}
		} else if (v.equals("(")) {
			String nm = getNextToken();
			if (nm.equals("vec")) {
				List<String> sv = new ArrayList<String>();
				while (true) {
					String vi = getNextToken();
					if (vi.equals(")")) {
						break;
					}
					sv.add(vi);
				}
				
				// this bracket is from the parent
				String end = getNextToken();
				if (end.equals(")")) {
					return sv;
				} else {
					throw new RuntimeException("Property not ends: " + end);
				}
			} else {
				throw new RuntimeException("Unknown head: " + v);
			}
		} else {
			throw new RuntimeException("Not a vector: " + v);
		}
	}
	
	
	public static void main(String[] args) {
		SuperPixelParser prsr = new SuperPixelParser();
		prsr.initFile("/home/mc/workplace/imgseg/weim/pic256/outputcnn256/image-233.local1");
		//prsr.initFile("/home/mc/workplace/imgseg/weim/pic256/outputcnn256/test1.txt");
		List<ImageSuperPixel> supx = prsr.doParsing();
		System.out.println("sup number = " + supx.size());
	}
	
}
