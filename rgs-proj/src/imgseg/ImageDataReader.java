package imgseg;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;

import javax.imageio.ImageIO;

import edu.berkeley.nlp.futile.util.Counter;
import edu.illinois.cs.cogcomp.sl.core.SLProblem;
import imgcnn.ActualPixel;
import search.SearchResult;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;
import sequence.hw.HwSearchInferencer;

public class ImageDataReader {
	
	public static final String allFolder = "All";
	public static final String labelFolder = "labels";	
	
	//private static final double voidFeatVal = 0;
	
	String rootFolder = "";
	
	String suballFolder = "";
	String sublabelFolder = "";
	String subGtFolder = "";
	String subdebugFolder = "";
 
	public ImageDataReader(String rfdr) {
		rootFolder = rfdr;
		suballFolder = rootFolder + "/" + allFolder;
		sublabelFolder = rootFolder + "/" + labelFolder;
		subdebugFolder = rootFolder + "/" + "debug";
		subGtFolder = rootFolder + "/" + "GroundTruth";
		
		checkOneFolder(rootFolder);
		checkOneFolder(suballFolder);
		checkOneFolder(sublabelFolder);
		
		File dbf = new File(subdebugFolder);
		if (!dbf.exists()) {
			dbf.mkdirs();
			System.out.println("Create folder: " + dbf.getAbsolutePath());
		}
	}

	public String getAllFolder() {
		return suballFolder;
	}
	public String getLabelFolder() {
		return sublabelFolder;
	}
	public String getDebugFolder() {
		return subdebugFolder;
	}
	
	public static String[] lineToTokens(String line) {
		return (line.trim().split("\\s+"));
	}
	
	public static SLProblem ExampleListToSLProblem(List<ImageInstance> insts) {
		SLProblem problem = new SLProblem();
		for (int i = 0; i < insts.size(); i++) {
			HwOutput goutput = insts.get(i).getGoldOutput();
			problem.addExample(insts.get(i), goutput);
		}
		return problem;
	}
	
	
	public static List<String> getNameListFromFolder(String folder) {
		List<String> names = new ArrayList<String>();
		
		File fd = new File(folder);
		File[] allfiles = fd.listFiles();
		for (File fn : allfiles) {
			if (fn.getName().endsWith("bmp")) {
				String[] tks = fn.getName().split("\\.");
				String nm = tks[0];
				//System.out.println(fn.getName() + " " + nm);
				names.add(nm); // return extension
			}
		}
		return names;
	}
	
	public static List<String> getNameListFromFile(String file) {
		List<String> names = new ArrayList<String>();
		try {
			BufferedReader br = new BufferedReader(new FileReader(file));
			String line;
			
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (!line.equals("")) {
					if (line.endsWith("bmp")) {
						String[] tks = line.split("\\.");
						String nm = tks[0];
						names.add(nm); // return extension
					}
				}
			}
			
			System.out.println("Get " + names.size() + " images from file " + file);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return names;
	}
	
	// load one instance
	
	public ImageInstance initInstGivenName(String imgName, String[] labelNames, boolean dropVoid) {
		
		ImageInstance inst = new ImageInstance(imgName,labelNames);
				
		////
		
		String allfd = rootFolder + "/" + allFolder;
		String labelfd = rootFolder + "/" + labelFolder;
		
		String imgFilename = imgName + ".bmp";
		File imgFile = new File(allfd, imgFilename);

		
		BufferedImage img;
		try {
			img = ImageIO.read(imgFile);
			inst.setWidthHeight(img.getWidth(), img.getHeight());
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		inst.imgPath = imgFile.getPath();// image
		inst.local1Path = allfd + "/" + imgName + ".local1"; // local feature 1
		inst.local2Path = allfd + "/" + imgName + ".local2"; // local feature 2
		inst.local3Path = allfd + "/" + imgName + ".local3"; // local feature 3
		inst.local4Path = allfd + "/" + imgName + ".local4"; // local feature 4
		inst.local5Path = allfd + "/" + imgName + ".local5"; // local feature 5
		inst.local6Path = allfd + "/" + imgName + ".local6"; // global feature
		inst.edgePath = allfd + "/" + imgName + ".edges"; //adject list of super pixels
		inst.mapPath = allfd + "/" + imgName + ".map"; // super pixel -> pixel
		// ground truth
		inst.gimgPath = labelfd + "/" + imgName + ".png";  // pixel label
		inst.labelPath = labelfd + "/" + imgName + ".txt"; 
		inst.gtPath  = subGtFolder + "/" + imgName + "_GT.bmp";  // bmp ground pixel label
		
		checkFiles(inst);
		
		initSuperPixels(inst, dropVoid);
		
		// loading data
		loadSuperPixels(inst, allfd);
		
		// refresh the hwsegments
		inst.refreshHwSegments(dropVoid);
		
		return inst;    
	}
	
	private static void initSuperPixels(ImageInstance inst, boolean dropVoid) {
		try {
			BufferedReader br = new BufferedReader(new FileReader(inst.edgePath));
			String line;
			
			int lineCnt = 0;
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (!line.equals("")) {
					lineCnt++;
				}
			}
			ImageSuperPixel[] spixels = new ImageSuperPixel[lineCnt];
			for (int i = 0; i < lineCnt; i++) {
				spixels[i] = new ImageSuperPixel(i);
				//spixels[i].features = new double[22][6];
				spixels[i].features = new double[21][6];
			}
			inst.setSuperPixels(spixels, dropVoid);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private static void checkFiles(ImageInstance inst) {
		checkOneFile(inst.imgPath);
		checkOneFile(inst.local1Path);
		checkOneFile(inst.local2Path);
		checkOneFile(inst.local3Path);
		checkOneFile(inst.local4Path);
		checkOneFile(inst.local5Path);
		checkOneFile(inst.local6Path);
		checkOneFile(inst.edgePath);
		checkOneFile(inst.mapPath);
		// ground truth
		checkOneFile(inst.gimgPath);
		checkOneFile(inst.labelPath);
	}
	
	public static void  checkOneFolder(String path) {
		File rfdf = new File(path);
		if (!(rfdf.exists() && rfdf.isDirectory())) {
			throw new RuntimeException("Path " + path + " is not a folder!");
		}
	}
	
	public static void checkOneFile(String path) {
		File rfdf = new File(path);
		if (!rfdf.exists()) {
			throw new RuntimeException("File " + path + " does not exist!");
		}
	}
	
	// load one super pixel
	
	public static void loadSuperPixels(ImageInstance inst, String allfd) {
		loadSuperPixelLabels(inst);
		loadSuperPixelEdges(inst);
		loadSuperPixelMaps(inst);
		// load features
		loadSuperPixelFeatures(allfd, inst, 0);// local1
		loadSuperPixelFeatures(allfd, inst, 1);// local2
		loadSuperPixelFeatures(allfd, inst, 2);// local3
		loadSuperPixelFeatures(allfd, inst, 3);// local4
		loadSuperPixelFeatures(allfd, inst, 4);// local5
		loadSuperPixelFeatures(allfd, inst, 5);// global
	}
	
	public static void loadSuperPixelLabels(ImageInstance inst) {
		try {
			BufferedReader br = new BufferedReader(new FileReader(inst.labelPath));
			ImageSuperPixel[] spx = inst.getSuPixArr();
			String line;
			int lineCnt = 0;
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (!line.equals("")) {
					lineCnt++;
					int lb = Integer.parseInt(line);
					
					///////////////////////////////
					//// Drop 22 23 label /////////
					///////////////////////////////
					if (lb > 21) {
						lb = 21; // turn all "horse" and "mountains" to "void"
					}
					///////////////////////////////
					////////////////////////////////
					
					//System.out.println("Label = " + lb);
					spx[lineCnt - 1].setLabel(lb); 
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static int[] parseCsvVector(String str) {
		String[] toks = str.split("\\,");
		int[] values = new int[toks.length];
		for (int i = 0; i < toks.length; i++) {
			values[i] = (Integer.parseInt(toks[i]));
		}
		return values;
	}
	
	public static void loadSuperPixelEdges(ImageInstance inst) {
		try {
			BufferedReader br = new BufferedReader(new FileReader(inst.edgePath));
			ImageSuperPixel[] spx = inst.getSuPixArr();
			String line;
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (!line.equals("")) {
					String[] parts = line.split("\\s+");
					int supIdx = Integer.parseInt(parts[0]);
					if (parts.length < 2) {
						//throw new RuntimeException("Error on edges:" + line);
						spx[supIdx].neighours = new int[0]; // no neighbour
					} else {
						spx[supIdx].neighours = parseCsvVector(parts[1]);
					}
					
					
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void loadSuperPixelMaps(ImageInstance inst) {
		try {
			BufferedReader br = new BufferedReader(new FileReader(inst.mapPath));
			ImageSuperPixel[] spx = inst.getSuPixArr();
			String line;
			int lineCnt = 0;
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (!line.equals("")) {
					lineCnt++;
					String[] strs = line.split("\\,");
					int idx = Integer.parseInt(strs[0]);
					int pixelCnt = strs.length - 1;
					assert (pixelCnt > 0);
					spx[idx].setPixCnt(pixelCnt); 
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	//public static HashMap<Integer, ArrayList<Integer>> loadSuperPixelMap(ImageInstance inst) {
	public static void loadSuperPixelMap(ImageInstance inst) {
		//HashMap<Integer, ArrayList<Integer>> superToPixel = new HashMap<Integer, ArrayList<Integer>>();
		
		try {
			BufferedReader br = new BufferedReader(new FileReader(inst.mapPath));
			String line;
			
			int lineIdx = 0;
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (!line.equals("")) {
					ArrayList<Integer> oneMap = new ArrayList<Integer>();
					ArrayList<ActualPixel> pixs = new ArrayList<ActualPixel>();
					
					String[] toks = line.split("\\,");
					int supIdx = Integer.parseInt(toks[0]);
					
					for (int i = 1; i < toks.length; i++) {
						// pixel index
						int xy = Integer.parseInt(toks[i]);
						oneMap.add(xy);
						// construct pixel
						ActualPixel pix = new ActualPixel();
						pix.supix_idx = supIdx;
						pix.x = xy % inst.getWidth(); 
						pix.y = xy / inst.getWidth(); 
						pix.xy = xy;
						pixs.add(pix);
					}

				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		//return superToPixel;
	}
	
	//public static double[] parseSpaceSplitVector(String str) {
	//	String[] toks = str.split("\\s+");
	//	double[] values = new double[toks.length];
	//	for (int i = 0; i < toks.length; i++) {
	//		values[i] = (Double.parseDouble(toks[i]));
	//	}
	//	return values;
	//}
	
	public static void loadSuperPixelFeatures(String suballfd, ImageInstance inst, int classifierIdx) {
		try {
			String localPath = suballfd + "/" + inst.getName() + ".local" + String.valueOf(classifierIdx + 1);
			BufferedReader br = new BufferedReader(new FileReader(localPath));
			ImageSuperPixel[] spx = inst.getSuPixArr();
			String line;
			int lineCnt = 0;
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (!line.equals("")) {
					
					String[] toks = line.split("\\s+");
					double[][] feats = spx[lineCnt].features;
					for (int i = 0; i < toks.length; i++) {
						feats[i][classifierIdx] = (Double.parseDouble(toks[i]));
					}
					//feats[toks.length][classifierIdx] = voidFeatVal; // for "void"
					
					lineCnt++;
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		
		ImageSegLabel[] labels = ImageSegLabel.loadLabelFromFile("../msrc/imageseg_label_color_map.txt");
		String[] labelNames = ImageSegLabel.getStrLabelArr(labels, false);
		
		
		ImageDataReader reader = new ImageDataReader("../msrc");
		ImageSegEvaluator evaluator = new ImageSegEvaluator(reader.getDebugFolder());
		
		
		//List<String> allNames = getNameListFromFolder(reader.getAllFolder());
		List<String> allNames = ImageDataReader.getNameListFromFile("../msrc/Test.txt");
		
		Counter<Integer> labelCntr = new Counter<Integer>();
		Counter<Integer> exCntr = new Counter<Integer>();
		Counter<Integer> pixelCntr = new Counter<Integer>();
		
		HashSet<String> name23 = new HashSet<String>();
		
		FracScore[] fscores = new FracScore[21];
		FracScore[] gtscs = new FracScore[21];
		for (int j = 0; j < 21; j++) {
			fscores[j] = new FracScore();
			gtscs[j] = new FracScore();
		}
		
		for (String nm : allNames) {
			System.out.println(nm + " ");
			ImageInstance inst = reader.initInstGivenName(nm, labelNames, true);
			
			
			HwOutput gold = inst.getGoldOutput();
			FracScore[] oneResult = ImageSegEvaluator.evaluateOneImage(inst, gold, labelNames);
			ImageSegEvaluator.accuFracScore(fscores, oneResult);
			
			evaluator.dumpImage(inst, null, labels);
			
			
			
			ImageSuperPixel[] spx = inst.getSuPixArr();
			
			HashSet<Integer> thisLbs = new HashSet<Integer>();
			for (ImageSuperPixel sup : spx) {
				labelCntr.incrementCount(sup.getLabel(), 1);
				pixelCntr.incrementCount(sup.getLabel(), sup.getPixCnt());
				thisLbs.add(sup.getLabel());
				if (sup.getLabel() > 21) {
					name23.add(nm);
				}
			}
			
			for (Integer lb : thisLbs) {
				exCntr.incrementCount(lb, 1);
			}
			
			//System.out.println(nm + " " + inst.getWidth() + "," + inst.getHeight());
			//break;
		}
		System.out.println(allNames.size());
		
		for (String nm : name23) {
			System.out.println("========>" + nm);
		}
		
		
		
		for (Entry<Integer,Double> e : labelCntr.entrySet()) {
			System.out.println(e);
		}
		System.out.println("-----------");
		for (Entry<Integer,Double> e : exCntr.entrySet()) {
			System.out.println(e);
		}
		System.out.println("-----------");
		for (Entry<Integer,Double> e : pixelCntr.entrySet()) {
			System.out.println(e);
		}
		
		ImageSegEvaluator.printMSRCscore(fscores, labelNames);
		System.out.println("**********************");
		ImageSegEvaluator.printMSRCscore(gtscs, labelNames);
	}
	
	
}
