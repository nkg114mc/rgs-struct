package imgcnn;

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
import imgseg.FracScore;
import imgseg.ImageInstance;
import imgseg.ImageSegEvaluator;
import imgseg.ImageSegLabel;
import imgseg.ImageSuperPixel;
import search.SearchResult;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;
import sequence.hw.HwSearchInferencer;

public class ImageCNNReader {
	
	public static final String allFolder = "All";
	public static final String cnnFolder = "superpix_cnn";
	
	String rootFolder = "";
	
	String suballFolder = "";
	//String sublabelFolder = "";
	String subGtFolder = "";
	String subdebugFolder = "";
	String subsuperpixCnnFolder = "";
 
	public ImageCNNReader(String rfdr) {
		rootFolder = rfdr;
		suballFolder = rootFolder + "/" + allFolder;
		//sublabelFolder = rootFolder + "/" + labelFolder;
		subdebugFolder = rootFolder + "/" + "debug_cnn";
		subGtFolder = rootFolder + "/" + "GroundTruth";
		subsuperpixCnnFolder = rootFolder + "/" + cnnFolder;
		
		checkOneFolder(rootFolder);
		checkOneFolder(suballFolder);
		checkOneFolder(subsuperpixCnnFolder);
		
		File dbf = new File(subdebugFolder);
		if (!dbf.exists()) {
			dbf.mkdirs();
			System.out.println("Create folder: " + dbf.getAbsolutePath());
		}
	}

	public String getAllFolder() {
		return suballFolder;
	}
	//public String getLabelFolder() {
	//	return sublabelFolder;
	//}
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
		
		String allfd = suballFolder; //rootFolder + "/" + allFolder; //String labelfd = rootFolder + "/" + labelFolder;
		
		String imgFilename = imgName + ".bmp";
		File imgFile = new File(allfd, imgFilename);

		
		BufferedImage img;
		try {
			img = ImageIO.read(imgFile);
			inst.setWidthHeight(img.getWidth(), img.getHeight());
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		inst.imgPath = imgFile.getPath(); // image
		inst.cnnPath = subsuperpixCnnFolder + "/" + imgName + ".cnnsup";
		// ground truth
		inst.gimgPath = subsuperpixCnnFolder + "/" + imgName + ".bmp";  // pixel label
		inst.gtPath  = subGtFolder + "/" + imgName + "_GT.bmp";  // bmp ground pixel label
		
		checkFiles(inst);

		// loading data
		//loadCNNSuperPixels(inst, dropVoid);
		loadCNNPixels(inst, dropVoid);
		
		// refresh the hwsegments
		inst.refreshHwSegments(dropVoid);
		
		return inst;    
	}
	
	private static void checkFiles(ImageInstance inst) {
		checkOneFile(inst.imgPath);
		checkOneFile(inst.cnnPath);
		// ground truth
		checkOneFile(inst.gimgPath);
		checkOneFile(inst.gtPath);
	}
	
	public static void checkOneFolder(String path) {
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
	
	public static void loadCNNSuperPixels(ImageInstance inst, boolean dropVoid) {
		/*
		initSuperPixels(inst, dropVoid);
		
		loadSuperPixelLabels(inst);
		loadSuperPixelEdges(inst);
		loadSuperPixelMaps(inst);
		// load features
		loadSuperPixelCNNFeatures(allfd, inst, 0, 24); // cnn patch 1
		loadSuperPixelCNNFeatures(allfd, inst, 1, 36); // cnn patch 2
		loadSuperPixelCNNFeatures(allfd, inst, 2, 48); // cnn patch 3
		loadSuperPixelCNNFeatures(allfd, inst, 3, 72); // cnn patch 4
		*/
		try {
			BufferedReader br = new BufferedReader(new FileReader(inst.cnnPath));
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
				spixels[i].features = null; //new double[21][6];
			}
			inst.setSuperPixels(spixels, dropVoid);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void loadCNNPixels(ImageInstance inst, boolean dropVoid) {
		try {
			BufferedReader br = new BufferedReader(new FileReader(inst.cnnPath));
			String line;
			
			ArrayList<ActualPixel> pixels = new ArrayList<ActualPixel>();
			
			int lineCnt = 0;
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (!line.equals("")) {
					lineCnt++;
					ActualPixel pix = ActualPixel.loadOnePixel(line);
					pix.xy = pix.x * inst.getHeight() + pix.y ;
					pixels.add(pix);
				}
			}
			
			
			//// index is the super-pixel index
			HashMap<Integer, ArrayList<ActualPixel>> pixToSup = new HashMap<Integer, ArrayList<ActualPixel>>();
			for (ActualPixel pix : pixels) {
				ArrayList<ActualPixel> lst = pixToSup.get(pix.supix_idx);
				if (lst == null) {
					lst =  new ArrayList<ActualPixel>();
					lst.add(pix);
					pixToSup.put(pix.supix_idx, lst);
				} else {
					lst.add(pix);
				}
			}
			
			ImageSuperPixel[] spixels = new ImageSuperPixel[pixToSup.size()];

			int i = 0;
			for (Integer supixIdx : pixToSup.keySet()) {
				ArrayList<ActualPixel> lst = pixToSup.get(supixIdx.intValue());
				spixels[i] = new ImageSuperPixel(i);
				spixels[i].features = null; //new double[21][6];
				spixels[i].setLabel(0);
				spixels[i].setPixCnt(lst.size());
				spixels[i].neighours = new int[0];
				spixels[i].setZMap(lst);
				i++;
			}

			inst.setSuperPixels(spixels, dropVoid);
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
/*	
	public static void loadSuperPixels(ImageInstance inst, String allfd, boolean dropVoid) {
		
		initSuperPixels(inst, dropVoid);
		
		loadSuperPixelLabels(inst);
		loadSuperPixelEdges(inst);
		loadSuperPixelMaps(inst);
		// load features
		loadSuperPixelCNNFeatures(allfd, inst, 0, 24); // cnn patch 1
		loadSuperPixelCNNFeatures(allfd, inst, 1, 36); // cnn patch 2
		loadSuperPixelCNNFeatures(allfd, inst, 2, 48); // cnn patch 3
		loadSuperPixelCNNFeatures(allfd, inst, 3, 72); // cnn patch 4
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
				spixels[i].features = null; //new double[21][6];
			}
			inst.setSuperPixels(spixels, dropVoid);
		} catch (IOException e) {
			e.printStackTrace();
		}
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
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (!line.equals("")) {
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
	
	public static void loadSuperPixelCNNFeatures(String suballfd, ImageInstance inst, int patchIdx, int patchSz) {
		try {
			String localPath = suballfd + "/" + inst.getName() + ".local" + String.valueOf(patchIdx + 1);
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
						feats[i][patchIdx] = (Double.parseDouble(toks[i]));
					}
					
					lineCnt++;
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
*/

	public static void main(String[] args) {
		
		ImageSegLabel[] labels = ImageSegLabel.loadLabelFromFile("../msrc/imageseg_label_color_map.txt");
		String[] labelNames = ImageSegLabel.getStrLabelArr(labels, false);		
		
		ImageCNNReader reader = new ImageCNNReader("../msrc");
		ImageSegEvaluator evaluator = new ImageSegEvaluator(reader.getDebugFolder());
		
		List<String> allNames = ImageCNNReader.getNameListFromFile("../msrc/Test.txt");
		
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
