package horse256;

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
import imgseg.FracScore;
import imgseg.ImageDataReader;
import imgseg.ImageInstance;
import imgseg.ImageSuperPixel;
import multilabel.instance.Label;
import search.SearchResult;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;
import sequence.hw.HwSearchInferencer;
import sequence.hw.HwSegment;

public class Horse256Reader {
	
	public static final String allFolder = "image256";
	public static final String labelFolder = "mask256";
	public static final String supixFolder = "outputcnn256";
	
	private static final String IMAGE_EXT = "png";
	
	String rootFolder = "";

	String suballFolder = "";
	String sublabelFolder = "";
	String subsupixFolder = "";
	String subdebugFolder = "";
 
	public Horse256Reader(String rfdr) {
		rootFolder = rfdr;
		suballFolder = rootFolder + "/" + allFolder;
		sublabelFolder = rootFolder + "/" + labelFolder;
		subsupixFolder = rootFolder + "/" + supixFolder;
		subdebugFolder = rootFolder + "/" + "debug";
		
		checkOneFolder(rootFolder);
		checkOneFolder(suballFolder);
		checkOneFolder(subsupixFolder);
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
	
	public static SLProblem ExampleListToSLProblem(List<Horse256Instance> insts) {
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
			if (fn.getName().endsWith(IMAGE_EXT)) {
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
					if (line.endsWith(IMAGE_EXT)) {
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
	public Horse256Instance initInstGivenName(String imgName) {
		
		Horse256Instance inst = new Horse256Instance(imgName, null, Label.MULTI_LABEL_DOMAIN);
		
		String imgFilename = imgName + ".png";
		File imgFile = new File(suballFolder, imgFilename);
		
		BufferedImage img;
		try {
			img = ImageIO.read(imgFile);
			inst.setWidthHeight(img.getWidth(), img.getHeight());
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		inst.imgPath = imgFile.getPath();// image
		inst.local1Path = subsupixFolder + "/" + imgName + ".local1"; // local feature 1
		// ground truth
		inst.labelPath = sublabelFolder + "/" + imgName.replaceAll("image", "mask") + ".png";  // bmp ground pixel label
		
		
		checkFiles(inst);
		
		// loading data
		loadSuperPixels(inst);
		
		return inst;    
	}

	private static void checkFiles(Horse256Instance inst) {
		checkOneFile(inst.imgPath);
		checkOneFile(inst.local1Path);
		// ground truth
		checkOneFile(inst.labelPath);
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
	public static void loadSuperPixels(Horse256Instance inst) {
		
		SuperPixelParser supParser = new SuperPixelParser();
		supParser.initFile(inst.local1Path);

		// load all super pixels
		List<ImageSuperPixel> loadedSupixs = supParser.doParsing();
		
		// set up super pixel
		ImageSuperPixel[] spixels = new ImageSuperPixel[loadedSupixs.size()];
		for (int i = 0; i < loadedSupixs.size(); i++) {
			spixels[i] = loadedSupixs.get(i);
		}
		inst.setSuperPixels(spixels);
		
		// match super pixel with image
		matchSuperPixelsWithImage(inst);
		
		inst.refreshHwSegs();
		
	}
	
	public static void matchSuperPixelsWithImage(Horse256Instance inst) {
		
		BufferedImage orgImg = Horse256Evaluator.getOriginImage(inst);
		BufferedImage gtImg = Horse256Evaluator.getGtImage(inst);
		
		ImageSuperPixel[] supixs = inst.getSuPixArr();
		for (ImageSuperPixel sup : supixs) {
			
			Counter<Integer> pixelLabelCntr = new Counter<Integer>();
			
			sup.getId();
			List<ActualPixel> pixs = sup.getZMap();
			for (ActualPixel pixel : pixs) {
				pixel.xy = inst.getGlobalIndex(pixel.x, pixel.y);
				pixel.supix_idx = sup.getId();
				
				// from original image
				int[] rgb = Horse256Evaluator.getRGBArrfromInt(orgImg.getRGB(pixel.y, pixel.x) & 0xffffff);
				pixel.gtColorR = rgb[0];
				pixel.gtColorG = rgb[1];
				pixel.gtColorB = rgb[2];
				
				// from gt image
				int gtRgb = gtImg.getRGB(pixel.y, pixel.x) & 0xffffff;
				//System.out.println("gtRgb = " + gtRgb);
				pixel.gt_label = HorseLabel.GtColorToLabelIndex(gtRgb);
				
				// count label
				pixelLabelCntr.incrementCount(pixel.gt_label, 1);
			}
			
			// set major label for super pixel
			
			int maxLb = -1;
			double maxCnt = -1;
			for (Entry<Integer,Double> e : pixelLabelCntr.getEntrySet()) {
				if (e.getValue() > maxCnt) {
					maxCnt = e.getValue();
					maxLb = e.getKey();
				}
			}
			sup.setLabel(maxLb);
			//System.out.println(sup.getId() + " = " + sup.getLabel() + " " + maxCnt + " " + sup.getPixCnt());
			
		}
		
	}
	
/*
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
*/

	public List<Horse256Instance> loadFromListFile(String listfile) {
		
		List<Horse256Instance> imgs = new ArrayList<Horse256Instance>();
		
		List<String> allNames = Horse256Reader.getNameListFromFile(listfile);
		for (String nm : allNames) {
			Horse256Instance inst = initInstGivenName(nm);
			imgs.add(inst);
		}
		
		System.out.println("Load instance: " + allNames.size());
		return imgs;
	}
	
	public static void main(String[] args) {

		Horse256Reader reader = new Horse256Reader("/home/mc/workplace/imgseg/weim/pic256");
		List<Horse256Instance> insts = reader.loadFromListFile("/home/mc/workplace/imgseg/weim/pic256/all_list.txt");
		
		Horse256Evaluator evaluator = new Horse256Evaluator(reader.getDebugFolder());
		
		evaluator.evaluateSuperPixelGt(insts, true);
		
		//for (Horse256Instance ins : insts) {
		//	evaluator.dumpImage(ins, null, Horse256Evaluator.lbcolor);
		//}
	}
	
}
