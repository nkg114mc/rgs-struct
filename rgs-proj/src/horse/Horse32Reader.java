package horse;

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
import multilabel.instance.Label;
import search.SearchResult;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;
import sequence.hw.HwSearchInferencer;

public class Horse32Reader {
	
	public static final String allFolder = "images";
	public static final String labelFolder = "musks";	
	
	String rootFolder = "";
	String suballFolder = "";
	String sublabelFolder = "";
	String subdebugFolder = "";
 
	public Horse32Reader(String rfdr) {
		rootFolder = rfdr;
		suballFolder = rootFolder + "/" + allFolder;
		sublabelFolder = rootFolder + "/" + labelFolder;
		subdebugFolder = rootFolder + "/" + "debug";
		
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
	
	public static SLProblem ExampleListToSLProblem(List<Horse32Instance> insts) {
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
	
	public Horse32Instance initInstGivenName(String imgName) {
		
		Horse32Instance inst = new Horse32Instance(imgName, null, Label.MULTI_LABEL_DOMAIN);
				
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
		// ground truth
		inst.labelPath = labelfd + "/" + imgName + ".bmp";  // bmp ground pixel label
		
		checkFiles(inst);
		
		//initSuperPixels(inst, dropVoid);
		
		// loading data
		//loadSuperPixels(inst, allfd);
		
		// refresh the hwsegments
		//inst.refreshHwSegments(dropVoid);
		
		return inst;    
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
*/
	private static void checkFiles(Horse32Instance inst) {
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
	
	/*
	public static void main(String[] args) {


		
		ImageSegLabel[] labels = ImageSegLabel.loadLabelFromFile("../msrc/imageseg_label_color_map.txt");
		String[] labelNames = ImageSegLabel.getStrLabelArr(labels, false);
		
		
		HorseReader reader = new HorseReader("../msrc");
		ImageSegEvaluator evaluator = new ImageSegEvaluator(reader.getDebugFolder());
		
		
		//List<String> allNames = getNameListFromFolder(reader.getAllFolder());
		List<String> allNames = HorseReader.getNameListFromFile("../msrc/Test.txt");
		
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
	}
	*/
	
}
