package imgseg;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import javax.imageio.ImageIO;

import edu.illinois.cs.cogcomp.sl.core.SLModel;
import experiment.ExperimentResult;
import experiment.TestingAcc;
import imgcnn.ActualPixel;
import search.GreedySearcher;
import search.SearchResult;
import sequence.hw.HwOutput;
import sequence.hw.HwSearchInferencer;

public class ImageSegEvaluator {
	
	public String debugDir;
	public static HashMap<Integer, Integer> rgbToLabels = null;
	
	public ImageSegEvaluator(String db) {
		debugDir = db;
		File rfdf = new File(debugDir);
		assert (rfdf.exists() && rfdf.isDirectory());
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}
	
	public static void initRgbToLabel(ImageSegLabel[] labels) {
		rgbToLabels = new  HashMap<Integer, Integer>();
		for (int i = 0; i < labels.length; i++) {
			int rgb = rgbToPixel(labels[i].r, labels[i].g, labels[i].b);
			rgbToLabels.put(rgb, labels[i].originIdx);
		}
	}
	
	public ExperimentResult evaluate(List<ImageInstance> images, SLModel model, ImageSegLabel[] labels, boolean ifDump, int alterRestart) { //String[] labelSet) {
		initRgbToLabel(labels);
		String[] labelSet = ImageSegLabel.getStrLabelArr(labels, false);
		
		double total = 0;
		double acc = 0;
		double avgTruAcc = 0;
		
		FracScore[] fscores = new FracScore[21];
		FracScore[] gtscs = new FracScore[21];
		for (int j = 0; j < 21; j++) {
			fscores[j] = new FracScore();
			gtscs[j] = new FracScore();
		}

		HwSearchInferencer searchInfr = (HwSearchInferencer)(model.infSolver);
		GreedySearcher schr = searchInfr.getSearcher();
		
		int restartsN = schr.randInitSize;
		if (alterRestart > 0) {
			restartsN = alterRestart;
		}
		
		System.out.println(" ----> Test Restart = " + restartsN + " <----");
		
		for (int i = 0; i < images.size(); i++) {
			
			HwOutput gold = images.get(i).getGoldOutput();
			//SearchResult infrRe = searchInfr.runSearchInference(model.wv, null, images.get(i), gold);
			SearchResult infrRe = schr.runSearchWithRestarts(model.wv, null, restartsN, images.get(i), gold, false); 
			HwOutput prediction = (HwOutput)(infrRe.predState.structOutput);
			
			
			FracScore[] oneResult = evaluateOneImage(images.get(i), prediction, labelSet);
			accuFracScore(fscores, oneResult);
			FracScore[] gtResult = evaluateOneImageGtPic(images.get(i), prediction, labels, labelSet);
			accuFracScore(gtscs, gtResult);
			
			for (int j = 0; j < prediction.output.length; j++) {
				total += 1.0;
				if (prediction.output[j] == gold.output[j]){
					acc += 1.0;
				}
			}
			
			// sum true Acc
			avgTruAcc += infrRe.accuarcy;
			
			if (ifDump) {
				dumpImage(images.get(i), prediction, labels);
			}
		}
		
		avgTruAcc = avgTruAcc / total;
		double accuracy = acc / total;
		System.out.println("Accuracy = " + acc + " / " + total + " = " + accuracy);
		
		double genAcc = avgTruAcc;
		double selAcc = genAcc - accuracy;
		
		if (genAcc < accuracy) {
			throw new RuntimeException("[ERROR]Generation accuracy is less than final output accuracy: " + genAcc + " < " + accuracy);
		}
		
		System.out.println("Generation Acc = " + genAcc);
		System.out.println("Selection AccDown = " + selAcc);
		
		printMSRCscore(fscores, labelSet);
		System.out.println("**********************");
		printMSRCscore(gtscs, labelSet);
		
		// compute acc (same as printMSRCscore)
		List<TestingAcc> avgAndGolfs = computeMSRCscore("SuprPix", fscores, labelSet);
		List<TestingAcc> avgAndGolgt = computeMSRCscore("RealPix", gtscs, labelSet);
		
		//////////////////////////////////////
		
		ExperimentResult res = new ExperimentResult();
		res.addAccBatch(avgAndGolfs);
		res.addAccBatch(avgAndGolgt);

		res.addAcc(new TestingAcc("OverallAcc",  accuracy));
		res.addAcc(new TestingAcc("GenerationAcc", genAcc));
		
		return res;
	}
	
	public static FracScore[] evaluateOneImage(ImageInstance img, HwOutput output, String[] labelSet) {
		
		FracScore[] scores = new FracScore[21];
		for (int j = 0; j < 21; j++) {
			scores[j] = new FracScore();
		}
		
		//ImageSuperPixel[] superPixels = img.getSuPixArr();
		
		for (int i = 0; i < img.size(); i++) {
			
			int supIdx = img.letterSegs.get(i).index;
			ImageSuperPixel supixel = img.getSuPix(supIdx);
			
			double pixelCnt = (double)(supixel.getPixCnt());
			int glabel = supixel.getLabel();
			int pred = output.getOutput(i);
			if (glabel <= 20) { // OK
				
				double num = 0;
				double den = pixelCnt;
				if (pred == glabel) { // correct!
					
					num = pixelCnt;
					
				} else { // wrong
					
					num = 0;
				}
				
				FracScore sc = new FracScore(num, den);
				FracScore.sumTo(scores[glabel], sc);
				
			} else {
				
			}
		}
		
		return scores;
	}
	
	public static FracScore[] evaluateOneImageGtPic(ImageInstance img, HwOutput output, ImageSegLabel[] labels, String[] labelSet) {
		
		FracScore[] scores = new FracScore[21];
		for (int j = 0; j < 21; j++) {
			scores[j] = new FracScore();
		}
		
		//HashMap<Integer, ArrayList<Integer>> zmap = loadSuperPixelMap(img.mapPath);
		BufferedImage ystrImg = getGtImage(img);
		//BufferedImage yhatImg = getLabelImage(img, output, zmap, labels);
		BufferedImage yhatImg = getLabelImage(img, output, labels);
		
		assert (ystrImg.getWidth() == yhatImg.getWidth());
		assert (ystrImg.getHeight() == yhatImg.getHeight());
		
		for (int i = 0; i < ystrImg.getWidth(); i++) {
			for (int j = 0; j < ystrImg.getHeight(); j++) {
				
				int rgbStar = ystrImg.getRGB(i,j) & 0xffffff;
				int rgbHat = yhatImg.getRGB(i,j) & 0xffffff;
				/*
				for (Entry<Integer, Integer> e : rgbToLabels.entrySet()) {
					System.out.println(e.toString());
				}
				System.out.println(rgbStar);
				System.out.println(rgbHat);
				*/
				if (rgbToLabels.containsKey(rgbStar)) { // color pixel is in the map
					int glabel = rgbToLabels.get(rgbStar);
					int pred = rgbToLabels.get(rgbHat);
					if (glabel <= 20) { // OK
						double num = 0;
						double den = 1;
						if (pred == glabel) { // correct!
							num = 1;
						} else { // wrong
							num = 0;
						}
						scores[glabel].addNumDen(num, den);
					}
				}
				
			}
		}
		
		return scores;
	}
	
	public static void accuFracScore(FracScore[] accum, FracScore[] onesc) {
		assert (accum.length == onesc.length);
		for (int i = 0; i < accum.length; i++) {
			FracScore.sumTo(accum[i], onesc[i]);
		}
	}
	
	public static void printMSRCscore(FracScore[] fscores, String[] labelSet) {
		
		FracScore globalFrac = new FracScore();
		double totalClass = 0;
		double accuAcc = 0;
		
		for (int i = 0 ; i < labelSet.length; i++) {
			FracScore.sumTo(globalFrac, fscores[i]);
			double acc = fscores[i].getFrac();
			accuAcc += acc;
			totalClass += 1;
		}
		
		double globalSc = globalFrac.getFrac();
		double averageSc = accuAcc / totalClass;
		
		System.out.println("---- MSRC-21 Evaluation ----------------");
		for (int i = 0 ; i < labelSet.length; i++) {
			System.out.println("  " + labelSet[i] + ": " + fscores[i].getFrac());
		}
		System.out.println("----------------------------------------");
		System.out.println("Average: " + averageSc);
		System.out.println("Global:  " + globalSc);
		System.out.println("----------------------------------------");
	}
	
	public static List<TestingAcc> computeMSRCscore(String prefix, FracScore[] fscores, String[] labelSet) {
		
		FracScore globalFrac = new FracScore();
		double totalClass = 0;
		double accuAcc = 0;
		
		for (int i = 0 ; i < labelSet.length; i++) {
			FracScore.sumTo(globalFrac, fscores[i]);
			double acc = fscores[i].getFrac();
			accuAcc += acc;
			totalClass += 1;
		}
		
		double globalSc = globalFrac.getFrac();
		double averageSc = accuAcc / totalClass;
		
		/*
		System.out.println("---- MSRC-21 Evaluation ----------------");
		for (int i = 0 ; i < labelSet.length; i++) {
			System.out.println("  " + labelSet[i] + ": " + fscores[i].getFrac());
		}
		System.out.println("----------------------------------------");
		System.out.println("Average: " + averageSc);
		System.out.println("Global:  " + globalSc);
		System.out.println("----------------------------------------");
		*/
		
		ArrayList<TestingAcc> averageAndGlobal = new ArrayList<TestingAcc>();
		averageAndGlobal.add(new TestingAcc(prefix + "ImgSegAverage", averageSc));
		averageAndGlobal.add(new TestingAcc(prefix + "ImgSegGlobal", globalSc));
		
		return averageAndGlobal;
	}
	
/*
	public static HashMap<Integer, ArrayList<Integer>> loadSuperPixelMap(String mapPath) {
		HashMap<Integer, ArrayList<Integer>> superToPixel = new HashMap<Integer, ArrayList<Integer>>();
		
		try {
			BufferedReader br = new BufferedReader(new FileReader(mapPath));
			String line;
			
			int lineIdx = 0;
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (!line.equals("")) {
					ArrayList<Integer> oneMap = new ArrayList<Integer>();
					
					String[] toks = line.split("\\,");
					int supIdx = Integer.parseInt(toks[0]);
					for (int i = 1; i < toks.length; i++) {
						oneMap.add(Integer.parseInt(toks[i]));
					}
					
					superToPixel.put(supIdx, oneMap);
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return superToPixel;
	}
*/
	
	public static BufferedImage printImageTile(BufferedImage x11, 
			BufferedImage x12,
			BufferedImage x21,
			BufferedImage x22) {

		
		try {
			File legend = new File("../msrc/msrc_legend.bmp");
			BufferedImage legendImage;
			legendImage = ImageIO.read(legend);

			int w_leg = legendImage.getWidth();
			int h_leg = legendImage.getHeight();

			int offset = 10;
			int margin = 10;
			int w_x = x11.getWidth();
			int h_x = x11.getHeight();

			int w = Math.max(2 * w_x, w_leg) + offset + 2 * margin;
			int h = 2 * h_x + h_leg + 2 * offset + 2 * margin;

			BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
			Graphics2D g2  = img.createGraphics();
			Color oldCol = g2.getColor();

			g2.setPaint(Color.WHITE);
			g2.fillRect(0, 0, w, h);
			g2.setColor(oldCol);

			g2.drawImage(x11, null, margin, margin);
			g2.drawImage(x12, null, w_x + offset + margin, margin);
			g2.drawImage(x21, null, margin, h_x + offset + margin);
			g2.drawImage(x22, null, w_x + offset + margin, h_x + offset + margin);

			int centeredx = w / 2 - w_leg / 2;
			g2.drawImage(legendImage, null, centeredx, 2 * (h_x + offset) + margin);
			g2.dispose();
			return img;

		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return null;
	}
	
	public static int rgbToPixel(int r, int g, int b) {
		return ((r << 16) | (g << 8) | (b));
	}

	public static void writeImage(BufferedImage img, String outFilePath) {
		try {
			ImageIO.write(img, "bmp", new File(outFilePath));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	
	
	
	/////////////////////////////
	//// Visualization only! ////
	/////////////////////////////
	
	public void dumpImage(ImageInstance instance,
						  HwOutput predict,
						  ImageSegLabel[] labels) {

		//String[] labelNames = ImageSegLabel.getStrLabelArr(labels, true);

		//System.out.println(instance.getName());
		String imageOutName = instance.getName() + "_dbg.bmp";
		File imageOutPath = new File(debugDir, imageOutName);
		String imageOutFile = imageOutPath.getAbsolutePath();

		//File mapf = new File(instance.mapPath);
		
		HashMap<Integer, ArrayList<Integer>> zmap = null; // loadSuperPixelMap(instance.mapPath);
		
		if (predict == null) {
			predict = (instance.getGoldOutput());
		}

		BufferedImage xImg = getOriginImage(instance);
		BufferedImage spImg = getSuperPixelColoredImage(instance);//, zmap);
		BufferedImage ystrImg = getGoldImage(instance);
		//BufferedImage yhatImg = getLabelImage(instance, predict, zmap, labels);
		BufferedImage yhatImg = getLabelImage(instance, predict,  labels);

		BufferedImage tile = printImageTile(xImg, ystrImg, spImg, yhatImg);
		writeImage(tile, imageOutFile.toString());
	}
	
	public static BufferedImage getOriginImage(ImageInstance inst)  {

		BufferedImage img = null;
		try {
			img = ImageIO.read(new File(inst.imgPath));
		} catch (IOException e) {
			e.printStackTrace();
		}
		return img;
	}
	
	public static BufferedImage getGtImage(ImageInstance inst)  {
		BufferedImage img = null;
		try {
			img = ImageIO.read(new File(inst.gtPath));
		} catch (IOException e) {
			e.printStackTrace();
		}
		return img;
	}

	public static BufferedImage getGoldImage(ImageInstance inst)  {
		BufferedImage gimg = null;
		try {
			gimg = ImageIO.read(new File(inst.gimgPath));
		} catch (IOException e) {
			e.printStackTrace();
		}
		return gimg;
	}
	
	public static BufferedImage getLabelImage(ImageInstance x,
											  HwOutput y,
											  //HashMap<Integer, ArrayList<Integer>> zmap,
											  ImageSegLabel[] labels) {

		int N = x.getWidth() * x.getHeight(); // # Pixels
		int[] rgbArray = new int[N];
		Arrays.fill(rgbArray,0);
		
		int h = x.getHeight();
		int w = x.getWidth();


		
		ImageSuperPixel[] supixs = x.getSuPixArr();
		
		// Color the super-pixel with some random color
		//for (Integer supIdx : zmap.keySet()) {
		for (ImageSuperPixel spix : supixs) {
			
			// get label
			//int yidx = x.getSuPix(supIdx).hwsegIndex;
			int yidx = spix.hwsegIndex;
			int rgb = 0;
			if (yidx >= 0) { // not void
				int lb = y.getOutput(yidx);
				rgb = rgbToPixel(labels[lb].r, labels[lb].g, labels[lb].b);
			}
			
			/*
			ArrayList<Integer> pixelIdxs = zmap.get(supIdx);
			for (Integer idx : pixelIdxs) {
				//if (lb <= 21) {
				//	rgb = 0;
				//}
				rgbArray[idx] = rgb;
			}*/
			List<ActualPixel> pixels = spix.getZMap();
			for (ActualPixel p : pixels) {
				//if (lb <= 21) {
				//	rgb = 0;
				//}
				int idx = p.xy;
				rgbArray[idx] = rgb;
			}
		}

		return pixelsToImage(rgbArray, x.getWidth(), x.getHeight());
	}
	
	/**
	 * Output super-pixel labels as an image
	 *
	 * z(i) = j : Super-pixel idx = `i` and pixel idx = `j`
	 */
	public static BufferedImage getSuperPixelColoredImage(ImageInstance x) { //, HashMap<Integer, ArrayList<Integer>> zmap) {

		int N = x.getWidth() * x.getHeight(); // # Pixels
		int h = x.getHeight();
		int w = x.getWidth();
		
		int[] rgbArray = new int[N];
		Random rnd = new Random(); 

		Arrays.fill(rgbArray,0);

		ImageSuperPixel[] supixs = x.getSuPixArr();
		
		// Color the super-pixel with some random color
		//for (Integer supIdx : zmap.keySet()) {
		for (ImageSuperPixel spix : supixs) {

			// Pick random RGB values
			int r = rnd.nextInt(256);
			int g = rnd.nextInt(256);
			int b = rnd.nextInt(256);
			int rgb = rgbToPixel(r, g, b);
			
			//if (supIdx > 0) { 
			//	rgb = 0;
			//}

			/*
			ArrayList<Integer> pixelIdxs = zmap.get(supIdx);
			for (Integer idx : pixelIdxs) {
				//if (lb <= 21) {
				//	rgb = 0;
				//}
				rgbArray[idx] = rgb;
			}*/
			List<ActualPixel> pixels = spix.getZMap();
			for (ActualPixel p : pixels) {
				//if (lb <= 21) {
				//	rgb = 0;
				//}
				int idx = p.xy;
				rgbArray[idx] = rgb;
			}
		}

		return pixelsToImage(rgbArray, x.getWidth(), x.getHeight());
	}
	
	private static BufferedImage pixelsToImage(int[] rgbArray, int width, int height) {
		BufferedImage img = new BufferedImage(height, width, BufferedImage.TYPE_INT_RGB);
		img.setRGB(0, 0, height, width, rgbArray, 0, height);
		
		BufferedImage img2 = new BufferedImage(width, height,  BufferedImage.TYPE_INT_RGB);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				img2.setRGB(j, i, img.getRGB(i, j));
			}
		}
		//BufferedImage img = new BufferedImage(width, height,  BufferedImage.TYPE_INT_RGB);
		//img.setRGB(0, 0, width, height, rgbArray, 0, width);
		return img2;
	}
	
	
	///////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////
	
	// Test super pixel ground truth 2017-10-23
	// For upper bound test only
	
	public ExperimentResult evaluateSuperPixelGt(List<ImageInstance> images, ImageSegLabel[] labels, boolean ifDump) {
		initRgbToLabel(labels);
		String[] labelSet = ImageSegLabel.getStrLabelArr(labels, false);
		
		double total = 0;
		double acc = 0;
		double avgTruAcc = 0;
		
		FracScore[] fscores = new FracScore[21];
		FracScore[] gtscs = new FracScore[21];
		for (int j = 0; j < 21; j++) {
			fscores[j] = new FracScore();
			gtscs[j] = new FracScore();
		}

		
		System.out.println(" ----> Super Pixel Upper Bound Test! <----");
		
		for (int i = 0; i < images.size(); i++) {
			
			HwOutput gold = images.get(i).getGoldOutput();
			HwOutput prediction = gold;
			
			
			FracScore[] oneResult = evaluateOneImage(images.get(i), prediction, labelSet);
			accuFracScore(fscores, oneResult);
			FracScore[] gtResult = evaluateOneImageGtPic(images.get(i), prediction, labels, labelSet);
			accuFracScore(gtscs, gtResult);
			
			for (int j = 0; j < prediction.output.length; j++) {
				total += 1.0;
				if (prediction.output[j] == gold.output[j]){
					acc += 1.0;
				}
			}
			
			if (ifDump) {
				dumpImage(images.get(i), prediction, labels);
			}
		}
		
		//avgTruAcc = avgTruAcc / total;
		double accuracy = acc / total;
		System.out.println("Accuracy = " + acc + " / " + total + " = " + accuracy);
		/*
		double genAcc = avgTruAcc;
		double selAcc = genAcc - accuracy;
		
		if (genAcc < accuracy) {
			throw new RuntimeException("[ERROR]Generation accuracy is less than final output accuracy: " + genAcc + " < " + accuracy);
		}
		*/
		//System.out.println("Generation Acc = " + genAcc);
		//System.out.println("Selection AccDown = " + selAcc);
		System.out.println("SuperPixel Acc = " + accuracy);
		
		printMSRCscore(fscores, labelSet);
		System.out.println("**********************");
		printMSRCscore(gtscs, labelSet);
		
		// compute acc (same as printMSRCscore)
		List<TestingAcc> avgAndGolfs = computeMSRCscore("SuprPix", fscores, labelSet);
		List<TestingAcc> avgAndGolgt = computeMSRCscore("RealPix", gtscs, labelSet);
		
		//////////////////////////////////////
		
		
		ExperimentResult res = new ExperimentResult();
		res.addAccBatch(avgAndGolfs);
		res.addAccBatch(avgAndGolgt);

		//res.addAcc(new TestingAcc("OverallAcc",  accuracy));
		//res.addAcc(new TestingAcc("GenerationAcc", genAcc));
		
		return res;
	}
}
