package horse256;

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

import edu.berkeley.nlp.futile.util.Counter;
import edu.illinois.cs.cogcomp.sl.core.SLModel;
import experiment.ExperimentResult;
import experiment.TestingAcc;
import imgcnn.ActualPixel;
import imgseg.FracScore;
import imgseg.ImageInstance;
import imgseg.ImageSegLabel;
import imgseg.ImageSuperPixel;
import search.GreedySearcher;
import search.SearchResult;
import sequence.hw.HwOutput;
import sequence.hw.HwSearchInferencer;

public class Horse256Evaluator {
	
	public String debugDir;
	public static HashMap<Integer, Integer> rgbToLabels = null;
	public static int[][] lbcolor = new int[2][3];
	
	public Horse256Evaluator(String db) {
		debugDir = db;
		File rfdf = new File(debugDir);
		assert (rfdf.exists() && rfdf.isDirectory());
	}

	public static void main(String[] args) {

	}
	
	public void initRgbToLabel() {
		rgbToLabels = new HashMap<Integer, Integer>();
		rgbToLabels.put(0x000000, 0);
		rgbToLabels.put(0xffffff, 1);

		lbcolor[0] = new int[3];
		lbcolor[1] = new int[3];		
		Arrays.fill(lbcolor[0], 0x00);
		Arrays.fill(lbcolor[1], 0xff);
	}
	
	public ExperimentResult evaluate(List<Horse256Instance> images, SLModel model, boolean ifDump, int alterRestart) {
		initRgbToLabel();

		double total = 0;
		double acc = 0;
		double avgTruAcc = 0;
		
		FracScore[] gtscs = new FracScore[21];
		for (int j = 0; j < 21; j++) {
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
			SearchResult infrRe = schr.runSearchWithRestarts(model.wv, null, restartsN, images.get(i), gold, false); 
			HwOutput prediction = (HwOutput)(infrRe.predState.structOutput);
			
			FracScore[] gtResult = evaluateOneImage(images.get(i), prediction);
			accuFracScore(gtscs, gtResult);
			
			for (int j = 0; j < prediction.output.length; j++) {
				total += 1.0;
				if (prediction.output[j] == gold.output[j]){
					acc += 1.0;
				}
			}
			
			// sum true Acc
			avgTruAcc += infrRe.accuarcy;
			
			//if (ifDump) {
			//	dumpImage(images.get(i), prediction, labels);
			//}
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
		System.out.println("**********************");
		//printMSRCscore(fscores, labelSet);
		//printMSRCscore(gtscs, labelSet);
		
		// compute acc (same as printMSRCscore)
		//List<TestingAcc> avgAndGolfs = computeMSRCscore("SuprPix", fscores, labelSet);
		//List<TestingAcc> avgAndGolgt = computeMSRCscore("RealPix", gtscs, labelSet);
		
		//////////////////////////////////////
		
		ExperimentResult res = new ExperimentResult();
		//res.addAccBatch(avgAndGolfs);
		//res.addAccBatch(avgAndGolgt);

		res.addAcc(new TestingAcc("OverallAcc",  accuracy));
		res.addAcc(new TestingAcc("GenerationAcc", genAcc));
		
		return res;
	}
	
	public static FracScore[] evaluateOneImage(Horse256Instance img, HwOutput output) {
		
		FracScore[] scores = new FracScore[1];
		for (int j = 0; j < 21; j++) {
			scores[j] = new FracScore();
		}
		
		BufferedImage ystrImg = getGtImage(img);
		BufferedImage yhatImg = getLabelImage(img, output);
		
		assert (ystrImg.getWidth() == yhatImg.getWidth());
		assert (ystrImg.getHeight() == yhatImg.getHeight());
		
		for (int i = 0; i < ystrImg.getWidth(); i++) {
			for (int j = 0; j < ystrImg.getHeight(); j++) {
				
				int rgbStar = ystrImg.getRGB(i,j) & 0xffffff;
				int rgbHat = yhatImg.getRGB(i,j) & 0xffffff;

				int glabel = rgbToLabels.get(rgbStar);
				int pred = rgbToLabels.get(rgbHat);
				
				
				//if (glabel <= 20) { // OK
					double num = 0;
					double den = 1;
					if (pred == glabel) { // correct!
						num = 1;
					} else { // wrong
						num = 0;
					}
					scores[glabel].addNumDen(num, den);
				//}
				
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
	
	public static void printHorse32score(FracScore[] fscores, String[] labelSet) {
		
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
		
		System.out.println("---- Horse32 Evaluation ----------------");
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
		

		ArrayList<TestingAcc> averageAndGlobal = new ArrayList<TestingAcc>();
		averageAndGlobal.add(new TestingAcc(prefix + "ImgSegAverage", averageSc));
		averageAndGlobal.add(new TestingAcc(prefix + "ImgSegGlobal", globalSc));
		
		return averageAndGlobal;
	}
	

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

			//int centeredx = w / 2 - w_leg / 2;
			//g2.drawImage(legendImage, null, centeredx, 2 * (h_x + offset) + margin);
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
	
	
	public static BufferedImage getOriginImage(Horse256Instance inst)  {

		BufferedImage img = null;
		try {
			img = ImageIO.read(new File(inst.imgPath));
		} catch (IOException e) {
			e.printStackTrace();
		}
		return img;
	}
	
	public static BufferedImage getOriginImage2222(Horse256Instance inst)  {
		BufferedImage img2 = new BufferedImage(inst.getWidth(), inst.getHeight(), BufferedImage.TYPE_INT_RGB);
		ImageSuperPixel[] supixs = inst.getSuPixArr();
		for (ImageSuperPixel sup : supixs) {
			List<ActualPixel> pixs = sup.getZMap();
			for (ActualPixel pixel : pixs) {
				img2.setRGB(pixel.y, pixel.x, rgbToPixel(pixel.gtColorR, pixel.gtColorG, pixel.gtColorB));
			}
		}
		return img2;
	}
	
	public static BufferedImage getGtImage2222(Horse256Instance inst)  {
		int[][] lbcolor = { {0,0,0}, {255,255,0} };
		BufferedImage img2 = new BufferedImage(inst.getWidth(), inst.getHeight(), BufferedImage.TYPE_INT_RGB);
		ImageSuperPixel[] supixs = inst.getSuPixArr();
		for (ImageSuperPixel sup : supixs) {
			List<ActualPixel> pixs = sup.getZMap();
			for (ActualPixel pixel : pixs) {
				int lb = pixel.gt_label;
				img2.setRGB(pixel.y, pixel.x, rgbToPixel(lbcolor[lb][0], lbcolor[lb][1], lbcolor[lb][2]));
			}
		}
		return img2;
	}
	
	public static BufferedImage getGtImage(Horse256Instance inst)  {
		BufferedImage img = null;
		try {
			img = ImageIO.read(new File(inst.labelPath));
		} catch (IOException e) {
			e.printStackTrace();
		}
		return img;
	}
	
	public static BufferedImage getLabelImage(Horse256Instance x, HwOutput y) {

		int N = x.getWidth() * x.getHeight(); // # Pixels
		int[] rgbArray = new int[N];
		Arrays.fill(rgbArray,0);

		int h = x.getHeight();
		int w = x.getWidth();
		
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				int idx = x.getGlobalIndex(i, j);
				int lb = y.getOutput(idx);
				int	rgb = rgbToPixel(lbcolor[lb][0], lbcolor[lb][1], lbcolor[lb][2]);
				rgbArray[idx] = rgb;	
			}
		}

		return pixelsToImage(rgbArray, x.getWidth(), x.getHeight());
	}
	
	private static BufferedImage pixelsToImage(int[] rgbArray, int width, int height) {
		BufferedImage img = new BufferedImage(height, width, BufferedImage.TYPE_INT_RGB);
		img.setRGB(0, 0, height, width, rgbArray, 0, height);
		BufferedImage img2 = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				img2.setRGB(j, i, img.getRGB(i, j));
			}
		}
		return img2;
	}
	
	public static int[] getRGBArrfromInt(int rgb) {
		int[] arr = new int[3];
		arr[0] = (rgb & 0xff0000) >> 16;
		arr[1] = (rgb & 0xff00) >> 8;
		arr[2] = (rgb & 0xff);
		return arr;
	}
	
	
	
	
	
	/////////////////////////////
	//// Visualization only! ////
	/////////////////////////////
	
	public void dumpImage(Horse256Instance instance,
						  HwOutput predict,
						  int[][] lbcolor) {

		String imageOutName = instance.getName() + "_dbg.bmp";
		File imageOutPath = new File(debugDir, imageOutName);
		String imageOutFile = imageOutPath.getAbsolutePath();
		
		if (predict == null) {
			predict = (instance.getGoldOutput());
		}

		BufferedImage xImg = getOriginImage2222(instance); // getOriginImage(instance);
		BufferedImage spImg = getSuperPixelColoredImage2222(instance);
		BufferedImage ystrImg = getGtImage2222(instance); // getGtImage(instance);
		BufferedImage yhatImg = getPredictImage(instance, predict, lbcolor);

		BufferedImage tile = printImageTile(xImg, ystrImg, spImg, yhatImg);
		writeImage(tile, imageOutFile.toString());
	}
	
	public static BufferedImage getPredictImage(Horse256Instance x,
											    HwOutput y,
											    int[][] lbclr) {
		Random rnd = new Random();
		int[][] lbcolor = { {0,0,0}, {0, 255,255} };
		BufferedImage img2 = new BufferedImage(x.getWidth(), x.getHeight(), BufferedImage.TYPE_INT_RGB);
		ImageSuperPixel[] supixs = x.getSuPixArr();
		for (ImageSuperPixel sup : supixs) {
			int lb = sup.getLabel();
			int r = rnd.nextInt(256);
			int g = rnd.nextInt(256);
			int b = rnd.nextInt(256);
			List<ActualPixel> pixs = sup.getZMap();
			for (ActualPixel pixel : pixs) {
				if (lb == 0) {
					img2.setRGB(pixel.y, pixel.x, rgbToPixel(lbcolor[lb][0], lbcolor[lb][1], lbcolor[lb][2]));
				} else {
					img2.setRGB(pixel.y, pixel.x, rgbToPixel(r, g, b));
				}
				
			}
		}
		return img2;
	}
	
	public static BufferedImage getSuperPixelColoredImage(Horse256Instance x) {

		int N = x.getWidth() * x.getHeight(); // # Pixels
		int h = x.getHeight();
		int w = x.getWidth();
		
		int[] rgbArray = new int[N];
		Random rnd = new Random(); 

		Arrays.fill(rgbArray,0);
		ImageSuperPixel[] supixs = x.getSuPixArr();
		
		// Color the super-pixel with some random color
		for (ImageSuperPixel spix : supixs) {

			// Pick random RGB values
			int r = rnd.nextInt(256);
			int g = rnd.nextInt(256);
			int b = rnd.nextInt(256);
			int rgb = rgbToPixel(r, g, b);

			List<ActualPixel> pixels = spix.getZMap();
			for (ActualPixel p : pixels) {
				int idx = p.xy;
				rgbArray[idx] = rgb;
			}
		}

		return pixelsToImage(rgbArray, x.getWidth(), x.getHeight());
	}
	
	public static BufferedImage getSuperPixelColoredImage2222(Horse256Instance x) {

		Random rnd = new Random();
		BufferedImage img2 = new BufferedImage(x.getWidth(), x.getHeight(), BufferedImage.TYPE_INT_RGB);
		ImageSuperPixel[] supixs = x.getSuPixArr();
		for (ImageSuperPixel sup : supixs) {
			int lb = sup.getLabel();
			int r = rnd.nextInt(256);
			int g = rnd.nextInt(256);
			int b = rnd.nextInt(256);
			List<ActualPixel> pixs = sup.getZMap();
			for (ActualPixel pixel : pixs) {
				img2.setRGB(pixel.y, pixel.x, rgbToPixel(r, g, b));
			}
		}
		return img2;
	}

	
	
	
	// Test super pixel ground truth 2017-10-23
	// For upper bound test only
	
	public ExperimentResult evaluateSuperPixelGt(List<Horse256Instance> images, boolean ifDump) {
		
		int[][] lbcolor = { {0,0,0}, {0, 255,255} };
		
		double total = 0;
		double acc = 0;
		
		int tpfn[] = new int[4];
		Arrays.fill(tpfn, 0);
		
		System.out.println(" ----> Super Pixel Upper Bound Test! <----");
		
		for (int i = 0; i < images.size(); i++) {
			
			HwOutput gold = images.get(i).getGoldOutput();
			HwOutput prediction = gold;

			int[] gtResult = evaluateOneImageGtPic(images.get(i), prediction);
			//accuFracScore(gtscs, gtResult);
			tpfn[0] += gtResult[0];
			tpfn[1] += gtResult[1];
			tpfn[2] += gtResult[2];
			tpfn[3] += gtResult[3];
			
			for (int j = 0; j < prediction.output.length; j++) {
				total += 1.0;
				if (prediction.output[j] == gold.output[j]){
					acc += 1.0;
				}
			}
			
			if (ifDump) {
				dumpImage(images.get(i), prediction, lbcolor);
			}
		}
		
		double accuracy = acc / total;
		System.out.println("Accuracy = " + acc + " / " + total + " = " + accuracy);
		System.out.println("SuperPixel Acc = " + accuracy);
		
		System.out.println("~~~~~~~~~~~~~~~~~~~~");

		double sa = computeHamm(tpfn);
		double so = computeF1(tpfn);
		System.out.println("Sa = " + sa);
		System.out.println("So = " + so);
		
		//////////////////////////////////////
		ExperimentResult res = new ExperimentResult();
		return res;
	}
	
	public static int[] evaluateOneImageGtPic(Horse256Instance img, HwOutput output) {
		
		int tpfn[] = new int[4];
		Arrays.fill(tpfn, 0);
		
		ImageSuperPixel[] supixs = img.getSuPixArr();
		for (ImageSuperPixel spix : supixs) {
			int pred = output.getOutput(spix.hwsegIndex);
			List<ActualPixel> pixels = spix.getZMap();
			for (ActualPixel p : pixels) {
				int glabel = p.gt_label;
				
				if (pred > 0 && glabel > 0) { // correct!
					tpfn[0]++;
				} else if (pred == 0 && glabel == 0) { // correct!
					tpfn[1]++;
				} else if (pred > 0 && glabel == 0) { // incorrect!
					tpfn[2]++;
				} else if (pred == 0 && glabel > 0) { // incorrect!
					tpfn[3]++;
				}
			}
		}

		return tpfn;
	}
	
	public static double computeF1(int[] tpfn) {
		double pnum = (double)(tpfn[0]);
		double pden = (double)(tpfn[0] + tpfn[2]);
		double rnum = (double)(tpfn[0]);
		double rden = (double)(tpfn[0] + tpfn[3]);
		double f1 = (2 * pnum * rnum) / (pden * rnum + rden * pnum);
		return f1;
	}
	
	public static double computeHamm(int[] tpfn) {
		int total = tpfn[0] + tpfn[1] + tpfn[2] + tpfn[3];
		int crr = tpfn[0] + tpfn[1];
		double acc = ((double)(crr)) / ((double)(total));
		return acc;
	}
	
	public static void printHORSEscore(FracScore[] fscores, int nClass) {
		
		FracScore globalFrac = new FracScore();
		double totalClass = 0;
		double accuAcc = 0;
		
		for (int i = 0 ; i < nClass; i++) {
			FracScore.sumTo(globalFrac, fscores[i]);
			double acc = fscores[i].getFrac();
			accuAcc += acc;
			totalClass += 1;
		}
		
		double globalSc = globalFrac.getFrac();
		double averageSc = accuAcc / totalClass;
		
		System.out.println("---- Horser-256 Evaluation ----------------");
		for (int i = 0 ; i < nClass; i++) {
			System.out.println("  " + "label" + i + ": " + fscores[i].getFrac());
		}
		System.out.println("----------------------------------------");
		System.out.println("Average: " + averageSc);
		System.out.println("Global:  " + globalSc);
		System.out.println("----------------------------------------");
	}
}
