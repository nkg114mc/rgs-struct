package horse;

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
import imgseg.FracScore;
import search.GreedySearcher;
import search.SearchResult;
import sequence.hw.HwOutput;
import sequence.hw.HwSearchInferencer;

public class Horse32Evaluator {
	
	public String debugDir;
	public static HashMap<Integer, Integer> rgbToLabels = null;
	static int[][] lbcolor = new int[2][3];
	
	public Horse32Evaluator(String db) {
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
		Arrays.fill(lbcolor[0], 0x000000);
		Arrays.fill(lbcolor[1], 0xffffff);
	}
	
	public ExperimentResult evaluate(List<Horse32Instance> images, SLModel model, boolean ifDump, int alterRestart) {
		initRgbToLabel();

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
	
	public static FracScore[] evaluateOneImage(Horse32Instance img, HwOutput output) {
		
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
	
	
	public static BufferedImage getOriginImage(Horse32Instance inst)  {

		BufferedImage img = null;
		try {
			img = ImageIO.read(new File(inst.imgPath));
		} catch (IOException e) {
			e.printStackTrace();
		}
		return img;
	}
	
	public static BufferedImage getGtImage(Horse32Instance inst)  {
		BufferedImage img = null;
		try {
			img = ImageIO.read(new File(inst.labelPath));
		} catch (IOException e) {
			e.printStackTrace();
		}
		return img;
	}
	
	public static BufferedImage getLabelImage(Horse32Instance x, HwOutput y) {

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

}
