package imgcnn;

import java.util.ArrayList;
import java.util.List;

import imgseg.ImageDataReader;
import imgseg.ImageInstance;
import imgseg.ImageSegEvaluator;
import imgseg.ImageSegLabel;
import imgseg.ImageSegMain;

public class CnnSuperPixelUpperBound {

	public static void main(String[] args) {
		TestUpperBound(args);
	}

	public static void TestUpperBound(String[] args) {
		try {
			
			ImageSegLabel[] labels = ImageSegLabel.loadLabelFromFile("../msrc/imageseg_label_color_map.txt");
			String[] labelNames = ImageSegLabel.getStrLabelArr(labels, false);
			String[] labelNamesFull = ImageSegLabel.getStrLabelArr(labels, true);
			
			ImageSegEvaluator.initRgbToLabel(labels);

			ImageCNNReader reader = new ImageCNNReader("../msrc");
			ImageSegEvaluator evaluator = new ImageSegEvaluator(reader.getDebugFolder());

			List<ImageInstance> trainInsts = ImageCNNMain.loadFromListFile(reader, "../msrc/Train.txt", labelNames, true);
			List<ImageInstance> testInsts = ImageCNNMain.loadFromListFile(reader, "../msrc/Test.txt", labelNamesFull, true);
			
			
			List<ImageInstance> totalInsts = new ArrayList<ImageInstance>();
			totalInsts.addAll(trainInsts);
			totalInsts.addAll(testInsts);
			
			ImageSegMain.computeAvgStructSize(trainInsts);

			evaluator.evaluateSuperPixelGt(testInsts, labels, true);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}


}
