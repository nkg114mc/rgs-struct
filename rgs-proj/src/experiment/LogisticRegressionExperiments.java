package experiment;

import java.util.List;

import edu.berkeley.nlp.futile.fig.basic.Option;
import edu.berkeley.nlp.futile.fig.exec.Execution;
import experiment.RndLocalSearchExperiment.DataSetName;
import imgseg.ImageDataReader;
import imgseg.ImageInstance;
import imgseg.ImageSegEvaluator;
import imgseg.ImageSegLabel;
import imgseg.ImageSegMain;
import init.HwFastSamplingRndGenerator;
import init.ImageSegSamplingRndGenerator;
import init.MultiLabelSamplingRndGenerator;
import init.SeqSamplingRndGenerator;
import multilabel.data.Dataset;
import multilabel.data.DatasetReader;
import multilabel.instance.Label;
import multilabel.utils.UtilFunctions;
import sequence.hw.HwDataReader;
import sequence.hw.HwInstance;
import sequence.hw.HwLabelSet;
import sequence.nettalk.NtkDataReader;
import sequence.nettalk.NtkPhonemeLabelSet;
import sequence.nettalk.NtkStressLabelSet;

public class LogisticRegressionExperiments implements Runnable {
	

	@Option(gloss = "Dataset name")
	public static DataSetName name = DataSetName.HW_SMALL;
	@Option(gloss = "Output debug info in logistic regression or not")
	public static boolean debug = false;
	@Option(gloss = "Iteration number")
	public static int IterationNum = -1;

	
	public static void main(String[] args) {
		LogisticRegressionExperiments man = new LogisticRegressionExperiments();
		Execution.run(args, man); // add .class here if that class should receive command-line args
	}

	public static void runOtherSeqData(String dsName, List<List<HwInstance>> trtstInsts, String[] domainVal) {
		SeqSamplingRndGenerator trn_g = SeqSamplingRndGenerator.trainLogisticModel(dsName, trtstInsts.get(0), trtstInsts.get(1), domainVal, debug, IterationNum);
		String logsPath = "../logistic_models/" + dsName + ".logistic";
		UtilFunctions.saveObj(trn_g, logsPath);
		System.out.println("Seq All Done!");
	}
	
	public static void runLinearSeqData(String dsName, List<List<HwInstance>> trtstInsts, String[] domainVal) {
		HwFastSamplingRndGenerator trn_g = HwFastSamplingRndGenerator.trainUnaryLinearModel(dsName, trtstInsts.get(0), trtstInsts.get(1), domainVal);
		String logsPath = "../logistic_models/" + dsName + ".logistic";
		UtilFunctions.saveObj(trn_g, logsPath);
		System.out.println("Seq All Done!");
	}
	
	public static void runMulLbData(String datasetName) {

		String dsName = datasetName;
		DatasetReader dataSetReader = new DatasetReader();
		Dataset ds = dataSetReader.readDefaultDataset(dsName);
		
		// load data
		List<List<HwInstance>> trtstInsts =  DatasetReader.convertToHwInstances(ds);
		System.out.println("List count = " + trtstInsts.size());
		
		MultiLabelSamplingRndGenerator trn_g = MultiLabelSamplingRndGenerator.trainLogisticModel(dsName, trtstInsts.get(0), trtstInsts.get(1), Label.MULTI_LABEL_DOMAIN, debug);
		String logsPath = "../logistic_models/" + dsName + ".logistic";
		UtilFunctions.saveObj(trn_g, logsPath);
		
		System.out.println("MultiLabel All Done!");
	}

	@Override
	public void run() {
		
		DataSetName ds = name;
		
		// Sequence
		if (ds == DataSetName.HW_SMALL) {
			runHandWritingSmall();
		} else if (ds == DataSetName.HW_LARGE) {
			runHandWritingLarge();
		} else if (ds == DataSetName.NETTALK_STREE) {
			runNetTalkStree();
		} else if (ds == DataSetName.NETTALK_PHONEME) {
			runNetTalkPhoneme();
		
		// Multi-label
		} else if (ds == DataSetName.YEAST) {
			runMulLbData("yeast");
		} else if (ds == DataSetName.ENRON) {
			runMulLbData("enron");
		} else if (ds == DataSetName.CAL500) {
			runMulLbData("CAL500");
		} else if (ds == DataSetName.COREL5K) {
			runMulLbData("Corel5k");
		} else if (ds == DataSetName.MEDIAMILL) {
			runMulLbData("mediamill");

		} else if (ds == DataSetName.BIBTEX) {
			runMulLbData("bibtex");
		} else if (ds == DataSetName.BOOKMARKS) {
			runMulLbData("bookmarks");
			
		// Image Segmentation
		} else if (ds == DataSetName.MSRC21) {
			runMSRC21();
		} else {
			throw new RuntimeException("Unknown dataset name ...");
		}
		
		System.out.println("Done.");
	}
	
	public static void runHandWritingSmall() {
		HwLabelSet hwLabels = new HwLabelSet();
		HwDataReader rder = new HwDataReader();
		for (int fdIdx = 0; fdIdx < 10; fdIdx++) {
			System.out.println("Train model HW-Small folder " + fdIdx);
			String fdName = "hw-small-" + String.valueOf(fdIdx);
			List<List<HwInstance>> trtstInsts = rder.readData("../datasets/hw", hwLabels, fdIdx, true);
			runLinearSeqData(fdName,  trtstInsts, hwLabels.getLabels());
			//runOtherSeqData("HW-Small",  trtstInsts, hwLabels.getLabels());
		}
	}
	
	public static void runHandWritingLarge() {
		HwLabelSet hwLabels = new HwLabelSet();
		HwDataReader rder = new HwDataReader();
		
		for (int fdIdx = 0; fdIdx < 10; fdIdx++) {
			System.out.println("Train model HW-Large folder " + fdIdx);
			String fdName = "hw-large-" + String.valueOf(fdIdx);
			List<List<HwInstance>> trtstInsts = rder.readData("../datasets/hw", hwLabels, fdIdx, false);
			runLinearSeqData(fdName, trtstInsts, hwLabels.getLabels());
			//runOtherSeqData("HW-Large", trtstInsts, hwLabels.getLabels());
		}
	}
	
	public static void runNetTalkStree() {
		NtkStressLabelSet stLabels = new NtkStressLabelSet();
		NtkDataReader rder = new NtkDataReader();
		List<List<HwInstance>> trtstInsts = rder.readData("../datasets/nettalk_stress_train.txt", "../datasets/nettalk_stress_test.txt", stLabels.getLabels());
		System.out.println("List count = " + trtstInsts.size());
		runOtherSeqData("nettalk_stress", trtstInsts, stLabels.getLabels());
	}
	
	public static void runNetTalkPhoneme() {
		NtkPhonemeLabelSet phLabels = new NtkPhonemeLabelSet();
		NtkDataReader rder = new NtkDataReader();
		List<List<HwInstance>> trtstInsts = rder.readData("../datasets/nettalk_phoneme_train.txt", "../datasets/nettalk_phoneme_test.txt", phLabels.getLabels());
		runOtherSeqData("nettalk_phoneme", trtstInsts, phLabels.getLabels());
	}
/*
	public void runYeast() {
		
	}
	
	public void runEnron() {
		
	}
	
	public void runCal500() {
		
	}
	
	public void runCorel5k() {
		
	}
*/
	public static void runMSRC21() {
		try {
			ImageSegLabel[] labels = ImageSegLabel.loadLabelFromFile("../msrc/imageseg_label_color_map.txt");
			String[] labelNames = ImageSegLabel.getStrLabelArr(labels, false);
			String[] labelNamesFull = ImageSegLabel.getStrLabelArr(labels, true);

			ImageSegEvaluator.initRgbToLabel(labels);

			ImageDataReader reader = new ImageDataReader("../msrc");
			//ImageSegEvaluator evaluator = new ImageSegEvaluator(reader.getDebugFolder());


			//List<ImageInstance> trainInsts = ImageSegMain.loadFromListFile(reader, "../msrc/Train3.txt", labelNames, true);
			//List<ImageInstance> trainInsts = ImageSegMain.loadFromListFile(reader, "../msrc/Train-small.txt", labelNames, true);
			//List<ImageInstance> trainInsts = ImageSegMain.loadFromListFile(reader, "../msrc/TrainValidation.txt", labelNames, true);
			List<ImageInstance> trainInsts = ImageSegMain.loadFromListFile(reader, "../msrc/Train.txt", labelNames, true);


			List<ImageInstance> testInsts = ImageSegMain.loadFromListFile(reader, "../msrc/Test.txt", labelNamesFull, true);


			//ImageSegSamplingRndGenerator genr = ImageSegSamplingRndGenerator.trainLogisticModel("msrc21", trainInsts, testInsts, labelNames, false, -1);
			ImageSegSamplingRndGenerator genr = ImageSegSamplingRndGenerator.trainUnaryLinearModel("msrc21", trainInsts, testInsts, labelNames);
			
			String logsPath = "../logistic_models/" + "msrc21" + ".logistic";
			UtilFunctions.saveObj(genr, logsPath);
			System.out.println("Image All Done!");
		} catch (Exception e) {
			e.printStackTrace();
		}

	}


	
}


