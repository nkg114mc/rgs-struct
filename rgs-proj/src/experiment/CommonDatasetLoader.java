package experiment;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import general.AbstractLabelSet;
import imgseg.ImageDataReader;
import imgseg.ImageInstance;
import imgseg.ImageSegMain;
import multilabel.data.DatasetReader;
import multilabel.data.HwInstanceDataset;
import sequence.hw.HwDataReader;
import sequence.hw.HwInstance;
import sequence.hw.HwLabelSet;
import sequence.nettalk.NtkDataReader;
import sequence.nettalk.NtkPhonemeLabelSet;
import sequence.nettalk.NtkStressLabelSet;
import sequence.protein.ProteinDataReader;
import sequence.protein.ProteinLabelSet;
import sequence.twitterpos.TwitterDataReader;
import sequence.twitterpos.TwitterPosExample;
import sequence.twitterpos.TwitterPosLabelSet;

public class CommonDatasetLoader {
	
	public static final double DEFAULT_TRAIN_SPLIT_RATE = 0.8;
	
	/**
	 * The common dataset loader that shared by all datasets 
	 */
	
	// label sets
	HashMap<String, AbstractLabelSet> knownLabelSets = null;
	HashSet<String> multilabel = null;
	
	ImageDataReader msrc21Reader = new ImageDataReader("../msrc");
	
	HwLabelSet hwLbSet = new HwLabelSet();
	NtkPhonemeLabelSet phonemeLbSet = new NtkPhonemeLabelSet();
	NtkStressLabelSet stressLbSet = new NtkStressLabelSet();
	
	TwitterPosLabelSet twitterposLbSet = new TwitterPosLabelSet();
	ProteinLabelSet proteinLbSet = new ProteinLabelSet();
	
	//// common parameters
	
	// If this is true, the 
	// Otherwise
	public boolean doTrainDevSplitPrediction = false; // default to be false
	
	// train/dev split ratio
	// this number = train / (train + dev)
	public double trainRate = 0.8;
	
	
	
	public CommonDatasetLoader(boolean predictOnDev) {
		this();
		doTrainDevSplitPrediction = predictOnDev;
	}
	
	public CommonDatasetLoader(boolean predictOnDev, double trnRt) {
		this();
		doTrainDevSplitPrediction = predictOnDev;
		trainRate = trnRt;
	}
	
	public CommonDatasetLoader() {
		initLabelSet(); // init label set
	}
	
	
	
	/////////////////////////////////////////////////////////////////
	// for Sequence-labeling and multi-label dataset
	/////////////////////////////////////////////////////////////////
	
	private void initLabelSet() {
		
		if (knownLabelSets == null) {
			knownLabelSets = new HashMap<String, AbstractLabelSet>();
			knownLabelSets.put("hw-small", hwLbSet);
			knownLabelSets.put("hw-large", hwLbSet);
			knownLabelSets.put("hw-small-0", hwLbSet);
			knownLabelSets.put("hw-large-0", hwLbSet);
			knownLabelSets.put("hw-small-1", hwLbSet);
			knownLabelSets.put("hw-large-1", hwLbSet);
			knownLabelSets.put("hw-small-2", hwLbSet);
			knownLabelSets.put("hw-large-2", hwLbSet);
			knownLabelSets.put("hw-small-3", hwLbSet);
			knownLabelSets.put("hw-large-3", hwLbSet);
			knownLabelSets.put("hw-small-4", hwLbSet);
			knownLabelSets.put("hw-large-4", hwLbSet);
			knownLabelSets.put("hw-small-5", hwLbSet);
			knownLabelSets.put("hw-large-5", hwLbSet);
			knownLabelSets.put("hw-small-6", hwLbSet);
			knownLabelSets.put("hw-large-6", hwLbSet);
			knownLabelSets.put("hw-small-7", hwLbSet);
			knownLabelSets.put("hw-large-7", hwLbSet);
			knownLabelSets.put("hw-small-8", hwLbSet);
			knownLabelSets.put("hw-large-8", hwLbSet);
			knownLabelSets.put("hw-small-9", hwLbSet);
			knownLabelSets.put("hw-large-9", hwLbSet);
			knownLabelSets.put("nettalk_phoneme", phonemeLbSet);
			knownLabelSets.put("nettalk_stress", stressLbSet);
			knownLabelSets.put("twitterpos", twitterposLbSet);
			knownLabelSets.put("protein", proteinLbSet);
		}
		
		if (multilabel == null) {
			multilabel = new HashSet<String>();
			multilabel.add("yeast");
			multilabel.add("enron");
			multilabel.add("CAL500");
			multilabel.add("bibtex");
			multilabel.add("bookmarks-umass");
		}
	}
	
	public AbstractLabelSet getCommonLabelSet(String name) {
		
		if (knownLabelSets.containsKey(name)) {
			return knownLabelSets.get(name);
		} else if (multilabel.contains(name)) {	
			return null;
		} else {
			throw new RuntimeException("unknown dataset name to load labelset: [" + name + "]!");
		}
	}
	
	public List<List<HwInstance>> getHwDs(boolean isSmall, int folderIndex) {
		HwLabelSet hwLabels = (HwLabelSet)knownLabelSets.get("hw-small");
		assert (hwLabels != null);
		HwDataReader rder = new HwDataReader();
		List<List<HwInstance>> trtstInsts = rder.readData("../datasets/hw", hwLabels, folderIndex, isSmall);
		if (doTrainDevSplitPrediction) {
			List<List<HwInstance>> trdevInsts = constructDoubleListsForDev(trtstInsts);
			return trdevInsts;
		}
		return trtstInsts;
	}
	
	public List<List<HwInstance>> getNetStressDs() {
		NtkStressLabelSet stLabels = stressLbSet;
		NtkDataReader rder = new NtkDataReader();
		List<List<HwInstance>> trtstInsts = rder.readData("../datasets/nettalk_stress_train.txt", "../datasets/nettalk_stress_test.txt", stLabels.getLabels());
		if (doTrainDevSplitPrediction) {
			List<List<HwInstance>> trdevInsts = constructDoubleListsForDev(trtstInsts);
			return trdevInsts;
		}
		return trtstInsts; 
	}
	
	public List<List<HwInstance>> getNetPhonemeDs() {
		NtkPhonemeLabelSet phLabels = phonemeLbSet;
		NtkDataReader rder = new NtkDataReader();
		List<List<HwInstance>> trtstInsts = rder.readData("../datasets/nettalk_phoneme_train.txt", "../datasets/nettalk_phoneme_test.txt", phLabels.getLabels());
		if (doTrainDevSplitPrediction) {
			List<List<HwInstance>> trdevInsts = constructDoubleListsForDev(trtstInsts);
			return trdevInsts;
		}
		return trtstInsts;
	}
	
	public HwInstanceDataset getMultiLabelDs(String name) {
		
		// the folder that the multi-label data files were located
		DatasetReader.ML_XML_FOLDER = "../datasets/ML_datasets";
		DatasetReader.ML_CSV_FOLDER = "../datasets/ML_large_DataSets"; // for large datasets like bookmarks
		
		if (doTrainDevSplitPrediction) { // train/dev
			HwInstanceDataset originDs = DatasetReader.loadHwInstances(name);
			HwInstanceDataset trnDevDs = constructMultiLbDsForDev(originDs);
			return trnDevDs;
		} else { // train/test
			return DatasetReader.loadHwInstances(name);
		}
		
	}

	public List<List<HwInstance>> getProteinDs() {
		ProteinLabelSet phLabels = proteinLbSet;
		ProteinDataReader rder = new ProteinDataReader();
		List<List<HwInstance>> trtstInsts = rder.readData("../datasets/protein/protein/sparse.protein.11.train",
			                                              "../datasets/protein/protein/sparse.protein.11.test", phLabels.getLabels());
		return trtstInsts;
	}
	
	public List<List<HwInstance>> getTwitterPosDs() {
		
		String twtposRoot = "../datasets/infnet_twitterpos";
		
		//TwitterDataReader rdr = new TwitterDataReader();
		TwitterPosLabelSet labels = new TwitterPosLabelSet();

		String trainf = twtposRoot + "/oct27.traindev.proc.cnn.txt";
		String devf = twtposRoot + "/oct27.test.proc.cnn.txt";
		String testf = twtposRoot + "/daily547.proc.cnn.txt";
		List<HwInstance> trainExs = TwitterDataReader.readFromFileAsHwInstance(trainf,labels.getLabels());
		List<HwInstance> devExs = TwitterDataReader.readFromFileAsHwInstance(devf,labels.getLabels());
		List<HwInstance> testExs = TwitterDataReader.readFromFileAsHwInstance(testf,labels.getLabels());
		
		List<List<HwInstance>> twoExSet = new ArrayList<List<HwInstance>>();
		twoExSet.add(trainExs);
		twoExSet.add(testExs);
		//twoExSet.add(devExs);
		return twoExSet;
	}

	/////////////////////////////////////////////////////////////////
	//// for image segmentation
	/////////////////////////////////////////////////////////////////
	
	
	public List<List<ImageInstance>> getMSRC21TrainPredictInstances(String[] trnlabelNames, boolean trnDropVoid, String[] tstlabelNames, boolean tstDropVoid) {
		
		List<List<ImageInstance>> trnPred = new ArrayList<List<ImageInstance>>();
		List<ImageInstance> trainInsts = ImageSegMain.loadFromListFile(msrc21Reader, "../msrc/Train.txt", trnlabelNames, trnDropVoid);//labelNames, true);		
		List<ImageInstance> validInsts = ImageSegMain.loadFromListFile(msrc21Reader, "../msrc/Validation.txt", tstlabelNames, tstDropVoid);//labelNamesFull, true);
		List<ImageInstance> testInsts = ImageSegMain.loadFromListFile(msrc21Reader, "../msrc/Test.txt", tstlabelNames, tstDropVoid);//labelNamesFull, true);
		
		if (doTrainDevSplitPrediction) { // train/dev
			trnPred.add(trainInsts); // train
			trnPred.add(validInsts); // dev
		} else { // train/test
			trnPred.add(trainInsts); // train
			trnPred.add(testInsts);  // test
		}
		
		return trnPred;
	}
	
	public String getReaderDbgFolder() {
		return msrc21Reader.getDebugFolder();
	}
	
	/////////////////////////////////////////////////////////////////
	//// for coreference resolution
	/////////////////////////////////////////////////////////////////

	
	//public List<List<HwInstance>> getAce05PredictInstances() {
	//	return null;
	//}
	
	
	
	
	/////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////
	
	// for sequence labeling datasets
	public List<List<HwInstance>> constructDoubleListsForDev(List<List<HwInstance>> trtstInsts) {
		List<HwInstance> trainList = trtstInsts.get(0);
		List<HwInstance> testList = trtstInsts.get(1);
		double devRate = 1.0 - trainRate;
		
		int cnt = (int) (((double)testList.size()) * devRate);
		if (cnt <= 0) {
			throw new RuntimeException("[WARN] test set is too small!");
		}
		
		List<HwInstance> devList = new ArrayList<HwInstance>();
		for (int j = 0; j < cnt; j++) {
			devList.add(testList.get(j));
		}
		
		List<List<HwInstance>> trnPred = new ArrayList<List<HwInstance>>();
		trnPred.add(trainList);
		trnPred.add(devList);
		
		return trnPred;
	}
	
	// for multi-label datasets
	public HwInstanceDataset constructMultiLbDsForDev(HwInstanceDataset originDs) {
		
		List<HwInstance> trainList = originDs.getTrainExamples();
		List<HwInstance> testList = originDs.getTestExamples();
		
		double devRate = 1.0 - trainRate;
		
		int cnt = (int) (((double)testList.size()) * devRate);
		if (cnt <= 0) {
			throw new RuntimeException("[WARN] test set is too small!");
		}
		
		List<HwInstance> devList = new ArrayList<HwInstance>();
		for (int j = 0; j < cnt; j++) {
			devList.add(testList.get(j));
		}
		
		HwInstanceDataset devDs = new HwInstanceDataset(originDs.name, 
														trainList, 
														devList, 
				                                        originDs.getFeatureDimension(), 
				                                        originDs.getLabelDimension());
		return devDs;
	}
}
