package multilabel.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import multilabel.instance.Example;
import multilabel.instance.Label;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;
import sequence.hw.HwSegment;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class DatasetReader {
	/*read the CSV files (both feature and label)*/
	//public static final String TRAIN_FEATURE_FILE_LOCATION = "DataSets/CAL500-train-feature.csv";
	//public static final String TRAIN_LABEL_FILE_LOCATION = "DataSets/CAL500-train-label.csv";
	//public static final String TEST_FEATURE_FILE_LOCATION = "DataSets/CAL500-test-feature.csv";
	//public static final String TEST_LABEL_FILE_LOCATION = "DataSets/CAL500-test-label.csv";
	
	public static String ML_XML_FOLDER = "../datasets/ML_datasets";
	public static String ML_CSV_FOLDER = "../datasets/ML_large_DataSets";
	
	public static final String CSV_SPLIT = ",";
	
	public static final int ARFF_READER = 1;
	public static final int CSV_READER  = 2;
	public static HashMap<String, Integer> readerMethod = getReaderMethod();//new HashMap<String, Integer>();
	public static HashMap<String, Boolean> readerIsDense = getReaderSparse();
	
	public class DatasetInfo {
		public int featDem;
		public int labelDem;
		public int nExample;
		public ArrayList<Example> exmples; 
	}
	
	public class HwInstanceInfo {
		public int featDem;
		public int labelDem;
		public int nExample;
		public List<HwInstance> exmples; 
	}
	
	public DatasetReader() {

		
	}
	
	public static HashMap<String, Integer> getReaderMethod() {
		HashMap<String, Integer> eaderMethod = new HashMap<String, Integer>();
		// arff files
		eaderMethod.put("scene", ARFF_READER);
		eaderMethod.put("emotions", ARFF_READER);
		eaderMethod.put("medical", ARFF_READER);
		eaderMethod.put("LLOG", ARFF_READER);
		eaderMethod.put("enron", ARFF_READER);
		eaderMethod.put("yeast", ARFF_READER);
		// csv files
		eaderMethod.put("CAL500", CSV_READER);
		eaderMethod.put("bibtex", CSV_READER);
		eaderMethod.put("bookmarks", CSV_READER);
		eaderMethod.put("bookmarks-umass", CSV_READER);
		eaderMethod.put("Corel5k", CSV_READER);
		eaderMethod.put("mediamill", CSV_READER);
		eaderMethod.put("delicious", CSV_READER);
		eaderMethod.put("eurlex-ev-fold1", CSV_READER);
		return eaderMethod;
	}
	
	public static HashMap<String, Boolean> getReaderSparse() {
		HashMap<String, Boolean> sprMethod = new HashMap<String, Boolean>();
		// arff files
		sprMethod.put("scene", false);
		sprMethod.put("emotions", false);
		sprMethod.put("medical", false);
		sprMethod.put("LLOG", false);
		sprMethod.put("enron", false);
		sprMethod.put("yeast", false);
		// csv files
		sprMethod.put("CAL500",    false);
		sprMethod.put("bibtex",    false);
		sprMethod.put("bookmarks", false);
		sprMethod.put("bookmarks-umass", false);
		sprMethod.put("Corel5k",   false);
		sprMethod.put("mediamill", false);
		sprMethod.put("delicious", false);
		sprMethod.put("eurlex-ev-fold1", false);
		return sprMethod;
	}
	
	public Dataset loadDataSetCSV(String trainFeatCSVpath, String trainLabelCSVpath, 
			                      String testFeatCSVpath, String testLabelCSVpath) {
		
		DatasetInfo trainInfo = readCSVseperate(trainFeatCSVpath, trainLabelCSVpath);
		System.out.println("------");
		DatasetInfo testInfo = readCSVseperate(testFeatCSVpath, testLabelCSVpath);
		
		// havea check!
		if (trainInfo.featDem !=  testInfo.featDem ||
			trainInfo.labelDem != testInfo.labelDem) {
			System.out.println("Dimension inconsistent in train and test!");
		}
		System.out.println("Load train " + trainInfo.nExample + ", test " + testInfo.nExample);
		System.out.println("Feature dimension " + trainInfo.featDem + " label dimension " + trainInfo.labelDem);
		
		Dataset dset = new Dataset("Cal500", trainInfo.exmples, testInfo.exmples, trainInfo.featDem, trainInfo.labelDem);
		return dset;
	}
	
	// read examples from csv files.
	// the feature and label csv files are split
	public DatasetInfo readCSVseperate(String featureLocation, String labelLocation) {
		
		ArrayList<Example> instances = new ArrayList<Example>(); 
		BufferedReader bufferedReaderFeature = null;
		BufferedReader bufferedReaderLabel = null;
		//BufferedWriter bufferedWriter = null;
		String lineFeature = "";
		String lineLabel = "";
		//String csvSplitBy = ",";

		
		int noOfLabels = -1;
		int noOfFeatures = -1;
		int exmpCnt = 0;
		
		try {

			bufferedReaderFeature = new BufferedReader(new FileReader(featureLocation));
			bufferedReaderLabel = new BufferedReader(new FileReader(labelLocation));
			//bufferedWriter = new BufferedWriter(new FileWriter(fileName));
			bufferedReaderFeature.readLine();//as the first line contains column names
			bufferedReaderLabel.readLine();//as the first line contains column names

			while ((lineFeature = bufferedReaderFeature.readLine()) != null && 
				   (lineLabel = bufferedReaderLabel.readLine()) != null) {
				
				/*
				String[] splittedLineFeature = lineFeature.split(csvSplitBy);
				noOfFeatures = splittedLineFeature.length;
				String[] splittedLineLabel = lineLabel.split(csvSplitBy);
				noOfLabels = splittedLineLabel.length;
				*/
				
				Example exmp = new Example();
				exmp.loadFromCsvString(exmpCnt, lineFeature, lineLabel); // parse a example line string
				
				instances.add(exmp);
				
				noOfLabels = exmp.labelDim();
				noOfFeatures = exmp.featDim();
				//System.out.println(exmp.featDim() + " " + exmp.labelDim());
				
				
				/*
				int labelCount = 1;
				for(String string : splittedLineLabel){
					
					
					
					
					//System.out.println("current Label: "+ string);
					if(string.equals("1")){
						bufferedWriter.write("1 qid:" +qidCounter);
					}
					else{
						bufferedWriter.write("0 qid:" +qidCounter);
					}
					//System.out.println(labelCount + " " + noOfLabels);
					int featureIndex = (labelCount - 1) * noOfFeatures + 1;
					//System.out.println("feature index value start: " + featureIndex);
//						for(int i = 1; i<= (labelCount - 1) * noOfLabels; ++i){
//							bufferedWriter.write(" "+ i+ ":0");
//						}						
					for(String stringFeature : splittedLineFeature){
						bufferedWriter.write(" "+ featureIndex+ ":" +stringFeature);
						featureIndex++;
					}
					//System.out.println(labelCount + " " + featureIndex);
//						for(int i = featureIndex; i<= noOfFeatures * noOfLabels; ++i){
//							bufferedWriter.write(" "+ i+ ":0");
//						}						
					bufferedWriter.write("\n");
				}


				labelCount++;*/
				//}
				//System.out.println(labelCount);
				exmpCnt++;
			}
			
			System.out.println("Loaded examplea: " + exmpCnt);
			
			bufferedReaderFeature.close();
			bufferedReaderLabel.close();

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (bufferedReaderFeature != null) {
				try {
					bufferedReaderFeature.close();
					bufferedReaderLabel.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}


		DatasetInfo info = new DatasetInfo();
		info.nExample = exmpCnt;
		info.featDem = noOfFeatures;
		info.labelDem = noOfLabels;
		info.exmples = instances;
		
		countSparsity(instances); // count 
		
		return info;
	}
	
	////////////////////////////////////////////////////
	////////////////////////////////////////////////////
	////////////////////////////////////////////////////
	
	
	public static ArrayList<String> loadParseXml(String xmlPath) {

		ArrayList<String> results = new ArrayList<String>();
		
		try {

			File fXmlFile = new File(xmlPath);
			DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
			DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
			Document doc = dBuilder.parse(fXmlFile);

			//optional, but recommended
			//read this - http://stackoverflow.com/questions/13786607/normalization-in-dom-parsing-with-java-how-does-it-work
			doc.getDocumentElement().normalize();
			
			Element rootEle = doc.getDocumentElement();
			NodeList nList = rootEle.getChildNodes();
			for (int temp = 0; temp < nList.getLength(); temp++) {
				Node nNode = nList.item(temp);
				//System.out.println("\nCurrent Element :" + nNode.getNodeName());

				if (nNode.getNodeName().equals("label")) { //  == Node.ELEMENT_NODE
					Element eElement = (Element) nNode;
					String lname = eElement.getAttribute("name");
					//System.out.println("Label Name : " + lname);
					results.add(lname);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return results;
	}
	
	
	public static Example getExampleFromWekaInstance(int cnt, Instance inst, ArrayList<String> labelAttrNames) {
		
		ArrayList<Label> ls = new ArrayList<Label>(); 
		ArrayList<Double> fs = new ArrayList<Double>();
		
		int lcnt = 0;
		int fcnt = 0;
		HashSet<String> nameSet = new HashSet<String>(labelAttrNames);
		for (int i = 0; i < inst.numAttributes(); i++) {
			Attribute atr = inst.attribute(i);
			if (nameSet.contains(atr.name())) { // is label
				ls.add(new Label(lcnt, (int)(inst.value(i))));
				lcnt++;
			} else { // is feature
				fs.add(inst.value(i));
				fcnt++;
			}
		}
		
		Example ex = new Example(cnt, ls, fs);
		return ex;
	}
	
	
	public DatasetInfo readArff(String arffPath, String xmlPath) {

		ArrayList<String> labelAttrNames = loadParseXml(xmlPath);
		DatasetInfo info = null;

		BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(arffPath));
			ArffReader arff = new ArffReader(reader);
			Instances data = arff.getData();

			ArrayList<Example> exs = new ArrayList<Example>();
			int exSize = data.numInstances();
			int totalAttrCnt = data.numAttributes();
			int labelCnt = labelAttrNames.size();
			for (int i = 0; i < exSize; i++) {
				Example ex = getExampleFromWekaInstance(i, data.instance(i), labelAttrNames);
				//System.out.println(ex.toString());
				exs.add(ex);
			}
			
			//////////////////////////////

			info = new DatasetInfo();
			info.nExample = exs.size();
			info.featDem = (totalAttrCnt - labelCnt);
			info.labelDem = labelCnt;
			info.exmples = exs;

			countSparsity(exs); // count 


		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return info;
	}
	
	public Dataset loadDataSetArff(String name, String trainArffPath, String testArffPath, String xmlPath) {

		DatasetInfo trainInfo = readArff(trainArffPath, xmlPath);
		System.out.println("------");
		DatasetInfo testInfo = readArff(testArffPath, xmlPath);

		System.out.println("Load train " + trainInfo.nExample + ", test " + testInfo.nExample);
		System.out.println("Feature dimension " + trainInfo.featDem + " label dimension " + trainInfo.labelDem);

		Dataset dset = new Dataset(name, trainInfo.exmples, testInfo.exmples, trainInfo.featDem, trainInfo.labelDem);
		return dset;
	}
	
	
	
	public static void countSparsity(ArrayList<Example> exs) {
		HashMap<Integer, Integer> sparsityCount = new HashMap<Integer, Integer>();
		
		int totalOne = 0;
		int total = 0;
		for (int i = 0; i < exs.size(); i++) {
			Example ex = exs.get(i);
			int oneCnt = 0;
			for (int j = 0; j < ex.labelDim(); j++) {
				Label lb = ex.getLabel().get(j);
				if (lb.value > 0) oneCnt++;
			}
			//////////////////////////
			Integer sprCnt = sparsityCount.get(oneCnt);
			if (sprCnt == null) {
				sparsityCount.put(oneCnt, 1);
			} else {
				int newCnt = sprCnt.intValue() + 1;
				sparsityCount.put(oneCnt, newCnt);
			}
			
			totalOne += oneCnt;
			total += ex.labelDim();
		}
		
		ArrayList<Integer> cntList = new ArrayList<Integer>(sparsityCount.keySet());
		Collections.sort(cntList);
		for (int k = 0; k < cntList.size(); k++) {
			System.out.println("Sparsity " + cntList.get(k) + ": " + sparsityCount.get(cntList.get(k)) + " examples");
		}
		System.out.println("One-count: " + totalOne + "/" + total);
	}
	
	public static void countSparsityHwInstance(ArrayList<HwInstance> exs) {
		
		HashMap<Integer, Integer> sparsityCount = new HashMap<Integer, Integer>();
		
		int totalOne = 0;
		int total = 0;
		for (int i = 0; i < exs.size(); i++) {
			HwInstance ex = exs.get(i);
			HwOutput gold = ex.getGoldOutput();
			int oneCnt = 0;
			for (int j = 0; j < gold.size(); j++) {
				int lb = gold.getOutput(j);
				if (lb > 0) oneCnt++;
			}
			//////////////////////////
			Integer sprCnt = sparsityCount.get(oneCnt);
			if (sprCnt == null) {
				sparsityCount.put(oneCnt, 1);
			} else {
				int newCnt = sprCnt.intValue() + 1;
				sparsityCount.put(oneCnt, newCnt);
			}
			
			totalOne += oneCnt;
			total += gold.size();
		}
		
		ArrayList<Integer> cntList = new ArrayList<Integer>(sparsityCount.keySet());
		Collections.sort(cntList);
		for (int k = 0; k < cntList.size(); k++) {
			System.out.println("Sparsity " + cntList.get(k) + ": " + sparsityCount.get(cntList.get(k)) + " examples");
		}
		System.out.println("One-count: " + totalOne + "/" + total);

	}
	
	/////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////
	public Dataset readDefaultDataset(String name) {
		int mthd = -1;

		if (readerMethod.containsKey(name)) {
			mthd = readerMethod.get(name);
		}

		if (mthd < 0) {
			throw new RuntimeException("Error: Unknown dataset name: " + name);
		}

		Dataset ds = null;
		if (mthd == ARFF_READER) {
			ds = readArffFile(name);
		} else if (mthd == CSV_READER) {
			ds = readCsvFile(name);
		}

		return ds;
	}
	
	public static Dataset readArffFile(String name) {
		
		System.out.println("==== " + name + " ====");
		DatasetReader dataSetReader = new DatasetReader();

		String xmlFile = ML_XML_FOLDER+"/"+name+"/"+name+".xml";
		String testArffFile = ML_XML_FOLDER+"/"+name+"/"+name+"-test.arff";
		String trainArffFile = ML_XML_FOLDER+"/"+name+"/"+name+"-train.arff";
		Dataset ds = dataSetReader.loadDataSetArff(name, trainArffFile, testArffFile, xmlFile);
		ds.name = name;
		return ds;
	}

	public static Dataset readCsvFile(String name) {

		DatasetReader dataSetReader = new DatasetReader();

		System.out.println("==== " + name + " ====");
		String TRAIN_FEATURE_FILE_LOCATION = ML_CSV_FOLDER+"/"+name+"/"+name+"-train-feature.csv";
		String TRAIN_LABEL_FILE_LOCATION = ML_CSV_FOLDER+"/"+name+"/"+name+"-train-label.csv";
		String TEST_FEATURE_FILE_LOCATION = ML_CSV_FOLDER+"/"+name+"/"+name+"-test-feature.csv";
		String TEST_LABEL_FILE_LOCATION = ML_CSV_FOLDER+"/"+name+"/"+name+"-test-label.csv";

		Dataset ds = dataSetReader.loadDataSetCSV(TRAIN_FEATURE_FILE_LOCATION, TRAIN_LABEL_FILE_LOCATION,
				TEST_FEATURE_FILE_LOCATION, TEST_LABEL_FILE_LOCATION);
		ds.name = name;
		return ds;
	}
	
	
	
	//this code will produce the training and test files.
	//please check that you have the right files in DataSets Location
	public static void main(String[] args) {

	}

	
	public static List<List<HwInstance>> convertToHwInstances(Dataset ds) {
		
		List<List<HwInstance>> insts = new ArrayList<List<HwInstance>>();
		
		// Train
		/////////////////////////////////
		List<Example> trns = ds.getTrainExamples();
		List<HwInstance> trainInsts = new ArrayList<HwInstance>();
		for (Example trnEx : trns) {
			trainInsts.add(exampleToInstance(trnEx, false));
			trnEx.clearAll();
		}
		insts.add(trainInsts);
		////////////////////////////////
		trns.clear();
		
		
		// Test
		/////////////////////////////////
		List<Example> tsts = ds.getTestExamples();
		List<HwInstance> testInsts = new ArrayList<HwInstance>();
		for (Example tstEx : tsts) {
			testInsts.add(exampleToInstance(tstEx, false));
			tstEx.clearAll();
		}
		insts.add(testInsts);
		////////////////////////////////
		tsts.clear();
		
		System.gc(); // re allocate memory
		
		return insts;
	}
	
	public static HwInstance exampleToInstance(Example ex, boolean unaryFeatAsSparse) {
		
		List<Label> lbs = ex.getLabel();
		Double[] feats = ex.getFeat().toArray(new Double[0]);
		double[] featCopy = copyFeat(feats);
		
		List<HwSegment> segs = new ArrayList<HwSegment>();
		for (int i = 0; i < lbs.size(); i++) {
			Label lb = lbs.get(i);
			int lbTruth = lb.getValue();
			
			HwSegment seg = new HwSegment(i, featCopy, String.valueOf(lbTruth), (!unaryFeatAsSparse));
			segs.add(seg);
		}
		
		HwInstance ins = new HwInstance(segs, Label.MULTI_LABEL_DOMAIN);
		
		return ins;
	}
	
	public static double[] copyFeat(Double[] dfeats) {
		double[] fcp = new double[dfeats.length];
		for (int i = 0; i < fcp.length; i++) fcp[i] = dfeats[i];
		return fcp;
	}

	//////////////////////
	
	public static HwInstanceDataset loadHwInstances(String name) {
		int mthd = -1;
		if (readerMethod.containsKey(name)) {
			mthd = readerMethod.get(name);
		}
		
		boolean unaryAsSparse = false;
		if (readerIsDense.containsKey(name)) {
			unaryAsSparse = readerIsDense.get(name);
		}
		

		if (mthd < 0) {
			throw new RuntimeException("Error: Unknown dataset name: " + name);
		}
		if (!readerIsDense.containsKey(name)) {
			throw new RuntimeException("Error: Unknown dataset sparse or dense: " + name);
		}

		HwInstanceDataset dset = null;
		HwInstanceInfo trainInfo = null;
		HwInstanceInfo testInfo = null;
		
		DatasetReader reader = new DatasetReader();
		if (mthd == ARFF_READER) {
			
			System.out.println("==== " + name + " ====");
			String xmlFile = ML_XML_FOLDER+"/"+name+"/"+name+".xml";
			String testArffFile = ML_XML_FOLDER+"/"+name+"/"+name+"-test.arff";
			String trainArffFile = ML_XML_FOLDER+"/"+name+"/"+name+"-train.arff";
			
			trainInfo = reader.readArffFileAsHwInstances(trainArffFile, xmlFile, unaryAsSparse);
			System.out.println("------");
			testInfo = reader.readArffFileAsHwInstances(testArffFile, xmlFile, unaryAsSparse);

		} else if (mthd == CSV_READER) {
			
			System.out.println("==== " + name + " ====");
			String TRAIN_FEATURE_FILE_LOCATION = ML_CSV_FOLDER+"/"+name+"/"+name+"-train-feature.csv";
			String TRAIN_LABEL_FILE_LOCATION = ML_CSV_FOLDER+"/"+name+"/"+name+"-train-label.csv";
			String TEST_FEATURE_FILE_LOCATION = ML_CSV_FOLDER+"/"+name+"/"+name+"-test-feature.csv";
			String TEST_LABEL_FILE_LOCATION = ML_CSV_FOLDER+"/"+name+"/"+name+"-test-label.csv";

			trainInfo = reader.readCSVseperateAsHwInstances(TRAIN_FEATURE_FILE_LOCATION, TRAIN_LABEL_FILE_LOCATION, unaryAsSparse);
			System.out.println("------");
			testInfo = reader.readCSVseperateAsHwInstances(TEST_FEATURE_FILE_LOCATION, TEST_LABEL_FILE_LOCATION, unaryAsSparse);

		}
		
		// havea check!
		if (trainInfo.featDem !=  testInfo.featDem ||
			trainInfo.labelDem != testInfo.labelDem) {
			System.out.println("Dimension inconsistent in train and test!");
		}
		System.out.println("Load train " + trainInfo.nExample + ", test " + testInfo.nExample);
		System.out.println("Feature dimension " + trainInfo.featDem + " label dimension " + trainInfo.labelDem);

		dset = new HwInstanceDataset(name, trainInfo.exmples, testInfo.exmples, trainInfo.featDem, trainInfo.labelDem);
		dset.name = name;

		return dset;
	}
	
	public HwInstanceInfo readArffFileAsHwInstances(String arffPath, String xmlPath, boolean unaryFeatAsSparse) {

		ArrayList<String> labelAttrNames = loadParseXml(xmlPath);
		HwInstanceInfo info = null;

		BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(arffPath));
			ArffReader arff = new ArffReader(reader);
			Instances data = arff.getData();

			ArrayList<HwInstance> exs = new ArrayList<HwInstance>();
			int exSize = data.numInstances();
			int totalAttrCnt = data.numAttributes();
			int labelCnt = labelAttrNames.size();
			for (int i = 0; i < exSize; i++) {
				Example ex = getExampleFromWekaInstance(i, data.instance(i), labelAttrNames);
				//System.out.println(ex.toString());
				HwInstance ins = exampleToInstance(ex, unaryFeatAsSparse);
				exs.add(ins);
			}

			//////////////////////////////

			info = new HwInstanceInfo();
			info.nExample = exs.size();
			info.featDem = (totalAttrCnt - labelCnt);
			info.labelDem = labelCnt;
			info.exmples = exs;

			//countSparsity(exs); // count
			countSparsityHwInstance(exs);

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		return info;
	}
	
	public HwInstanceInfo readCSVseperateAsHwInstances(String featureLocation, String labelLocation, boolean unaryFeatAsSparse) {
		
		ArrayList<HwInstance> instances = new ArrayList<HwInstance>(); 
		BufferedReader bufferedReaderFeature = null;
		BufferedReader bufferedReaderLabel = null;
		//BufferedWriter bufferedWriter = null;
		String lineFeature = "";
		String lineLabel = "";
		//String csvSplitBy = ",";

		
		int noOfLabels = -1;
		int noOfFeatures = -1;
		int exmpCnt = 0;
		
		try {

			bufferedReaderFeature = new BufferedReader(new FileReader(featureLocation));
			bufferedReaderLabel = new BufferedReader(new FileReader(labelLocation));
			bufferedReaderFeature.readLine();//as the first line contains column names
			bufferedReaderLabel.readLine();//as the first line contains column names

			while ((lineFeature = bufferedReaderFeature.readLine()) != null && 
				   (lineLabel = bufferedReaderLabel.readLine()) != null) {
	
				
				Example exmp = new Example();
				exmp.loadFromCsvString(exmpCnt, lineFeature, lineLabel); // parse a example line string
				
				HwInstance ins = exampleToInstance(exmp, unaryFeatAsSparse);
				instances.add(ins);
				
				noOfLabels = exmp.labelDim();
				noOfFeatures = exmp.featDim();
				exmpCnt++;
			}
			
			System.out.println("Loaded examplea: " + exmpCnt);
			
			bufferedReaderFeature.close();
			bufferedReaderLabel.close();

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (bufferedReaderFeature != null) {
				try {
					bufferedReaderFeature.close();
					bufferedReaderLabel.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}


		HwInstanceInfo info = new HwInstanceInfo();
		info.nExample = exmpCnt;
		info.featDem = noOfFeatures;
		info.labelDem = noOfLabels;
		info.exmples = instances;
		
		//countSparsity(instances); // count 
		countSparsityHwInstance(instances);
		
		return info;
	}
	
}
