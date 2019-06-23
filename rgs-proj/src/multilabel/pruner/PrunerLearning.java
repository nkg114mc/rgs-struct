package multilabel.pruner;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;

import multilabel.data.Dataset;
import multilabel.data.DatasetReader;
import multilabel.utils.UtilFunctions;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.SparseInstance;
import multilabel.instance.Example;
import multilabel.instance.Label;
import multilabel.instance.OldWeightVector;

public class PrunerLearning {


	private static class PrunerArgs {
		public int topk;
		public String name;
		public String metric;
		public PrunerArgs() {
			topk = 0;
			name = "";
			metric = "";
		}
	}

	public static PrunerArgs parseArgs(String[] args) {
		PrunerArgs pargs = new PrunerArgs();
		for (int i = 0; i < args.length; i++) {
			if (args[i].equals("-topk")) {
				pargs.topk = Integer.parseInt(args[i + 1]);
			} else if (args[i].equals("-name")) {
				pargs.name = (args[i + 1]).trim();
			} else if (args[i].equals("-metric")) {
				pargs.metric = (args[i + 1]).trim();
			}
		}
		return pargs;
	}

	public static void main(String[] args) {
		/*
		DatasetReader dataSetReader = new DatasetReader();

		//Dataset ds = dataSetReader.loadDataSetCSV(TRAIN_FEATURE_FILE_LOCATION, TRAIN_LABEL_FILE_LOCATION,
		//		                                  TEST_FEATURE_FILE_LOCATION, TEST_LABEL_FILE_LOCATION);

		String name = "yeast";//"scene";//"emotions";
		String xmlFile = "ML_datasets/"+name+"/"+name+".xml";
		String testArffFile = "ML_datasets/"+name+"/"+name+"-test.arff";
		String trainArffFile = "ML_datasets/"+name+"/"+name+"-train.arff";
		Dataset ds = dataSetReader.loadDataSetArff(trainArffFile, testArffFile, xmlFile);
		ds.name = name;

		PrunerLearning.dumpRanklibFeatureFile(ds.getTrainExamples(), "pruner_rank/" + name + "_prune_train_feat.txt");
		PrunerLearning.dumpRanklibFeatureFile(ds.getTestExamples(), "pruner_rank/" + name + "_prune_test_feat.txt");
		//PrunerLearning.dumpWekaFeatureFile(ds.getTrainExamples(), "sence_train_feat.arff");
		//PrunerLearning.dumpWekaFeatureFile(ds.getTestExamples(), "sence_test_feat.arff");


		//String modelFn = "pruner_rank/emotions_rank_lambdamart.txt";
		//LabelPruner pruner = new LambdaMartLabelPruner(modelFn, 4);
		//PruningEvaluator.evaluatePruner(ds.getTestExamples(), pruner);
		 */

		/*
		String[] dsNames = { "scene", "emotions", "yeast", "enron", "medical", "LLOG", "SLASHDOT", "tmc2007-500", "genbasebin"};
		for (int i = 0; i < dsNames.length; i++) {
			readArffFileTest(dsNames[i]);
		}*/
		/*
		String[] dsNames = { "CAL500", "bibtex",  "Corel5k", "delicious", "mediamill", "bookmarks"};
		for (int i = 0; i < dsNames.length; i++) {
			readCsvFileTest(dsNames[i]);
		}*/


		PrunerArgs pargs = parseArgs(args);
		System.out.println("Name: " + pargs.name);
		System.out.println("TopK: " + pargs.topk);
		System.out.println("Metric: " + pargs.metric);

		// read dataset
		DatasetReader dataSetReader = new DatasetReader();
		//Dataset ds = readArffFileTest(pargs.name);
		Dataset ds = dataSetReader.readDefaultDataset(pargs.name);

		// train!
		//LambdaMartLabelPruner.trainPrunerRanklib(ds, pargs.topk, pargs.metric); // comment out at 2016-3-6
		//dumpArffFileTest(ds);
		dumpSvmPerfFileTest(ds);
		
		//RandomForestLabelPruner.trainPrunerRanklib(ds, pargs.topk);
		//LambdaMartLabelPruner.dumpingDatasetFeatureRanklib(ds);
		//CrobirankPruner.train(ds);
		//SofiaMlPruner.train(ds);
		
		
		


		// done.
		System.out.println("Done Pruner Traing.");
	}
	
	public static void dumpSvmPerfFileTest(Dataset ds) {
		String trainFile = "pruner_rank/"+ds.name+"-pruner-train.dat";
		dumpSvmperfFeatureFile(ds.getTrainExamples(), trainFile);
		String testFile = "pruner_rank/"+ds.name+"-pruner-test.dat";
		dumpSvmperfFeatureFile(ds.getTestExamples(), testFile);
	}
	
	public static void dumpArffFileTest(Dataset ds) {
		String testArffFile = "pruner_rank/"+ds.name+"-pruner-train.arff";
		dumpWekaFeatureFile(ds.getTrainExamples(), testArffFile);
		String trainArffFile = "pruner_rank/"+ds.name+"-pruner-test.arff";
		dumpWekaFeatureFile(ds.getTestExamples(), trainArffFile);
	}

	public static Dataset readArffFileTest(String name) {

		System.out.println("==== " + name + " ====");
		DatasetReader dataSetReader = new DatasetReader();

		String xmlFile = "ML_datasets/"+name+"/"+name+".xml";
		String testArffFile = "ML_datasets/"+name+"/"+name+"-test.arff";
		String trainArffFile = "ML_datasets/"+name+"/"+name+"-train.arff";
		Dataset ds = dataSetReader.loadDataSetArff(name, trainArffFile, testArffFile, xmlFile);
		ds.name = name;
		return ds;
	}

	public static Dataset readCsvFileTest(String name) {

		DatasetReader dataSetReader = new DatasetReader();

		System.out.println("==== " + name + " ====");
		String TRAIN_FEATURE_FILE_LOCATION = "DataSets/"+name+"/"+name+"-train-feature.csv";
		String TRAIN_LABEL_FILE_LOCATION = "DataSets/"+name+"/"+name+"-train-label.csv";
		String TEST_FEATURE_FILE_LOCATION = "DataSets/"+name+"/"+name+"-test-feature.csv";
		String TEST_LABEL_FILE_LOCATION = "DataSets/"+name+"/"+name+"-test-label.csv";

		Dataset ds = dataSetReader.loadDataSetCSV(TRAIN_FEATURE_FILE_LOCATION, TRAIN_LABEL_FILE_LOCATION,
				TEST_FEATURE_FILE_LOCATION, TEST_LABEL_FILE_LOCATION);
		ds.name = name;
		return ds;
	}


	// LambdaMart
	public static void prunerTraining() {



	}


	// if the ground truth label of current instance contains 0 only, 
	// then it is need to prune this instance
	public static boolean containOneLabel(ArrayList<Label> lbs) {
		int cnt1 = 0;
		for (int i = 0; i < lbs.size(); i++) {
			if (lbs.get(i).value > 0) {
				cnt1++;
			}
		}
		return (cnt1 > 0);
	}

	public static void dumpRanklibFeatureFile(ArrayList<Example> examples, String fname) {

		int all0Cnt = 0;

		try {
			PrintWriter writer = new PrintWriter(fname);

			int qid = 0;
			for (Example exmp : examples) {

				qid++;
				OldWeightVector[] allwv = PrunerFeaturizer.featurizeAll(exmp);

				ArrayList<Label> goldLabels = exmp.getLabel(); 

				boolean contain1 = containOneLabel(goldLabels);

				if (contain1) { // ok to dump!
					for (int i = 0; i < goldLabels.size(); i++) {
						int rank = goldLabels.get(i).value;
						writer.println(rank + " " + "qid:" + qid + " " + allwv[i].toSparseRanklibStr());
					}
				} else {
					all0Cnt++;
				}
			}

			writer.close();

			System.out.println("Dumping instance count " + examples.size() + ", among which " + all0Cnt + " are all-0-labeled.");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}



	}

	public static void dumpWekaFeatureFile(ArrayList<Example> examples, String fname) {

		FastVector fvClassVal = new FastVector(2);
		fvClassVal.addElement("0");
		fvClassVal.addElement("1");
		Attribute ClassAttribute = new Attribute("class", fvClassVal);

		FastVector attrs = new FastVector();
		int nl = examples.get(0).labelDim();
		int nf = examples.get(0).featDim();
		int prunedFeatDim = nl * nf;
		for (int i = 0; i < prunedFeatDim; i++) {
			String attrName = "Attr-" + String.valueOf(i);
			attrs.addElement(new Attribute(attrName));
		}
		attrs.addElement(ClassAttribute);


		////////////////////////////////////////


		ArrayList<SparseInstance> prunerInsts = new ArrayList<SparseInstance>();

		int qid = 0;
		for (Example exmp : examples) {

			qid++;
			OldWeightVector[] allwv = PrunerFeaturizer.featurizeAll(exmp);

			ArrayList<Label> goldLabels = exmp.getLabel(); 
			for (int i = 0; i < goldLabels.size(); i++) {
				int rank = goldLabels.get(i).value;
				//writer.println(rank + " " + "qid:" + qid + " " + allwv[i].toSparseRanklibStr());
				SparseInstance inst = getWekaInstance(allwv[i], rank);
				prunerInsts.add(inst);

			}

		}

		Instances wekaDataSet = new Instances("pruning_cal500", attrs, prunerInsts.size());
		wekaDataSet.setClassIndex(attrs.size() - 1);
		for (int j = 0; j < prunerInsts.size(); j++) {
			wekaDataSet.add(prunerInsts.get(j));
		}
		UtilFunctions.saveArff(wekaDataSet, fname);

	}

	public static SparseInstance getWekaInstance(OldWeightVector fv, int label) {
		int dimen = fv.getMaxLength() + 1;

		ArrayList<Integer> idxs = new ArrayList<Integer>(fv.getKeys());
		Collections.sort(idxs);

		int[] featIdx = new int[idxs.size() + 1];
		double[] featVals = new double[idxs.size() + 1];

		for (int i = 0; i < idxs.size(); i++) {
			int idx = idxs.get(i).intValue();
			featIdx[i] = idx;
			featVals[i] = fv.get(idx);
		}

		featIdx[idxs.size()] = dimen - 1;
		featVals[idxs.size()] = label;

		SparseInstance wekaInst = new SparseInstance(1.0, featVals, featIdx, dimen);
		return wekaInst;
	}


	public static void dumpSvmperfFeatureFile(ArrayList<Example> examples, String fname) {

		try {
			PrintWriter writer = new PrintWriter(fname);

			for (Example exmp : examples) {
				OldWeightVector[] allwv = PrunerFeaturizer.featurizeAll(exmp);
				ArrayList<Label> goldLabels = exmp.getLabel(); 

				for (int i = 0; i < goldLabels.size(); i++) {
					int rank = goldLabels.get(i).value;
					int label = -1;
					if (rank > 0) {
						label = +1;
					}
					writer.println(label + " " + allwv[i].toSparseRanklibStr());
				}
			}

			writer.close();

			System.out.println("Dumping instance count " + examples.size() + ".");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

	}


}
