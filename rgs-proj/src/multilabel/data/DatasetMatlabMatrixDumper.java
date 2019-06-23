package multilabel.data;



import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;

import multilabel.instance.Example;
import multilabel.instance.Label;

public class DatasetMatlabMatrixDumper {

	public static void main(String[] args) {
		// read dataset
		String name = "bookmarks-umass";
		
		DatasetReader reader = new DatasetReader();
		Dataset ds = reader.readDefaultDataset(name);
		
		// Dataset ds = PrunerLearning.readArffFileTest(name);
		
		// dumping
		//dumpMatrix("/scratch/learner/mlc_lsdr-master", ds);
		//dumpMatrix2("/home/mc/workplace/deepsp/SPEN-master/icml_mlc_data/data/yeast", ds);
	}
	
	public static void dumpMatrix2(String folder, Dataset ds) {
		PrintWriter trainFeatWriter;
		PrintWriter trainLabelWriter;
		PrintWriter testFeatWriter;
		PrintWriter testLabelWriter;
		
		try {
			// prepare folder
			String folderPath = folder + "/" + ds.name + "_mat";
		    File dir = new File(folderPath);
		     
		    // attempt to create the directory here
		    if (!dir.exists()) {
		    	dir.mkdir();
		    }

			// start dumpping
			trainFeatWriter = new PrintWriter(folderPath + "/" + ds.name + "_train_feat.txt");
			trainLabelWriter = new PrintWriter(folderPath + "/" + ds.name + "_train_label.txt");
			testFeatWriter = new PrintWriter(folderPath + "/" + ds.name + "_test_feat.txt");
			testLabelWriter = new PrintWriter(folderPath + "/" + ds.name + "_test_label.txt");

			// train file
			ArrayList<Example> trainExs = ds.getTrainExamples();
			for (int i = 0; i < trainExs.size(); i++) {
				dumpExample(trainExs.get(i), trainFeatWriter, trainLabelWriter);
			}

			// test file
			ArrayList<Example> testExs = ds.getTestExamples();
			for (int i = 0; i < testExs.size(); i++) {
				dumpExample(testExs.get(i), testFeatWriter, testLabelWriter);
			}

			trainFeatWriter.close();
			trainLabelWriter.close();
			testFeatWriter.close();
			testLabelWriter.close();
			System.out.println("Finish dumping!");

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	public static void dumpMatrix(String folder, Dataset ds) {
		PrintWriter trainFeatWriter;
		PrintWriter trainLabelWriter;
		PrintWriter testFeatWriter;
		PrintWriter testLabelWriter;
		
		try {
			// prepare folder
			String folderPath = folder + "/" + ds.name;
		    File dir = new File(folderPath);
		     
		    // attempt to create the directory here
		    if (!dir.exists()) {
		    	dir.mkdir();
		    }


			// start dumpping
			trainFeatWriter = new PrintWriter(folderPath + "/X_tr");
			trainLabelWriter = new PrintWriter(folderPath + "/Y_tr");
			testFeatWriter = new PrintWriter(folderPath + "/X_tt");
			testLabelWriter = new PrintWriter(folderPath + "/Y_tt");

			// train file
			ArrayList<Example> trainExs = ds.getTrainExamples();
			for (int i = 0; i < trainExs.size(); i++) {
				dumpExample(trainExs.get(i), trainFeatWriter, trainLabelWriter);
			}

			// test file
			ArrayList<Example> testExs = ds.getTestExamples();
			for (int i = 0; i < testExs.size(); i++) {
				dumpExample(testExs.get(i), testFeatWriter, testLabelWriter);
			}


			trainFeatWriter.close();
			trainLabelWriter.close();
			testFeatWriter.close();
			testLabelWriter.close();
			System.out.println("Finish matlab matrix dumping!");

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static void dumpExample(Example ex, PrintWriter featWriter, PrintWriter labelWriter) {
		
		String featStr = "";
		String labelStr = "";
		
		// feat
		ArrayList<Double> originFeat = ex.getFeat();
		for (int i = 0; i < originFeat.size(); i++) {
			if (!featStr.equals("")) {
				featStr += " ";
			}
			featStr += String.valueOf(originFeat.get(i).doubleValue());
		}
		
		// label
		ArrayList<Label> labels = ex.getLabel(); // the labels you need to predict
		for (int j = 0; j < labels.size(); j++) {
			if (!labelStr.equals("")) {
				labelStr += " ";
			}
			labelStr += String.valueOf(labels.get(j).value);
		}
		
		// output!
		featWriter.println(featStr);
		labelWriter.println(labelStr);
	}
}
