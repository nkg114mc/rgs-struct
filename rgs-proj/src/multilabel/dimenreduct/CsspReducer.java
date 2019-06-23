package multilabel.dimenreduct;

import java.util.ArrayList;

import multilabel.data.Dataset;
import multilabel.data.DatasetReader;
import multilabel.evaluation.MultiLabelEvaluator;
import multilabel.instance.Example;
import multilabel.instance.Label;
import multilabel.learning.StructOutput;
import multilabel.learning.search.OldSearchState;

public class CsspReducer extends LabelDimensionReducer {

	//int fullDimen = -1;
	//int lowDimen = -1;
	CsspModel csspModel;
	
	public CsspReducer(String modelPath1, String modelPath2) {
		csspModel = new CsspModel(modelPath1, modelPath2);
		//fullDimen = csspModel.getFullDim();
		//lowDimen = csspModel.getLowDim();
	}
	
	public int getFullDimen() {
		return csspModel.getFullDim();
	}
	
	public int getLowDimen() {
		return csspModel.getLowDim();
	}
	
	
	/////////////////////////////////////////////////////////////////////////
	
	// get a label array with ground truth in low dimension
	public ArrayList<Label> encodeExample(Example ex) {
		StructOutput gold = ex.getGroundTruthOutput();
		ArrayList<Label> lowLabels = new ArrayList<Label>();
		int[] picked = csspModel.getPickIdx();
		for (int j = 0; j < picked.length; j++) {
			int rdv = gold.getValue(picked[j] - 1);
			Label label = new Label(j, (int)rdv);
			lowLabels.add(label);
		}
		return lowLabels;
	}
	
	public StructOutput encodeGroundTruth(Example ex) {
		StructOutput gold = ex.getGroundTruthOutput();
		StructOutput rdOut = new StructOutput(csspModel.getLowDim());
		int[] picked = csspModel.getPickIdx();
		for (int j = 0; j < rdOut.size(); j++) {
			int rdv = gold.getValue(picked[j] - 1);
			rdOut.setValue(j, rdv);
		}
		return rdOut;
	}

	
	public StructOutput decodeOutput(Example ex, StructOutput output) {
	//public StructOutput decodeOutput(StructOutput output) {
		if (output.size() != csspModel.getLowDim()) {
			throw new RuntimeException("Error on DR decoding: " +output.size()+" != "+ csspModel.getLowDim());
		}
		
		int ldim = csspModel.getLowDim();
		int fdim = csspModel.getFullDim();
		double[][] Vm = csspModel.getVm();
		
		// decoding
		StructOutput fullDimOutput = new StructOutput(fdim);
		for (int i = 0; i < fdim; i++) { // full dim
			double realSum = 0;
			for (int j = 0; j < ldim; j++) {
				double lowdVal = ((double)(output.getValue(j))) * 2 - 1.0;
				realSum += (Vm[i][j] * lowdVal);
			}
			int intVal = roundPredict01(realSum);
			fullDimOutput.setValue(i, intVal);
		}
		return fullDimOutput;
	}
	
	public int roundPredict01(double realVal) {
		int sign = (int) Math.signum(realVal);
		if (sign > 0) {
			return 1;
		} else {
			return 0;
		}
	}
	public int roundPredictPosNeg1(double realVal) {
		int sign = (int) Math.signum(realVal);
		if (sign > 0) {
			return 1;
		} else {
			return -1;
		}
	}
	
	
	
	
	public static void main(String[] args) {
		
		test2();
		
	}
	
	public static void test1() {
		
		DatasetReader dataSetReader = new DatasetReader();
		
		String name = "medical";
		String xmlFile = "ML_datasets/"+name+"/"+name+".xml";
		String testArffFile = "ML_datasets/"+name+"/"+name+"-test.arff";
		String trainArffFile = "ML_datasets/"+name+"/"+name+"-train.arff";
		Dataset ds = dataSetReader.loadDataSetArff(name, trainArffFile, testArffFile, xmlFile);
		ds.name = name;
		
		///////////////////////
		
		CsspReducer csspRd = new CsspReducer("/scratch/learner/mlc_lsdr-master/medical/V24", "/scratch/learner/mlc_lsdr-master/medical/pickIdx24");
		
		///////////////////////
		
		ArrayList<Example> testExs = ds.getTestExamples();
		for (int i = 0; i < testExs.size(); i++) {
			Example ex = testExs.get(i);
			// rd!
			StructOutput rdOut = csspRd.encodeGroundTruth(ex);
			// reconstruct
			ex.predictOutput = csspRd.decodeOutput(ex, rdOut);
		}
		
		// scoring!
		MultiLabelEvaluator evaluator = new MultiLabelEvaluator();
		evaluator.evaluationDataSet(ds.name, ds.getTestExamples());
	}
	
	public static void test2() {
		
		String name = "medical";
		DatasetReader dataSetReader = new DatasetReader();
		Dataset ds = dataSetReader.readDefaultDataset(name);
		
		///////////////////////
		
		CsspReducer csspRd = new CsspReducer("/scratch/learner/mlc_lsdr-master/medical/V24", "/scratch/learner/mlc_lsdr-master/medical/pickIdx24");
		
		///////////////////////
		
		
		ArrayList<Example> testExs = ds.getTestExamples();
		
		
		// RD!
		for (int i = 0; i < testExs.size(); i++) {
			Example ex = testExs.get(i);
			ex.encodeToLowDimension(csspRd);
			ex.predictOutput = ex.getGroundTruthOutput();
		}
		
		System.out.println("Low dimension = " + testExs.get(0).labelDim());
		DatasetReader.countSparsity(testExs);
		
		// scoring!
		MultiLabelEvaluator evaluator = new MultiLabelEvaluator();
		evaluator.evaluationDataSet(ds.name, ds.getTestExamples());
		
		// =====
		
		// reconstruct
		for (int i = 0; i < testExs.size(); i++) {
			Example ex = testExs.get(i);
			StructOutput lowGold = ex.getGroundTruthOutput();
			ex.decodeToFullDimension(csspRd);
			//ex.predictOutput = ex.getGroundTruthOutput();
			ex.predictOutput = csspRd.decodeOutput(ex, lowGold);
		}
		
		System.out.println("Full dimension = " + testExs.get(0).labelDim());
		
		// scoring!
		evaluator.evaluationDataSet(ds.name, ds.getTestExamples());
	}
	
}
