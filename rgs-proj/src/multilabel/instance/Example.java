package multilabel.instance;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;

import multilabel.learning.StructOutput;
import multilabel.data.DatasetReader;
import multilabel.dimenreduct.LabelDimensionReducer;

public class Example {
	
	/**
	 * A example is just a pair of (x, y) given by the data set.
	 * In training, y is given, while in testing, we assume that y is unknown.
	 */

	int index;
	ArrayList<Label> labels; // the labels you need to predict 
	ArrayList<Double> originFeature; // feature vector given by the dataset ()
	
	ArrayList<Label> fullDimenLabels;
	boolean isInLowDim;
	
	HashMap<Integer, Label> idxLblMap = null;
	
	// for evaluation only
	public StructOutput predictOutput = null;
	
	public Example() {}
	
	public Example(int idx, ArrayList<Label> ls, ArrayList<Double> fs) {
		index = idx;
		labels = ls;
		originFeature = fs;
		isInLowDim = false;
	}
	
	public void clearAll() {
		labels = null;
		originFeature = null;
		fullDimenLabels = null;
		idxLblMap = null;
		predictOutput = null;
	}
	
	public HashMap<Integer, Label> getIndexLabelMap() {
		if (idxLblMap == null) {
			idxLblMap = new HashMap<Integer, Label>();
			for (Label lb : labels) {
				idxLblMap.put(lb.originIndex, lb);
			}
		}
		return idxLblMap;
	}
	
	public Label getLabelGivenIndex(int idx) {
		HashMap<Integer, Label> ilm = getIndexLabelMap();
		return ilm.get(idx);
	}
	
	public void loadFromCsvString(int idx, String featLineStr, String labelLineStr) {

		index = idx;
		
		int noOfLabels = -1;
		int noOfFeatures = -1;

		String[] splittedLineFeature = featLineStr.split(DatasetReader.CSV_SPLIT);
		noOfFeatures = splittedLineFeature.length;
		String[] splittedLineLabel = labelLineStr.split(DatasetReader.CSV_SPLIT);
		noOfLabels = splittedLineLabel.length;

		// all feats
		originFeature = new ArrayList<Double>();
		for (int i = 0; i < splittedLineFeature.length; i++) {
			double fval = Double.parseDouble(splittedLineFeature[i]);
			originFeature.add(fval);
		}

		// all labels				
		labels = new ArrayList<Label> ();
		for (int j = 0; j < splittedLineLabel.length; j++) {
			double lval = Double.parseDouble(splittedLineLabel[j]);
			Label label = new Label(j, (int)lval);
			labels.add(label);
		}

	}
	
	public StructOutput getGroundTruthOutput() {
		StructOutput goldOutput = new StructOutput(this.labelDim());
		ArrayList<Label> labels = this.getLabel();
		for (int i = 0; i < this.labelDim(); i++) {
			goldOutput.setValue(labels.get(i).originIndex, labels.get(i).value);
		}
		return goldOutput;
	}
	
	
	
	/////// About the dimension reduction
	public boolean isUnderLowDimension() {
		return isInLowDim;
	}
	public void encodeToLowDimension(LabelDimensionReducer reducer) {
		ArrayList<Label> lowDimLabels = reducer.encodeExample(this);
		// replace
		fullDimenLabels = labels;
		labels = lowDimLabels;
		isInLowDim = true;
		idxLblMap = null;
	}
	public void decodeToFullDimension(LabelDimensionReducer reducer) {
		if (isInLowDim) {
			labels = fullDimenLabels;
			fullDimenLabels = null;
			isInLowDim = false;
			idxLblMap = null;
		}
	}
	
	/////////////////////////////////////
	
	public int getIndex() {
		return index;
	}
	
	public int labelDim() {
		return (labels.size());
	}
	public int featDim() {
		return (originFeature.size());
	}

	public ArrayList<Label> getLabel() {
		return (labels);
	}
	public ArrayList<Double> getFeat() {
		return (originFeature);
	}
	
	public int getOneLabelCount() {
		int lcnt = 0;
		ArrayList<Label> labels = this.getLabel();
		for (int i = 0; i < this.labelDim(); i++) {
			if (labels.get(i).value > 0) {
				lcnt++;
			}
		}
		return lcnt;
	}
	
	public int getFeatureDimen() {
		int nlabel = this.labelDim();
		int featDim = this.featDim();
		
		int unaryDim =  nlabel * featDim;
		int binaryDim = 4 * (((nlabel - 1) * nlabel) / 2);
		
		int count = unaryDim + binaryDim;
		return count;
	}
	
	public String toString() {
		String str = "(";
		for (int i = 0; i < originFeature.size(); i++) {
			str += originFeature.get(i) + ",";
		}
		str += " | ";
		for (int i = 0; i < labels.size(); i++) {
			str += labels.get(i).value + ",";
		}
		str += ")";
		return str;
	}
}
