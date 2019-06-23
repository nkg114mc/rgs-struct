package multilabel.data;

import multilabel.instance.Example;

import java.io.PrintWriter;
import java.util.ArrayList;

public class Dataset {

	public String name;
	//ArrayList<Example> allExamples;
			
	ArrayList<Example> trainExamps;
	ArrayList<Example> testExamps;
	
	int featDimension;
	int labelDimension;
	
	private Dataset() {
		name = "unknown";
		trainExamps = new ArrayList<Example>();
		testExamps = new ArrayList<Example>();
		featDimension = -1;
		labelDimension = -1;	
	}
	
	
	
	public Dataset(String nm, ArrayList<Example> trExmps, ArrayList<Example> tsExmps, int nFeatures, int nLabels) {
		name = nm;
		
		trainExamps = trExmps;
		testExamps = tsExmps;
		
		featDimension = nFeatures;
		labelDimension = nLabels;		
	}
	
	// assume load data from csv files
	public void loadFromFile() {
		
	}
	
	public int getLabelDimension() {
		return labelDimension;
	}
	public int getFeatureDimension() {
		return featDimension;
	}
	public int getExampleNumber() {
		return labelDimension;
	}
	public int getTestNumber() {
		return labelDimension;
	}
	public int getTrainNumber() {
		return labelDimension;
	}
	
	public ArrayList<Example> getTrainExamples() {
		return trainExamps;
	}
	public ArrayList<Example> getTestExamples() {
		return testExamps;
	}
	public void setTrainExamples(ArrayList<Example> trExmps) {
		trainExamps = trExmps;
	}
	public void setTestExamples(ArrayList<Example> tsExmps) {
		testExamps = tsExmps;
	}
	
	
	public void addTrainExample(Example e) {
		trainExamps.add(e);
	}
	public void addTestExample(Example e) {
		testExamps.add(e);
	}
	
	/////////////////////////////////////
	// dump to matlab readable matrix
	public void dumpMatlabMatrix(String folder) {
	
		
	}
}
