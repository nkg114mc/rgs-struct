package multilabel.data;

import java.util.ArrayList;
import java.util.List;

import multilabel.instance.Example;
import sequence.hw.HwInstance;

public class HwInstanceDataset {

	public String name;
			
	List<HwInstance> trainExamps;
	List<HwInstance> testExamps;
	
	int featDimension;
	int labelDimension;
	
	private HwInstanceDataset() {
		name = "unknown";
		trainExamps = new ArrayList<HwInstance>();
		testExamps = new ArrayList<HwInstance>();
		featDimension = -1;
		labelDimension = -1;	
	}
	
	
	
	public HwInstanceDataset(String nm, List<HwInstance> trExmps, List<HwInstance> tsExmps, int nFeatures, int nLabels) {
		name = nm;
		
		trainExamps = trExmps;
		testExamps = tsExmps;
		
		featDimension = nFeatures;
		labelDimension = nLabels;		
	}
	
	public List<List<HwInstance>> getInstListList() {
		List<List<HwInstance>> insts = new ArrayList<List<HwInstance>>();
		insts.add(trainExamps);
		insts.add(testExamps);
		return insts;
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
	
	public List<HwInstance> getTrainExamples() {
		return trainExamps;
	}
	public List<HwInstance> getTestExamples() {
		return testExamps;
	}
	public void setTrainExamples(List<HwInstance> trExmps) {
		trainExamps = trExmps;
	}
	public void setTestExamples(List<HwInstance> tsExmps) {
		testExamps = tsExmps;
	}
	
	
	public void addTrainExample(HwInstance e) {
		trainExamps.add(e);
	}
	public void addTestExample(HwInstance e) {
		testExamps.add(e);
	}

}
