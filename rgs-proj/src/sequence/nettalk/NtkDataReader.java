package sequence.nettalk;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import sequence.hw.HwDataReader;
import sequence.hw.HwInstance;
import sequence.hw.HwSegment;

public class NtkDataReader {
	
	public static void main(String[] args) {
		NtkPhonemeLabelSet phLabels = new NtkPhonemeLabelSet();
		NtkDataReader rder = new NtkDataReader();
		rder.readData("../datasets/nettalk_phoneme_train.txt", "../datasets/nettalk_phoneme_test.txt", phLabels.getLabels());
		rder.readData("../datasets/nettalk_stress_train.txt", "../datasets/nettalk_stress_test.txt", phLabels.getLabels());
		
	}
	
	public List<List<HwInstance>> readData(String trainFn, String testFn, String[] lbs) {
		

		checkFileExistance(trainFn);
		checkFileExistance(testFn);
		
		System.out.println("Reading training file: " + trainFn);
		System.out.println("Reading testing file: " + testFn);
		
		List<HwInstance> trainExs = HwDataReader.readFromFile(trainFn,lbs);
		List<HwInstance> testExs = HwDataReader.readFromFile(testFn,lbs);
		
		List<List<HwInstance>> twoExSet = new ArrayList<List<HwInstance>>();
		twoExSet.add(trainExs);
		twoExSet.add(testExs);
		
		//collectAllLabels(twoExSet);
		
		return twoExSet;
	}
	
	public void checkFileExistance(String dfolder) {
		// file
		File dfile = new File(dfolder);
		if (!dfile.exists()) {
			throw new RuntimeException("Can not read " + dfile.getAbsolutePath());
		}
	}
	
	public HashSet<String> collectAllLabels(List<List<HwInstance>> twoExSet) {
		
		HashSet<String> lset = new HashSet<String>();
		for (List<HwInstance> lst : twoExSet) {
			for (HwInstance inst: lst) {
				List<HwSegment> segs = inst.letterSegs;
				for (HwSegment seg : segs) {
					lset.add(seg.letter);
				}
			}
		}
		
		System.out.println("Size = " + lset.size());
		for (String l : lset) {
			System.out.print("\""+l+"\",");
		}
		
		return lset;
	}

}
