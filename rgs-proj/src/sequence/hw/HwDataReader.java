package sequence.hw;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import edu.illinois.cs.cogcomp.sl.core.SLProblem;
import general.AbstractInstance;
import general.AbstractOutput;

public class HwDataReader {
	
	//public static void main(String[] args) {
	//	HwDataReader rder = new HwDataReader();
	//	rder.readData("../datasets/hw", 0, true);
	//}
	
	public List<List<HwInstance>> readData(String dfolder, HwLabelSet hwLabels, int cvIndex, boolean isSmall) {
		
		// folder
		File dfile = new File(dfolder);
		if (!(dfile.exists() && dfile.isDirectory())) {
			throw new RuntimeException("Can not read " + dfile.getAbsolutePath());
		}
		
		// cv index
		if (cvIndex < 0 && cvIndex >= 10) {
			throw new RuntimeException("Cv foder index (0 - 9) does not exist: " + cvIndex);
		}
		
		///home/mc/Desktop/rand_search/datasets/hw/ocr_fold0_sm_test.txt
		///home/mc/Desktop/rand_search/datasets/hw/ocr_fold0_sm_train.txt
		////home/mc/Desktop/rand_search/datasets/hw/ocr_fold0_test.txt
		///home/mc/Desktop/rand_search/datasets/hw/ocr_fold0_train.txt
		String smallstr = "";
		if (isSmall) smallstr = "sm_";
		String testFn = dfolder + "/" + ("ocr_fold" + String.valueOf(cvIndex) + "_" + smallstr + "test.txt"); 
		String trainFn =  dfolder + "/" + ("ocr_fold" + String.valueOf(cvIndex) + "_" + smallstr + "train.txt"); 
		
		System.out.println("Reading training file: " + trainFn);
		System.out.println("Reading testing file: " + testFn);
		
		List<HwInstance> trainExs = readFromFile(trainFn,hwLabels.getLabels());
		List<HwInstance> testExs = readFromFile(testFn,hwLabels.getLabels());
		
		List<List<HwInstance>> twoExSet = new ArrayList<List<HwInstance>>();
		twoExSet.add(trainExs);
		twoExSet.add(testExs);
		return twoExSet;
	}
	
	public static List<SLProblem> convertToSLProblem(List<List<HwInstance>> trainTestSets) {
		List<SLProblem> prbs = new ArrayList<SLProblem>();
		prbs.add(ExampleListToSLProblem(trainTestSets.get(0))); // train
		prbs.add(ExampleListToSLProblem(trainTestSets.get(1))); // test
		return prbs;
	}
	
	public List<SLProblem> readDataAsSLProblem(String dfolder, HwLabelSet hwLabels, int cvIndex, boolean isSmall) {
		List<List<HwInstance>> trainTestSets = readData(dfolder, hwLabels, cvIndex, isSmall);
		List<SLProblem> prbs = convertToSLProblem(trainTestSets);
		return prbs;
	}
	
/*
	public static SLProblem ExampleListToSLProblem(List<HwInstance> insts) {
		SLProblem problem = new SLProblem();
		for (int i = 0; i < insts.size(); i++) {
			HwOutput goutput = HwOutput.getGoldOutput(insts.get(i));
			problem.addExample(insts.get(i), goutput);
		}
		return problem;
	}
*/
	public static SLProblem ExampleListToSLProblem(List<HwInstance> insts) {
		SLProblem problem = new SLProblem();
		for (int i = 0; i < insts.size(); i++) {
			HwOutput goutput = insts.get(i).getGoldOutput();
			problem.addExample(insts.get(i), goutput);
		}
		return problem;
	}
	
	public static List<HwInstance> readFromFile(String fn, String[] alphabet) {

		List<HwInstance> results = new ArrayList<HwInstance>();
		
		try {
			BufferedReader br = new BufferedReader(new FileReader(fn));
			String line;
			ArrayList<HwSegment> ltseq = new ArrayList<HwSegment>();
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (line.equals("")) {
					// end of sequence
					HwInstance inst = new HwInstance(ltseq,alphabet);
					results.add(inst);
					// prepare for the next
					ltseq = new ArrayList<HwSegment>();
				} else {
					HwSegment hwltr = new HwSegment(line);
					ltseq.add(hwltr);
				}
			}
			
			if (ltseq.size() > 0) {
				HwInstance inst = new HwInstance(ltseq,alphabet);
				results.add(inst);
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		int maxLen = -1;
		for (HwInstance inst : results) {
			if (inst.size() > maxLen) {
				maxLen = inst.size();
			}
		}
		System.out.println("Max length " + maxLen);
		
		
		System.out.println("Loaded instances " + results.size());
		
		return results;
	}
	

}
