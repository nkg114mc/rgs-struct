package sequence.twitterpos;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import edu.illinois.cs.cogcomp.sl.core.SLProblem;
import general.AbstractInstance;
import general.AbstractOutput;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;
import sequence.hw.HwSegment;

public class TwitterDataReader {

	public List<List<TwitterPosExample>> readData(String trainFn, String testFn, TwitterPosLabelSet hwLabels) {

		System.out.println("Reading training file: " + trainFn);
		System.out.println("Reading testing file: " + testFn);
		
		List<TwitterPosExample> trainExs = readFromFile(trainFn,hwLabels.getLabels());
		List<TwitterPosExample> testExs = readFromFile(testFn,hwLabels.getLabels());
		
		List<List<TwitterPosExample>> twoExSet = new ArrayList<List<TwitterPosExample>>();
		twoExSet.add(trainExs);
		twoExSet.add(testExs);
		return twoExSet;
	}
	
	public static List<SLProblem> convertToSLProblem(List<List<TwitterPosExample>> trainTestSets) {
		List<SLProblem> prbs = new ArrayList<SLProblem>();
		prbs.add(ExampleListToSLProblem(trainTestSets.get(0))); // train
		prbs.add(ExampleListToSLProblem(trainTestSets.get(1))); // test
		return prbs;
	}
	
	public List<SLProblem> readDataAsSLProblem(String trnFn, String tstFn, TwitterPosLabelSet hwLabels) {
		List<List<TwitterPosExample>> trainTestSets = readData(trnFn, tstFn, hwLabels);
		List<SLProblem> prbs = convertToSLProblem(trainTestSets);
		return prbs;
	}

	public static SLProblem ExampleListToSLProblem(List<TwitterPosExample> insts) {
		SLProblem problem = new SLProblem();
		for (int i = 0; i < insts.size(); i++) {
			HwOutput goutput = insts.get(i).getGoldOutput();
			problem.addExample(insts.get(i), goutput);
		}
		return problem;
	}
	
	public static List<TwitterPosExample> readFromFile(String fn, String[] alphabet) {

		List<TwitterPosExample> results = new ArrayList<TwitterPosExample>();
		
		try {
			BufferedReader br = new BufferedReader(new FileReader(fn));
			String line;
			ArrayList<HwSegment> ltseq = new ArrayList<HwSegment>();
			while ((line = br.readLine()) != null) {
				line = line.trim();
				if (line.equals("")) {
					// end of sequence
					TwitterPosExample inst = new TwitterPosExample(ltseq, alphabet);
					results.add(inst);
					// prepare for the next
					ltseq = new ArrayList<HwSegment>();
				} else {
					HwSegment hwltr = parseSegLocal(line);
					ltseq.add(hwltr);
				}
			}
			
			if (ltseq.size() > 0) {
				TwitterPosExample inst = new TwitterPosExample(ltseq, alphabet);
				results.add(inst);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		int maxLen = -1;
		for (TwitterPosExample inst : results) {
			if (inst.size() > maxLen) {
				maxLen = inst.size();
			}
		}
		System.out.println("Max length " + maxLen);
		System.out.println("Loaded instances " + results.size());
		
		return results;
	}
	
	public static List<HwInstance> readFromFileAsHwInstance(String fn, String[] alphabet) {
		List<TwitterPosExample> l1 = readFromFile(fn, alphabet);
		List<HwInstance> results = new ArrayList<HwInstance>();
		for (TwitterPosExample tex : l1) {
			results.add((HwInstance)tex);
		}
		return results;
	}
	
	private static HwSegment parseSegLocal(String line) {
		
		String[] tokens = line.split("\\s+");
		assert(tokens.length == 3);
		String word = tokens[0];
		String tag = tokens[1];
		String feat = tokens[2];
		
		String[] featElms = feat.split(",");
		double[] featVec = new double[featElms.length];
		for (int i = 0; i < featElms.length; i++) {
			featVec[i] = Double.parseDouble(featElms[i]);
		}
		
		HwSegment seg = new HwSegment(0, featVec, tag);
		return seg;
	}
	
	public static void main(String[] args) {
		TwitterDataReader rdr = new TwitterDataReader();
		TwitterPosLabelSet labels = new TwitterPosLabelSet();
		String f2 = "/home/mc/workplace/rand_search/infnet_twitterpos/CRF/daily547.proc.cnn.txt";
		String f1 = "/home/mc/workplace/rand_search/infnet_twitterpos/CRF/oct27.traindev.proc.cnn.txt";
		rdr.readData(f1, f2, labels);
	}

}
