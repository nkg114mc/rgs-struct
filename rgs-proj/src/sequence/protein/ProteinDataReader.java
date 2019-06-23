package sequence.protein;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import sequence.hw.HwInstance;
import sequence.hw.HwSegment;

public class ProteinDataReader {
	
	public static final int PROTEIN_FEATURE_LENGTH = 231;
	
	public static void main(String[] args) {
		ProteinLabelSet prtLabels = new ProteinLabelSet();
		ProteinDataReader rder = new ProteinDataReader();
		rder.readData("../datasets/protein/protein/sparse.protein.11.train",
				      "../datasets/protein/protein/sparse.protein.11.test", prtLabels.getLabels());
	}
	
	public List<List<HwInstance>> readData(String trainFn, String testFn, String[] lbs) {
		

		checkFileExistance(trainFn);
		checkFileExistance(testFn);
		
		System.out.println("Reading training file: " + trainFn);
		System.out.println("Reading testing file: " + testFn);
		
		List<HwInstance> trainExs = readFromFile(trainFn,lbs);
		List<HwInstance> testExs = readFromFile(testFn,lbs);
		
		List<List<HwInstance>> twoExSet = new ArrayList<List<HwInstance>>();
		twoExSet.add(trainExs);
		twoExSet.add(testExs);
		
		return twoExSet;
	}
	
	public void checkFileExistance(String dfolder) {
		// file
		File dfile = new File(dfolder);
		if (!dfile.exists()) {
			throw new RuntimeException("Can not read " + dfile.getAbsolutePath());
		}
	}
	
	public static List<HwInstance> readFromFile(String fn, String[] alphabet) {

		List<HwInstance> results = new ArrayList<HwInstance>();
		
		try {
			int lineCnt = 0;
			BufferedReader br = new BufferedReader(new FileReader(fn));
			String line;
			ArrayList<HwSegment> ltseq = new ArrayList<HwSegment>();
			
			int seqId = -1;
			int elmId = -1;
	
			int featLen = 0;
			int lbLen = 0;
			int exmpCnt = 0;
			
			while ((line = br.readLine()) != null) {
				line = line.trim();
				lineCnt++;
				String[] tokens = line.split("\\s+");
				
				if (line.equals("")) {
					continue;
				}
				
				if ((lineCnt == 1) && (tokens.length == 3)) {
					featLen = Integer.parseInt(tokens[2]);
					lbLen = Integer.parseInt(tokens[1]);
					exmpCnt = Integer.parseInt(tokens[0]);
					System.out.println("FeatLen = " + featLen + " lbLen = " + lbLen + " exmpCnt = " + exmpCnt);
				} else if (tokens.length == 15) {
					int newId = Integer.parseInt(tokens[0]);
					if (newId != seqId) {
						// end of sequence
						if (ltseq.size() > 0) {
							HwInstance inst = new HwInstance(ltseq,alphabet);
							results.add(inst);
						}
						// prepare for the next
						ltseq = new ArrayList<HwSegment>();
						HwSegment hwltr = parseSeg(tokens, featLen); // new HwSegment(line);
						ltseq.add(hwltr);
					} else {
						HwSegment hwltr = parseSeg(tokens, featLen); // new HwSegment(line);
						ltseq.add(hwltr);
					}
					seqId = newId;
				} else {
					throw new RuntimeException(line);
				}
			}
			
			if (ltseq.size() > 0) {
				HwInstance inst = new HwInstance(ltseq,alphabet);
				results.add(inst);
			}
		} catch (IOException e) {
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
	
	public static HwSegment parseSeg(String[] tks, int density) {
		// 16 312 11 12 30 50 72 99 113 126 164 174 197 210 1
		int seqId = Integer.parseInt(tks[0]);
		int elmId = Integer.parseInt(tks[1]);
		int sparseLen = Integer.parseInt(tks[2]);
		int lb = Integer.parseInt(tks[tks.length - 1]);
		double[] feat = new double[density];
		Arrays.fill(feat, 0);
		for (int i = 0; i < sparseLen; i++) {
			int idx = Integer.parseInt(tks[i + 3]);
			feat[idx] = 1;
		}
		HwSegment seg = new HwSegment(elmId, feat, String.valueOf(lb), true);
		return seg;
	}
	
	/*
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
	}*/

}
