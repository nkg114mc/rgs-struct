package multilabel.utils;

import java.io.File;

import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import init.SeqSamplingRndGenerator;

public class WeightDumper {
	
	public String folder;
	
	public String prefix;
	
	public WeightDumper(String fd, String prf) {
		folder = fd;
		prefix = prf;
		SeqSamplingRndGenerator.checkArffFolder(folder);
	}
	
	public String dumpWeight(int iteration, WeightVector wv) {
		 String fileNm = folder + "/" + prefix+ "_" + String.valueOf(iteration);
		 UtilFunctions.saveObj(wv, fileNm);
		 return fileNm;
	}
	
	public WeightVector loadWeight(int iteration) {
		String fileNm = folder + "/" + prefix+ "_" + String.valueOf(iteration);
		WeightVector wv = loadWeightFromFile(fileNm);
		return wv;
	}
	
	public static WeightVector loadWeightFromFile(String fileNm) {
		File wf = new File(fileNm);
		if (wf.exists()) { // ok
			WeightVector wv = (WeightVector) UtilFunctions.loadObj(fileNm);
			//System.out.println("Load weight to file: [" + fileNm + "] with dimension " + wv.getLength());
			return wv;
		} else {
			System.err.println("[WARNING] file [" + fileNm + "] does not exist!");
			return null;
		}
	}
	
	public static boolean checkLength(int len, WeightVector wv) {
		if (wv.getLength() == len) {
			return true;
		}
		return false; // length is not equal...
	}
	
}
