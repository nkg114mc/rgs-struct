package experiment;

import java.io.File;

import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearning.LowLevelCostLearning.StopType;
import experiment.RndLocalSearchExperiment.InitType;
import experiment.RndLocalSearchExperiment.MulLbLossType;
import multilabel.utils.UtilFunctions;

public class CostFuncCacherAndLoader {
	
	private String folder;
	private File folderFile;
	
	public static boolean cacheCostWeight = true;
	public static String defaultFolder = "../CacheCost";
	
	public CostFuncCacherAndLoader(String fd) {
		checkArffFolder(fd);
		folder = fd;
		folderFile = new File(fd);
	}
	
	public static String initTypeStr(InitType initType, double initAlfa) {
		if (initType == InitType.LOGISTIC_INIT) {
			return "logisitc";
		} else if (initType == InitType.UNIFORM_INIT) {
			return "uniform";
		} else if (initType == InitType.ALLZERO_INIT) {
			return "allzero";
		} else if (initType == InitType.ALPHA_INIT) {
			return ("alpha" + String.valueOf(initAlfa));//"logisitc";  //// the same as 
		}
		return null;
	}
	
	public static String stopTypeStr(StopType stopTyp) {
		if (stopTyp == StopType.ITER_STOP) {
			return "iterstop";
		} else if (stopTyp == StopType.PERC_STOP) {
			return "percstop";
		}
		return null;
	}
	
	public static String getFileName(String name, InitType initTp, int restarts, int featDim, MulLbLossType loss, float ssvmC, double initAlfa) {
		//String fnm = "cached_" + name + "_" + initTypeStr(initTp) + "_" + "restr" + String.valueOf(restarts) + "_" + "feat" + String.valueOf(featDim) + "_" + "loss" + String.valueOf(loss)+ "_" + "c" + String.valueOf(ssvmC) + ".cost";
		String fnm = "cached_" + name + "_" + initTypeStr(initTp, initAlfa) + "_" + "restr" + String.valueOf(restarts) + "_" + "feat" + String.valueOf(featDim) + "_" + "loss" + String.valueOf(loss) + ".cost";
		return fnm;
	}
	
	
	public String getFolder() {
		return folder;
	}

	public static boolean checkLength(int len, WeightVector wv) {
		if (wv.getLength() == len) {
			return true;
		}
		return false; // length is not equal...
	}
	
	public WeightVector loadCachedWeight(String name, InitType initTp, int restarts, int featDim, MulLbLossType loss, float ssvmC, double initAlfa) {
		String fileNm = folder + "/" + getFileName(name, initTp, restarts, featDim, loss, ssvmC, initAlfa);
		File wf = new File(fileNm);
		if (wf.exists()) { // ok
			WeightVector wv = (WeightVector) UtilFunctions.loadObj(fileNm);
			System.out.println("Load weight to file: [" + fileNm + "] with dimension " + wv.getLength());
			return wv;
		} else {
			System.err.println("[WARNING] file [" + fileNm + "] does not exist!");
			return null;
		}
	}
	
	public void saveCachedWeight(WeightVector wv, String name, InitType initTp, int restarts, int featDim, MulLbLossType loss, float ssvmC, double initAlfa) {
		String fileNm = folder + "/" + getFileName(name, initTp, restarts, featDim, loss, ssvmC, initAlfa);
		UtilFunctions.saveObj(wv, fileNm);
		System.out.println("Cache weight to file: [" + fileNm + "] with dimension " + wv.getLength());
	}
	
/*
	// functions without alpha
	public WeightVector loadCachedWeight(String name, InitType initTp, int restarts, int featDim, MulLbLossType loss, float ssvmC) {
		return loadCachedWeight(name, initTp, restarts, featDim, loss, ssvmC, -1);
	}
	public void saveCachedWeight(WeightVector wv, String name, InitType initTp, int restarts, int featDim, MulLbLossType loss, float ssvmC) {
		saveCachedWeight(wv, name, initTp, restarts, featDim, loss, ssvmC, -1);
	}
*/
	
	public void checkArffFolder(String fdPath) {
		File fd = new File(fdPath);
		if (!fd.exists()) {
			System.err.println("[WARNING] folder " + fdPath + " does not exist!");
			// create folder
			fd.mkdir();
			System.out.println("Create folder: " + fdPath);
		} else if (fd.exists()) {
			if (fd.isDirectory()) {
				// ok
			} else {
				throw new RuntimeException(fd + ": Folder exists and is not a folder!");
			}
		}
	}
	
	public static int getFeatDim(boolean usePairFeat, boolean useTernFeat, boolean useQuadFeat) {
		if (useQuadFeat) {
			return 4;
		} else if (useTernFeat) {
			return 3;
		} else if (usePairFeat) {
			return 2;
		}
		return 1; // should not reach here
	}
	
	public static void main(String[] args) {

	}

}
