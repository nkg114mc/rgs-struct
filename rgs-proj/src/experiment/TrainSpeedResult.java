package experiment;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import essvm.TrainResult;
import essvm.TrainSnapshot;
import experiment.RndLocalSearchExperiment.InitType;

public class TrainSpeedResult {

	// results
	public TrainResult inaccurateResult; // (1)
	public TrainResult accurateResult;   // (2)
	public TrainResult evalResult;       // (3)
	
	public TrainSpeedResult(TrainResult r1, TrainResult r20, TrainResult r1e) {
		inaccurateResult = r1;
		accurateResult = r20;
		evalResult = r1e;
	}
	
		
	public void computeTrainAccuracy(ArrayList<TrainSnapshot> curve) {
		
	}
	
	//////////////////////////////////////////////////////////////////

	public static String initTypeStr(InitType initType) {
		if (initType == InitType.LOGISTIC_INIT) {
			return "logisitc";
		} else if (initType == InitType.UNIFORM_INIT) {
			return "uniform";
		}
		return null;
	}
	
/*
	public static void dumpTrainSpeedCurveCsv(String folder, String dsName, InitType initType, List<TrainSnapshot> curve, String settingName) {

		String fn1 = dsName + "_" + initTypeStr(initType) + "_" + (settingName) + ".csv";

		try {
			PrintWriter pw = new PrintWriter(folder + "/" + fn1);
			for (int i = 1; i < res.size(); i++) {
				double t = baseSpeed / res.get(i).timeConsum;
				double a = baseAcc / res.get(i).accuracy;
				pw.println(res.get(i).x + "," + t + "," + res.get(i).timeConsum + "," + baseSpeed +  "," + a + "," + res.get(i).accuracy + "," + baseAcc);
			}
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
*/
	public void dumpTrainSpeedResult(String folder, String dsName, InitType initType){//, TrainSpeedResult speedResult){//, String settingName) {
		dumpTrainSpeedCurvesCsv(folder, dsName, initType, this.accurateResult.snapshots, "restart20");
		dumpTrainSpeedCurvesCsv(folder, dsName, initType, this.inaccurateResult.snapshots, "restart1");
		dumpTrainSpeedCurvesCsv(folder, dsName, initType, this.evalResult.snapshots, "restartwithEval");
	}
	
	public static void dumpTrainSpeedCurvesCsv(String folder, String dsName, InitType initType,  List<TrainSnapshot> curve, String settingName) {

		String fn1 = dsName + "_" + initTypeStr(initType) + "_" + (settingName) + ".csv";
		
		System.out.println();
		System.out.println("Dump curve to file:" + fn1);

		try {
			PrintWriter pw = new PrintWriter(folder + "/" + fn1);
			for (int i = 1; i < curve.size(); i++) {
				double it = curve.get(i).iter;
				long   tm = curve.get(i).time;
				double acc = curve.get(i).trainAccuracy;
				pw.println(it + "," + tm + "," + acc);
				System.out.println(it + "," + tm + "," + acc);
			}
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}
