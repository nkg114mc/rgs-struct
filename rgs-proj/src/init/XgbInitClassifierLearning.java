package init;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;
import sequence.hw.HwSegment;

public class XgbInitClassifierLearning {
	
	public static Booster trainSingleClassifier(List<HwSegment> segs, String[] alphabet) {
		
		DMatrix trainMtrx = createDMatrix(segs, alphabet, true);
		Booster bstr = performLearningGvienTrainTestDMatrix(trainMtrx, null);
		
		trainMtrx.dispose();
		return (bstr);
	}
	
	public static double scoringGivenBoosterAndFeature(double[] feat, Booster booster) {
		try {
			DMatrix mx = createSingleMatrix(feat, false);

			float[][] predicts = booster.predict(mx);
			float ret = predicts[0][0];
			//System.out.println("predict-lbs: " + predicts[0].length);

			//// release memory
			mx.dispose();

			return (double)ret;
		
		} catch (XGBoostError e) {
			e.printStackTrace();
		}
		
		return -1;
	}

	////////////////////////////////////////////
	////////////////////////////////////////////
	////////////////////////////////////////////


	public static Booster performLearningGvienTrainTestDMatrix(DMatrix trainMax, DMatrix testMax) {
		try {

			//// train
			System.out.println("Trainset size: " + trainMax.rowNum());

			HashMap<String, Object> params = new HashMap<String, Object>();
			params.put("eta", 0.1);
			params.put("max_depth", 20);
			params.put("silent", 0);
			params.put("objective", "binary:logistic");
			params.put("eval_metric", "error");
			params.put("nthread", 1);

			HashMap<String, DMatrix> watches = new HashMap<String, DMatrix>();
			watches.put("train", trainMax);
			//watches.put("test", testMax);

			int round = 100;
			// train a model
			Booster booster = XGBoost.train(trainMax, params, round, watches, null, null);

			// save model to model path
			//File file = new File("./xgbmodel");
			//if (!file.exists()) {
			//	file.mkdirs();
			//}

			//booster.saveModel(file.getAbsolutePath() + "/xgb-regres-"+dsname+".model");
			trainMax.dispose();

			return booster;

		} catch (XGBoostError e) {
			e.printStackTrace();
		}

		return null;
	}

	public static DMatrix createDMatrix(List<HwSegment> regrData, String[] alphabet, boolean verbose) {

		try {

			int totalCnt = 0;
			int maxFeatIdx = 0;
			ArrayList<Float> tlabels = new ArrayList<Float>();
			ArrayList<Float> tdata   = new ArrayList<Float>();
			ArrayList<Long> theaders = new ArrayList<Long>();
			ArrayList<Integer> tindex = new ArrayList<Integer>();

			int rowCnt = 0;
			long rowheader = 0L;
			theaders.add(rowheader);

			for (HwSegment dp : regrData) {

				rowCnt += 1;
				int lbl = dp.goldIndex;

				HashMap<Integer, Double> feat =  arrToMap(MultiLabelXgbRndGenerator.getFeatureWithSingleLabel(dp, lbl));
				for (Integer i : feat.keySet()) {
					int fidx = i.intValue() + 1;
					double fval = feat.get(i);

					tdata.add((float)fval);
					tindex.add(fidx);
				}


				long totalFeats = (long)feat.size();

				rowheader += totalFeats;
				theaders.add(rowheader);
				tlabels.add((float)lbl);
			}

			float[] splabels = listToArrFloat(tlabels);
			float[] spdata = listToArrFloat(tdata);
			int[] spcolIndex = listToArrInt(tindex);
			long[] sprowHeaders = listToArrLong(theaders);

			if (verbose) {
				System.out.println("splabels = " + splabels.length);
				System.out.println("spdata = " + spdata.length);
				System.out.println("spcolIndex = " + spcolIndex.length);
				System.out.println("sprowHeaders = " + sprowHeaders.length);
			}

			DMatrix mx = new DMatrix(sprowHeaders, spcolIndex, spdata, DMatrix.SparseType.CSR, 0);

			mx.setLabel(splabels);

			// print some statistics
			if (verbose) {
				System.out.println("Rows: " + rowCnt);
			}
			return (mx);

		} catch (XGBoostError e) {
			e.printStackTrace();
		}

		return null; // should not arrive here
	}
	
	
	private static float[] listToArrFloat(List<Float> list) {
		float[] arr = new float[list.size()];
		for (int i = 0; i < list.size(); i++) {
			arr[i] = list.get(i);
		}
		return arr;
	}
	
	private static int[] listToArrInt(List<Integer> list) {
		int[] arr = new int[list.size()];
		for (int i = 0; i < list.size(); i++) {
			arr[i] = list.get(i);
		}
		return arr;
	}
	
	private static long[] listToArrLong(List<Long> list) {
		long[] arr = new long[list.size()];
		for (int i = 0; i < list.size(); i++) {
			arr[i] = list.get(i);
		}
		return arr;
	}
	
	private static HashMap<Integer, Double> arrToMap(double[] f) {
		HashMap<Integer, Double> feat = new HashMap<Integer, Double>();
		for (int i = 0; i < f.length; i++) {
			if (f[i] != 0) {
				feat.put(i, f[i]);
			}
		}
		return feat;
	}
	
	public static DMatrix createSingleMatrix(double[] ft,  boolean verbose) {

		try {

			ArrayList<Float> tlabels = new ArrayList<Float>();
			ArrayList<Float> tdata   = new ArrayList<Float>();
			ArrayList<Long> theaders = new ArrayList<Long>();
			ArrayList<Integer> tindex = new ArrayList<Integer>();

			int rowCnt = 0;
			long rowheader = 0L;
			theaders.add(rowheader);

			rowCnt += 1;
			float lbl = 0f;
			HashMap<Integer, Double> feat = arrToMap(ft);

			for (Integer i : feat.keySet()) {
				int fidx = i.intValue() + 1;
				double fval = feat.get(i);

				tdata.add((float)fval);
				tindex.add(fidx);
			}


			long totalFeats = (long)feat.size();

			rowheader += totalFeats;
			theaders.add(rowheader);
			tlabels.add(lbl);

			float[] splabels = listToArrFloat(tlabels);
			float[] spdata = listToArrFloat(tdata);
			int[] spcolIndex = listToArrInt(tindex);
			long[] sprowHeaders = listToArrLong(theaders);

			if (verbose) {
				System.out.println("splabels = " + splabels.length);
				System.out.println("spdata = " + spdata.length);
				System.out.println("spcolIndex = " + spcolIndex.length);
				System.out.println("sprowHeaders = " + sprowHeaders.length);
			}

			DMatrix mx = new DMatrix(sprowHeaders, spcolIndex, spdata, DMatrix.SparseType.CSR, 0);
			mx.setLabel(splabels);

			if (verbose) {
				System.out.println("Rows: " + rowCnt);
			}
			return (mx);

		} catch (XGBoostError e) {
			e.printStackTrace();
		}

		return null; // should not arrive here
	}
}
