package elearning;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import elearning.einfer.SearchStateScoringFunction;
import elearning.einfer.XgbSearchStateScorer;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

public class XgbRegressionLearner extends AbstractRegressionLearner {
	
	public String dsname;
	
	public XgbRegressionLearner(String nm) {
		dsname = nm;
	}

	@Override
	public SearchStateScoringFunction regressionTrain(ArrayList<RegressionInstance> regrData, int featLen, int iterNum) {

		DMatrix trainMtrx = createDMatrix(regrData, true);

		//// train regressioner
		Booster bstr = performLearningGvienTrainTestDMatrix(trainMtrx);
		
		return (new XgbSearchStateScorer(bstr));
	}


	public Booster performLearningGvienTrainTestDMatrix(DMatrix trainMax) {
		try {

			//// train
			System.out.println("Trainset size: " + trainMax.rowNum());

			HashMap<String, Object> params = new HashMap<String, Object>();
			params.put("eta", 0.1);
			params.put("max_depth", 20);
			params.put("silent", 1);
			params.put("objective", "reg:linear");
			//params += "eval_metric" -> "pre@1"
			params.put("nthread", 4);

			HashMap<String, DMatrix> watches = new HashMap<String, DMatrix>();
			watches.put("train", trainMax);
			//watches += "test" -> testMax

			int round = 121;
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

	public static DMatrix createDMatrix(ArrayList<RegressionInstance> regrData, boolean verbose) {

		try {

			int totalCnt = 0;
			int maxFeatIdx = 0;
			ArrayList<Float> tlabels = new ArrayList<Float>();
			ArrayList<Float> tdata   = new ArrayList<Float>();
			ArrayList<Long> theaders = new ArrayList<Long>();
			ArrayList<Integer> tindex = new ArrayList<Integer>();
			//val tgroup = new ArrayBuffer[Int]();

			int rowCnt = 0;
			long rowheader = 0L;
			theaders.add(rowheader);

			for (RegressionInstance dp : regrData) {

				rowCnt += 1;
				float lbl = (float)dp.value;

				HashMap<Integer, Double> feat = dp.sparseFeat;
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
				//tgroup += (ment.values.length);
			}

			float[] splabels = listToArrFloat(tlabels);
			float[] spdata = listToArrFloat(tdata);
			int[] spcolIndex = listToArrInt(tindex);
			long[] sprowHeaders = listToArrLong(theaders);

			if (verbose) {
				System.out.println("splabels = " + splabels.length);
				//System.out.println("spgroups = " + spgroups.length);
				System.out.println("spdata = " + spdata.length);
				System.out.println("spcolIndex = " + spcolIndex.length);
				System.out.println("sprowHeaders = " + sprowHeaders.length);
			}

			DMatrix mx = new DMatrix(sprowHeaders, spcolIndex, spdata, DMatrix.SparseType.CSR, 0);

			mx.setLabel(splabels);

			//mx.setGroup(spgroups);

			// print some statistics
			if (verbose) {
				System.out.println("Rows: " + rowCnt);
				//System.out.println("Max feature index: " + maxFeatIdx);
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
	
	public static DMatrix createSingleMatrix(HashMap<Integer, Double> feat, boolean verbose) {

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
