package berkeleyentity.mentions;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

public class StateActionXgboostPredictor {

	private Booster booster;

	public StateActionXgboostPredictor() { // for training
		booster = null;
	}

	public StateActionXgboostPredictor(String modelFn) { // for testing, with a loaded model
		loadModelFile(modelFn);
	}

	public void loadModelFile(String modelPath)
	{
		System.out.println("Loading xgboost model file: " + modelPath);
		try {
			booster = XGBoost.loadModel(modelPath);
		} catch (XGBoostError e) {
			e.printStackTrace();
		}
	}
	
	public Booster getBooster() {
		return booster;
	}
/*
	@Override
	public double scoringStateActionGivenFeatureVec(double[] featVec) {
		try {
			DMatrix singleVec = featVecToDMatrix(featVec, false);
			float[][] predicts = booster.predict(singleVec);
			double sc = predicts[0][0];
			return sc;
		} catch (XGBoostError e) {
			e.printStackTrace();
		}
		
		return 0;
	}
*/
	public double getRankScore(double[] featVec) {
		throw new RuntimeException("not implemented ...");
	}
	
	public void trainRanker(String givenModelName, String trainFeatName, String validFeatName, int prunerBeamSize) {
		
		try {

			String modelPath = "./ontonotes5-xgb.model";
			
			//////////////////////////////////////////////////////


			//// train

			///////////////////////////////////////////////////////

			DMatrix trainMat = loadSvmrankFileToDMatrix(trainFeatName, true);
			DMatrix testMat = loadSvmrankFileToDMatrix(validFeatName, true);


			System.out.println("Trainset size: " + trainMat.rowNum());
			System.out.println("Testset size: " + testMat.rowNum());

			HashMap<String, Object> params = new HashMap<String, Object>();
			params.put("eta", 0.1);
			params.put("max_depth", 10);
			params.put("silent", 0);
			params.put("objective", "rank:pairwise");
			params.put("eval_metric", "pre@1");
			params.put("nthread", 4);
			//params += "colsample_bytree" -> 0.9
			//params += "min_child_weight" -> 10

			HashMap<String, DMatrix> watches = new HashMap<String, DMatrix>();
			watches.put("train", trainMat);
			watches.put("test", testMat);

			//set round
			int round = 600;

			//train a boost model
			Booster booster = XGBoost.train(trainMat, params, round, watches, null, null);

			//save model to modelPath
			booster.saveModel(modelPath);

		} catch (XGBoostError e) {
			e.printStackTrace();
		}
		//return booster;
	}
	

	////////////////////////////////////////////////////////////////////////////////
	///// Construct DMatrix from product items  ////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////
	

	public static DMatrix featVecToDMatrix(double[] feat, boolean verbose) {
		
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

			int firedFeat = 0;
			for (int i = 0; i < feat.length; i++) {
				int fidx = i + 1;
				double fval = feat[i];
				if (fval != 0) {
					firedFeat++;
					tdata.add((float)fval);
					tindex.add(fidx);
				}
			}


			long totalFeats = (long)firedFeat;

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
	
	public static DMatrix strFeatVecToDMatrix(String featStr, boolean verbose) {
		
		try {
			
			String[] terms = featStr.trim().split("\\s+");
			
			float[] ffv = new float[140];
			Arrays.fill(ffv, 0);
			
			for (int i = 2; i < terms.length; i++) {
				int fidx = Integer.parseInt(StateActionXgboostPredictor.getLeft(terms[i]));
				float fval = Float.parseFloat(StateActionXgboostPredictor.getRight(terms[i]));
				ffv[fidx - 1] = fval;
			}
			
			DMatrix mx = new DMatrix(ffv,  1, ffv.length, 0);
			return (mx);

		} catch (XGBoostError e) {
			e.printStackTrace();
		}

		return null; // should not arrive here
	}
	
	public static DMatrix loadSvmrankFileToDMatrix(String filePath, boolean verbose) {
		
		try {

			ArrayList<Float> tlabels = new ArrayList<Float>();
			ArrayList<Float> tdata   = new ArrayList<Float>();
			ArrayList<Long> theaders = new ArrayList<Long>();
			ArrayList<Integer> tindex = new ArrayList<Integer>();
			ArrayList<Integer> tgroup = new ArrayList<Integer>();

			int rowCnt = 0;
			long rowheader = 0L;
			theaders.add(rowheader);
			
			String line;
			FileReader reader;
			BufferedReader br;
			
			try {
				reader = new FileReader(filePath);
				br = new BufferedReader(reader);
				
				int lastQid = -1;
				int curGroupCnt = 0;
				while ((line = br.readLine()) != null) {
					
					String[] terms = line.trim().split("\\s+");
					
					rowCnt += 1;
					float lbl = Float.parseFloat(terms[0]);
					int qid = Integer.parseInt(getRight(terms[1]));

					for (int i = 2; i < terms.length; i++) {
						int fidx = Integer.parseInt(getLeft(terms[i]));
						float fval = Float.parseFloat(getRight(terms[i]));
						tdata.add(fval);
						tindex.add(fidx);
					}


					long totalFeats = (long)(terms.length - 2);

					rowheader += totalFeats;
					theaders.add(rowheader);
					tlabels.add(lbl);
					
					if (lastQid != qid) {
						if (curGroupCnt > 0) {
							tgroup.add(curGroupCnt);
						}
						curGroupCnt = 1;
					} else {
						curGroupCnt++;
					}
					
					lastQid = qid;
				}
				
				if (curGroupCnt > 0) {
					tgroup.add(curGroupCnt);
				}
				
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (ClassCastException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
			
			int gsum = computeSum(tgroup);
			System.out.println(gsum +"=="+ rowCnt + " " + tgroup.size());
			assert (gsum == rowCnt);
			
			if (verbose) {
				System.out.println("Start list-to-array converting");
			}
			

			float[] splabels = listToArrFloat(tlabels);
			float[] spdata = listToArrFloat(tdata);
			int[] spcolIndex = listToArrInt(tindex);
			long[] sprowHeaders = listToArrLong(theaders);
			int[] spgroup = listToArrInt(tgroup);
			

			if (verbose) {
				System.out.println("splabels = " + splabels.length);
				System.out.println("spdata = " + spdata.length);
				System.out.println("spcolIndex = " + spcolIndex.length);
				System.out.println("sprowHeaders = " + sprowHeaders.length);
				System.out.println("spgroup = " + spgroup.length);
			}

			DMatrix mx = new DMatrix(sprowHeaders, spcolIndex, spdata, DMatrix.SparseType.CSR, 0);
			mx.setLabel(splabels);
			mx.setGroup(spgroup);

			if (verbose) {
				System.out.println("Rows: " + rowCnt);
			}
			return (mx);

		} catch (XGBoostError e) {
			e.printStackTrace();
		}
		
		return null;
	}
	
	private static int computeSum(List<Integer> intlist) {
		int sum = 0;
		for (Integer i : intlist) {
			//System.out.println("group " + i.intValue());
			sum += i.intValue();
		}
		return sum;
	}
	
	public static String getLeft(String term) {
		if (term.contains(":")) {
			String[] arr = term.split(":");
			return arr[0];
		} else {
			throw new RuntimeException("No : in term " + term);
		}
	}
	
	public static String getRight(String term) {
		if (term.contains(":")) {
			String[] arr = term.split(":");
			return arr[1];
		} else {
			throw new RuntimeException("No : in term " + term);
		}
	}
	
/*
	public static float[] listToArrFloat(List<Float> list) {
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
*/
	public static float[] listToArrFloat(List<Float> list) {
		float[] arr = new float[list.size()];
		for (int i = 0; i < list.size(); i++) {
			arr[i] = list.get(i);
		}
		list.clear();
		System.gc();
		return arr;
	}
	
	private static int[] listToArrInt(List<Integer> list) {
		int[] arr = new int[list.size()];
		for (int i = 0; i < list.size(); i++) {
			arr[i] = list.get(i);
		}
		list.clear();
		System.gc();
		return arr;
	}
	
	private static long[] listToArrLong(List<Long> list) {
		long[] arr = new long[list.size()];
		for (int i = 0; i < list.size(); i++) {
			arr[i] = list.get(i);
		}
		list.clear();
		System.gc();
		return arr;
	}
}
