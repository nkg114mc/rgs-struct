package multilabel.pruner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;
import multilabel.data.DatasetReader;
import multilabel.data.HwInstanceDataset;
import multilabel.instance.Label;
import sequence.hw.HwInstance;
import sequence.hw.HwSegment;

public class HwMultiLabelPruner {
	
	///////////////////////////////////
	public static void main(String[] args) {
		String dsName = "bibtex";//"bookmarks-umass";//"yeast";//"yeast";//
		HwInstanceDataset ds = DatasetReader.loadHwInstances(dsName);
		trainRanker(dsName, ds.getTrainExamples(), ds.getTestExamples(), false);
		System.out.println("=======");
	}
	
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	
	
	//// Training
	
	public static Booster trainRanker(String datasetName, List<HwInstance> trnInsts, List<HwInstance> tstInsts, boolean doDebug) {
		
		try {
			System.out.println("==== Train Model on TrainSet ====");

			DMatrix trainMtrx = createDMatrix(trnInsts, true);
			DMatrix testMtrx = createDMatrix(tstInsts, true);
			Booster booster = performLearningGvienTrainTestDMatrix(trainMtrx, testMtrx);
			
			trainMtrx.dispose();
			testMtrx.dispose();

			///// quick test
			System.out.println("==== Eval Logistic Model on TrainSet ====");
			testLogisticModel(trnInsts, booster);
			System.out.println("");
			System.out.println("==== Eval Logistic Model on TestSet ====");
			testLogisticModel(tstInsts, booster);

			return booster;
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return null; // should reach here ...
	}

	public static void testLogisticModel(List<HwInstance> trnInsts, Booster booster) {
		
		
		int labelCnt = trnInsts.get(0).size();
		
		try {
			
			int totalOne = 0;
			int[] tp = new int[labelCnt];
			Arrays.fill(tp, 0);
			
			for (int j = 0; j < trnInsts.size(); j++) {
				HwInstance ins = trnInsts.get(j);
				ArrayList<Label> sortedLabels = rankScoringForExample(ins, booster);

				int extp = 0;
				for (int i = 0; i < sortedLabels.size(); i++) {
					if (sortedLabels.get(i).value > 0) {
						extp++;
						totalOne++;
					}
					tp[i] += extp;
				}
			}
			
			///////////////
			
			for (int l = 0; l < labelCnt; l++) {
				double recall = (((double)tp[l]) / ((double)totalOne));
				System.out.println("Top-" + (l+1) +" Recall: " + tp[l] + "/" + totalOne + " = " + recall);
			}
			
		} catch (Exception e) {
			e.printStackTrace();
		}	

	}
	
	public static int getGoldValueIdx(String[] alphabet, String goldlb) {
		int gv = -1;
		for (int i = 0; i < alphabet.length; i++) {
			if (goldlb.equals(alphabet[i])) {
				gv = i;
				break;
			}
		}
		if (gv < 0) {
			throw new RuntimeException("Gold value index = " + gv);
		}
		return gv;
	}

	public static HashMap<Integer, Double> featurizeSingleLabel(HwSegment seg, int whichLabel) {
		
		HashMap<Integer, Double> sparseValues = new HashMap<Integer, Double>();
		double[] commonMlFeat = seg.getFeatArr();
		
		// unary features
		int i = whichLabel;
		double[] feat = commonMlFeat;
		for (int j = 0; j < feat.length; j++) {
			int idx = i * feat.length + j;//getCompatibleFeatIndex(i, j);
			if (feat[j] != 0) {
				sparseValues.put(idx, feat[j]);
			}
		}
		
		return sparseValues;
	}

	////////////////////////////////////////////
	////////////////////////////////////////////
	////////////////////////////////////////////

	////////////////////////////////////////////
	////////////////////////////////////////////
	////////////////////////////////////////////

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
			params.put("objective", "rank:pairwise");
			params.put("eval_metric", "pre@100");
			params.put("nthread", 1);

			HashMap<String, DMatrix> watches = new HashMap<String, DMatrix>();
			watches.put("train", trainMax);
			watches.put("test", testMax);

			int round = 200;
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

	public static DMatrix createDMatrix(List<HwInstance> regrData, boolean verbose) {

		try {

			//int totalCnt = 0;
			//int maxFeatIdx = 0;
			ArrayList<Float> tlabels = new ArrayList<Float>();
			ArrayList<Float> tdata   = new ArrayList<Float>();
			ArrayList<Long> theaders = new ArrayList<Long>();
			ArrayList<Integer> tindex = new ArrayList<Integer>();
			ArrayList<Integer> tgroups = new ArrayList<Integer>();

			int rowCnt = 0;
			long rowheader = 0L;
			theaders.add(rowheader);

			for (int i2 = 0; i2 < regrData.size(); i2++) {
				
				List<HwSegment> segs = regrData.get(i2).letterSegs;
				
				if (i2 % 1000 == 0) {
					System.out.println("DMatrix finsih " + i2 + " rows.");
				}
				
				for (int j = 0; j < segs.size(); j++) {
					HwSegment seg = segs.get(j);
					seg.goldIndex = getGoldValueIdx(regrData.get(i2).alphabet, seg.letter);
					
					rowCnt += 1;
					int lbl = seg.goldIndex;

					HashMap<Integer, Double> feat = featurizeSingleLabel(seg, j);
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
				
				tgroups.add(segs.size());
			}

			float[] splabels = listToArrFloat(tlabels);
			float[] spdata = listToArrFloat(tdata);
			int[] spcolIndex = listToArrInt(tindex);
			long[] sprowHeaders = listToArrLong(theaders);
			int[] sgroups = listToArrInt(tgroups);

			if (verbose) {
				System.out.println("spdata = " + spdata.length);
				System.out.println("spcolIndex = " + spcolIndex.length);
				System.out.println("sprowHeaders = " + sprowHeaders.length);
				System.out.println("splabels = " + splabels.length);
				System.out.println("sgroups = " + sgroups.length);
			}

			DMatrix mx = new DMatrix(sprowHeaders, spcolIndex, spdata, DMatrix.SparseType.CSR, 0);
			mx.setLabel(splabels);
			mx.setGroup(sgroups);

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

	
	///////////////////////////////////////////////////
	///////////////////////////////////////////////////
	///////////////////////////////////////////////////

	///////////////////////////////////////////////////
	///////////////////////////////////////////////////
	///////////////////////////////////////////////////
	
	///////////////////////////////////////////////////
	///////////////////////////////////////////////////
	///////////////////////////////////////////////////
	
	public static DMatrix createSingleMatrix(List<HwSegment> segs, String[] alphabet, boolean verbose) {
		
		try {

			ArrayList<Float> tlabels = new ArrayList<Float>();
			ArrayList<Float> tdata   = new ArrayList<Float>();
			ArrayList<Long> theaders = new ArrayList<Long>();
			ArrayList<Integer> tindex = new ArrayList<Integer>();
			ArrayList<Integer> tgroups = new ArrayList<Integer>();

			int rowCnt = 0;
			long rowheader = 0L;
			int groupCnt = 1;
			theaders.add(rowheader);


			for (int j = 0; j < segs.size(); j++) {
				HwSegment seg = segs.get(j);
				seg.goldIndex = getGoldValueIdx(alphabet, seg.letter);

				rowCnt += 1;
				int lbl = seg.goldIndex;

				HashMap<Integer, Double> feat = featurizeSingleLabel(seg, j);
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
				tgroups.add(groupCnt);
			}


			float[] splabels = listToArrFloat(tlabels);
			float[] spdata = listToArrFloat(tdata);
			int[] spcolIndex = listToArrInt(tindex);
			long[] sprowHeaders = listToArrLong(theaders);
			int[] sgroups = listToArrInt(tgroups);

			if (verbose) {
				System.out.println("splabels = " + splabels.length);
				System.out.println("spdata = " + spdata.length);
				System.out.println("spcolIndex = " + spcolIndex.length);
				System.out.println("sprowHeaders = " + sprowHeaders.length);
				System.out.println("sgroups = " + sgroups.length);
			}

			DMatrix mx = new DMatrix(sprowHeaders, spcolIndex, spdata, DMatrix.SparseType.CSR, 0);
			mx.setLabel(splabels);
			mx.setGroup(sgroups);

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
	
	// return the top-k labels
	public static ArrayList<Label> rankScoringForExample(HwInstance ins, Booster booster) {
		
		try {

			DMatrix mx = createSingleMatrix(ins.letterSegs, ins.alphabet, false);
			float[][] predicts= booster.predict(mx);

			//// release memory
			mx.dispose();

			ArrayList<Label> labels = new ArrayList<Label>();
			for (int i = 0; i < ins.letterSegs.size(); i++) {
				Label lbl = new Label(i, ins.letterSegs.get(i).goldIndex);
				lbl.rankScore = predicts[i][0];
				//System.out.println("RankScore = " + lbl.rankScore + " " + lbl.value);
				labels.add(lbl);
			}

			// sort according to rank score
			Collections.sort(labels, new LabelPrunComparator());
			return labels;

		} catch (XGBoostError e) {
			e.printStackTrace();
		}
		
		return null;
	}
/*
	// return the top-k labels
	public ArrayList<Label> prunForExample(HwInstance ins) {
		
		OldWeightVector[] allwv = PrunerFeaturizer.featurizeAll(example);
		
		ArrayList<Label> labels = example.getLabel(); 
		for (int i = 0; i < labels.size(); i++) {
			Label lbl = labels.get(i);
			lbl.rankScore = getRankScore(allwv[i]);
			//System.out.println("RankScore = " + lbl.rankScore + " " + lbl.value);
		}
		
		// sort according to rank score
		Collections.sort(labels, new LabelPrunComparator());
		
		ArrayList<Label> topkLabels = new ArrayList<Label>();
		
		for (int i = 0; i < labels.size(); i++) {
			Label lbl = labels.get(i);
			lbl.isPruned = true;
			if (i < pruningK) {
				lbl.isPruned = false;
				topkLabels.add(lbl);
			}
			//int gtrtuh = goldLabels.get(i).value;
			//writer.println(rank + " " + "qid:" + qid + " " + allwv[i].toSparseRanklibStr());
		}
		
		// recover the ordering
		Collections.sort(labels, new LabelIndexComparator());
		
		return topkLabels;

		
		return labels;
	}
*/
	
	public static class LabelPrunComparator implements Comparator<Label> {
		@Override
		public int compare(Label l1, Label l2) {
			if (l1.rankScore > l2.rankScore) {
				return -1;
			} else if (l1.rankScore < l2.rankScore) {
				return 1;
			}
			return 0;
		}
	}
	
	public static class LabelIndexComparator implements Comparator<Label> {
		@Override
		public int compare(Label l1, Label l2) {
			if (l1.originIndex < l2.originIndex) {
				return -1;
			} else if (l1.originIndex > l2.originIndex) {
				return 1;
			}
			return 0;
		}
	}
	
	
}
