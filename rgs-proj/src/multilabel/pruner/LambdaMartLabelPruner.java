package multilabel.pruner;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import ciir.umass.edu.eval.Evaluator;
import ciir.umass.edu.learning.DataPoint;
import ciir.umass.edu.learning.SparseDataPoint;
import multilabel.data.Dataset;
import multilabel.instance.Example;
import multilabel.instance.Label;
import multilabel.instance.OldWeightVector;

public class LambdaMartLabelPruner extends LabelPruner {
	
	private static String metricName = "R"; // default
	private UMassRankLib ranker;
	private int pruningK;
	
	public LambdaMartLabelPruner(String modelFileName, int topK) {
		ranker = new UMassRankLib();
		ranker.loadModelFile(modelFileName);
		pruningK = topK;
	}
	
	@Override
	public int getTopK() {
		return pruningK;
	}
	
	public static void setMetricName(String nm) {
		metricName = nm;
	}
	
	// return the top-k labels
	public ArrayList<Label> prunForExample(Example example) {
		
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
	}
	
	public double getRankScore(OldWeightVector fv) {
		String dpstr = new String("1 qid:1 " + fv.toSparseRanklibStr());
		//System.out.println(dpstr);
		DataPoint rankSample = new SparseDataPoint( dpstr );
		double sc = ranker.getRankerScore(rankSample);
		//System.out.println("score = " + sc);
		return sc;
	}
	
	/*
	public double getRankScore(WeightVector fv) {
		String dpstr = new String("1 qid:1 " + fv.toDenseRanklibStr());
		//System.out.println(dpstr);
		DataPoint rankSample = new DenseDataPoint( dpstr );
		double sc = ranker.getRankerScore(rankSample);
		//System.out.println("score = " + sc);
		return sc;
	}*/
	
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

	@Override
	public double getScore(OldWeightVector fv) {
		return getRankScore(fv);
	}
	
	
	
	//////////////////////////////////////////////
	
	public static void dumpingDatasetFeatureRanklib(Dataset ds) {
		// 1) dump file
		String prunerFolder = "pruner_rank";
		String trainFeatFn = prunerFolder + "/" + ds.name + "_prune_train_feat.txt";
		String testFeatFn = prunerFolder + "/" + ds.name + "_prune_test_feat.txt";
		PrunerLearning.dumpRanklibFeatureFile(ds.getTrainExamples(),  trainFeatFn);
		PrunerLearning.dumpRanklibFeatureFile(ds.getTestExamples(), testFeatFn);
		System.out.println("Done dumping features!");
	}
	
	public static LabelPruner trainPrunerRanklib(Dataset ds, int topK, String metric) {
		
		if (metric.equals("")) {
			metric = "R"; // default Recall@K
		}
		
		/////// TRAIN ////////
		
		System.out.println("Start training!");
		System.out.println("Dumping Features!");
		
		// 1) dump file
		String prunerFolder = "pruner_rank";
		String trainFeatFn = prunerFolder + "/" + ds.name + "_prune_train_feat.txt";
		String testFeatFn = prunerFolder + "/" + ds.name + "_prune_test_feat.txt";
		PrunerLearning.dumpRanklibFeatureFile(ds.getTrainExamples(),  trainFeatFn);
		PrunerLearning.dumpRanklibFeatureFile(ds.getTestExamples(), testFeatFn);
		
		// 2) prepare runing cmd

		int fullDim = ds.getLabelDimension();
		String modelFn = ds.name + "_pruner_lambdamart_p" + String.valueOf(fullDim) + "to" + String.valueOf(topK) +".txt";
		String modelPath = prunerFolder + "/" + modelFn;
		String basicCmd = "-sparse -tree 1000 -leaf 100 -shrinkage 0.1 -tc -1 -mls 1 -estop 50 -ranker 6";
		//String metricCmd = " -metric2t " + "R@" + String.valueOf(topK);
		String metricCmd = " -metric2t " + metric + "@" + String.valueOf(topK);
		String fileCmd = " -train " + trainFeatFn + " -validate " + testFeatFn + " -save " + modelPath;
		String trainCmd = basicCmd + metricCmd + fileCmd;
		 
		// 3) run training
		System.out.println("Call command:\n" + trainCmd);
		String[] cmds = trainCmd.split("\\s+");
		Evaluator.main(cmds);
		
		
		////// TEST //////////
		System.out.println("Start testing!");
		LabelPruner pruner = new LambdaMartLabelPruner(modelPath, topK);
		PruningEvaluator.evaluatePruner(ds.getTestExamples(), pruner);
		
		return pruner;
	}



}