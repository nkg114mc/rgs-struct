package multilabel.pruner;



import ciir.umass.edu.eval.Evaluator;
import multilabel.data.Dataset;

public class ParallelLambdaMartLabelPruner extends LambdaMartLabelPruner {
	
	public ParallelLambdaMartLabelPruner(String modelFileName, int topK) {
		super(modelFileName, topK); 
	}
	
	public static LabelPruner trainPrunerRanklib(Dataset ds, int topK) {
		
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
		String modelFn = ds.name + "_pruner_parallellambdamart_p" + String.valueOf(fullDim) + "to" + String.valueOf(topK) +".txt";
		String modelPath = prunerFolder + "/" + modelFn;
		String basicCmd = "-sparse -tree 1000 -leaf 100 -shrinkage 0.1 -tc -1 -mls 1 -estop 50 -ranker 10";
		String metricCmd = " -metric2t " + "R@" + String.valueOf(topK);
		String fileCmd = " -train " + trainFeatFn + " -validate " + testFeatFn + " -save " + modelPath;
		String trainCmd = basicCmd + metricCmd + fileCmd;
		 
		// 3) run training
		System.out.println("Call command:\n" + trainCmd);
		String[] cmds = trainCmd.split("\\s+");
		Evaluator.main(cmds);
		
		
		////// TEST //////////
		System.out.println("Start testing!");
		LabelPruner pruner = new ParallelLambdaMartLabelPruner(modelPath, topK);
		PruningEvaluator.evaluatePruner(ds.getTestExamples(), pruner);
		
		return pruner;
	}



}
