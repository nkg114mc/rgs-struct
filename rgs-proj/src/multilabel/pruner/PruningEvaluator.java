package multilabel.pruner;

import multilabel.instance.Example;
import multilabel.instance.Label;
import multilabel.learning.StructOutput;
import multilabel.learning.search.OldGreedySearcher;
import multilabel.learning.search.OldSearchState;
import multilabel.utils.UtilFunctions;

import java.util.ArrayList;

import multilabel.data.Dataset;
import multilabel.evaluation.MultiLabelEvaluator;

public class PruningEvaluator {
	
	public static void evaluatePruner(ArrayList<Example> textExs, LabelPruner pruner) {
		evaluatePrunerRecall(textExs, pruner); // pruning recall
		pruningUpperBound(textExs, pruner); // pruning upper-bound
	}
	
	public static void evaluatePrunerRecall(ArrayList<Example> textExs, LabelPruner pruner) {//String modelFile, int topK) {

		
		double totalLabel = 0;
		double total1Lbl = 0;
		double topK1Lbl = 0;
		double unprunedLbl = 0;
		int nlabel = -1;
		
		for (Example exmp : textExs) {
			
			ArrayList<Label> lbl1s = pruner.prunForExample(exmp);
			ArrayList<Label> allLbl = exmp.getLabel();
			
			nlabel = allLbl.size();
			
			totalLabel += allLbl.size();
			for (Label l : allLbl) {
				if (l.value > 0) {
					total1Lbl++;
					if (!l.isPruned) {
						topK1Lbl++;
					}
				}
				if (!l.isPruned) {
					unprunedLbl++;
				}
			}
			
		}
		
		System.out.println("PruningRate = " + pruner.getTopK() + "/" + nlabel);
		System.out.println("Total = " + totalLabel);
		double recall = topK1Lbl / total1Lbl;
		unprunedLbl = pruner.getTopK() * textExs.size();
		double precision = topK1Lbl / unprunedLbl;
		System.out.println("Precision = " + topK1Lbl + "/" + unprunedLbl + " = " + precision);
		System.out.println("Recall = " + topK1Lbl + "/" + total1Lbl + " = " + recall);
	}
	
	
	public static void pruningUpperBound(ArrayList<Example> textExs, LabelPruner pruner) {
		
		for (Example exmp : textExs) {
			
			//ArrayList<Label> lbl1s = pruner.prunForExample(exmp);
			ArrayList<Label> allLbl = exmp.getLabel();
	
			//final StructOutput goldOut = UtilFunctions.getGoldOutputFromExmp(exmp);
			StructOutput bestOutWithPrunning = OldSearchState.getAllZeroState(exmp.labelDim()).getOutput();
			
			for (Label l : allLbl) {
				int idx = l.originIndex;
				if (l.isPruned) {
					bestOutWithPrunning.setValue(idx, 0); // pruned default value: 0
				} else {
					bestOutWithPrunning.setValue(idx, l.value);
				}
			}
			
			exmp.predictOutput = bestOutWithPrunning;
		}
		
		//System.out.println(" ======= Pruner Upperbound ======= ");
		MultiLabelEvaluator evalr = new MultiLabelEvaluator();
		evalr.evaluationDataSet("PRuner UpperBound", textExs);

	}
	
	public static void main(String[] args) {
		testLambdaMartPruner();
	}
	
	public static void testLambdaMartPruner() {
		String name = "medical";
		System.out.println("Name: " + name);
		
		// read dataset
		Dataset ds = PrunerLearning.readArffFileTest(name);
		
		// train!
		String modelPath = "/scratch/large_multi-label/largemultilabel_my/my_largelabel/pruner_rank/medical_rank_lambdamart_p10.txt";
		LabelPruner pruner = new LambdaMartLabelPruner(modelPath, 10);
		PruningEvaluator.evaluatePruner(ds.getTestExamples(), pruner);
		
		// done.
		System.out.println("Done pruner testing.");
	}
	
}
