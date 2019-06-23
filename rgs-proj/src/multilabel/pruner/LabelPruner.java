package multilabel.pruner;

import java.util.ArrayList;

import multilabel.data.Dataset;
import multilabel.instance.Example;
import multilabel.instance.Label;
import multilabel.instance.OldWeightVector;

public abstract class LabelPruner {

	public abstract ArrayList<Label> prunForExample(Example example);

	public abstract double getScore(OldWeightVector fv);
	
	public abstract int getTopK();
	
	///////////////////
	
	// do pruning on both training and testing set on a dataset
	public static void pruneDataset(Dataset ds, LabelPruner pruner) {
		pruneExamples(ds.getTrainExamples(), pruner);
		pruneExamples(ds.getTrainExamples(), pruner);
	}
	
	public static void pruneExamples(ArrayList<Example> exs, LabelPruner pruner) {
		for (Example exmp : exs) {
			ArrayList<Label> lbl1s = pruner.prunForExample(exmp);	
		}
	}

}
