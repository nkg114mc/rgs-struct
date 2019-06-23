package multilabel.learning.cost;
 
import ciir.umass.edu.learning.DataPoint;
import ciir.umass.edu.learning.SparseDataPoint;
import multilabel.instance.Example;
import multilabel.instance.Featurizer;
import multilabel.instance.OldWeightVector;
import multilabel.learning.search.OldSearchState;
import multilabel.pruner.UMassRankLib;

public class RankingCostFunction extends CostFunction {
	
	
	private UMassRankLib ranker;
	private Featurizer featurizer;
	
	public RankingCostFunction(Featurizer fizer) {
		featurizer = fizer;
	}
	
	public void loadModel(String modelFileName) {
		ranker = new UMassRankLib();
		ranker.loadModelFile(modelFileName);
	}
	
	public double getCost(OldSearchState state, Example ex) {
		OldWeightVector fv = featurizer.getFeatureVector(ex, state.getOutput());
		double sc = predict(fv);
		return sc;
	}
	
	public double predict(OldWeightVector fv) {
		String dpstr = new String("1 qid:1 " + fv.toSparseRanklibStr());
		DataPoint rankSample = new SparseDataPoint( dpstr );
		double sc = ranker.getRankerScore(rankSample);
		//System.out.println("score = " + sc);
		return sc;
	}

}
