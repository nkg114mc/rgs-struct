package berkeleyentity.randsearch;

import java.util.HashMap;

import berkeleyentity.coref.PairwiseIndexingFeaturizer;
import edu.berkeley.nlp.futile.fig.basic.Indexer;
import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.util.FeatureVectorBuffer;
import edu.illinois.cs.cogcomp.sl.util.IFeatureVector;
import general.AbstractFeaturizer;
import general.AbstractInstance;
import general.AbstractOutput;
//import ims.hotcoref.mentiongraph.Edge;
//import ims.hotcoref.oregonstate.HotCorefDocInstance;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;

public class AceCorefFeaturizer extends AbstractFeaturizer {

	private static final long serialVersionUID = -2934915987222021246L;
	
	boolean useHighOrder;
	
	PairwiseIndexingFeaturizer mpairFeaturizer;
	Indexer<String> featureIndexer;
	boolean addToIdxer = true;
	

	public AceCorefFeaturizer(PairwiseIndexingFeaturizer pf, boolean useho) {
		useHighOrder = useho;
		mpairFeaturizer = pf;
		featureIndexer = pf.getIndexer();
	}

	public void setAddToIndexer(boolean ati) {
		addToIdxer = ati;
	}
	public boolean getAddToIndexer() {
		return addToIdxer;
	}
	public void openIndexer() {
		addToIdxer = true;
	}
	public void closeIndexer() {
		addToIdxer = false;
	}
	
	public int[] getEdgeFeat(AceCorefInstance cinst, int curIdx, int anteDecisionIdx) {//, boolean addToIndexer) {
		int[] f = cinst.getMentPairFeature(curIdx, anteDecisionIdx);
		return f;
	}
	
	
	
	
	
	private void addValueToVector(HashMap<Integer,Double> myMap, int index, double value) {
		if (myMap.containsKey(index)) {
			double newV = myMap.get(index).doubleValue() + value;
			myMap.put(index, newV);
		} else {
			myMap.put(index, value);
		}
	}

	
	@Override
	public HashMap<Integer, Double> featurize(AbstractInstance xi, AbstractOutput yi) {
		
		AceCorefInstance x = (AceCorefInstance)xi; 
		HwOutput y = (HwOutput)yi;
		
		HashMap<Integer, Double> sparseValues = new HashMap<Integer, Double>();
		
		// mention pair features
		for (int i = 0; i < x.size(); i++) {
			//if (y.getOutput(i) > 0) {
				int ante = y.getOutput(i);
				int cur = i;
				int[] efeat = getEdgeFeat(x, cur, ante);//, addToIdxer);
				for (int j = 0; j < efeat.length; j++) {
					int idx = efeat[j];
					addValueToVector(sparseValues, idx, 1.0);
					//sparseValues.put(idx, 1.0);
				}
			//}
		}
		
/*
		// cluster-mention features
		if (considerPairs) {
				for (int i = 0; i < y.size(); i++) {
					for (int j = 0; j < y.size(); j++) {
						if (i != j) {
							//String pairfn = getLabelPairFeatName(i, j, y.getOutput(i), y.getOutput(j));
							//int idx2 = getIndex(pairfn);
							int idx2 = getLabelPairFeatIndex(i, j, y.getOutput(i), y.getOutput(j));
							
							//chechIdx(idx2, pairfn);
							sparseValues.put(idx2, 1.0);
						}
					}
				}
		}
*/
		return sparseValues;
	}

	@Override
	public int getFeatLen() {
		return featureIndexer.size();
	}

	@Override
	public IFeatureVector getFeatureVector(IInstance xi, IStructure yi) {
		FeatureVectorBuffer fv = new FeatureVectorBuffer();
		HwInstance x = (HwInstance) xi;
		HwOutput y = (HwOutput) yi;
		
		HashMap<Integer, Double> feats = featurize(x, y);
		for (int idx : feats.keySet()) {
			fv.addFeature(idx, feats.get(idx));			
		}
		return fv.toFeatureVector();
	}
	
	@Override
	public IFeatureVector getFeatureVectorDiff(IInstance x, IStructure y1, IStructure y2) {
		IFeatureVector f1 = getFeatureVector(x, y1);
		IFeatureVector f2 = getFeatureVector(x, y2);		
		return f1.difference(f2);
	}
	
}
