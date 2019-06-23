package berkeleyentity.randsearch;

import java.util.ArrayList;
import java.util.List;

import edu.illinois.cs.cogcomp.sl.core.AbstractInferenceSolver;
import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.latentsvm.AbstractLatentInferenceSolver;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearning.EInferencer;
import general.AbstractInstance;
import general.AbstractLossFunction;
import general.AbstractOutput;
import search.GreedySearcher;
import search.SearchResult;
import search.SearchState;
import search.SearchTrajectory;
import sequence.hw.HwOutput;
import sequence.hw.HwSearchInferencer;

public class AceCorefInferencerEdgeOnly extends AceCorefInferencer {


	private static final long serialVersionUID = 4945014988220649255L;
	
	AbstractLossFunction lossfunc;

	public AceCorefInferencerEdgeOnly(GreedySearcher gschr, GreedySearcher goldschr) {
		super(gschr, goldschr);
		lossfunc = gsearcher.getLossFunc();
	}


	@Override
	public IStructure getBestStructure(WeightVector wv, IInstance input) throws Exception {
		SearchResult result = getEdgeOnlyInferenceResult(wv, (AbstractInstance)input, null, false, false);//gsearcher.runSearchWithRestarts(wv, gsearcher.getEvalInferencer(), (AbstractInstance)input, null, false);
		AbstractOutput resultOutput = result.predState.structOutput;
		return ((IStructure)(resultOutput));
	}
	
	@Override
	public float getLoss(IInstance ins, IStructure goldStructure,  IStructure structure) {
		return gsearcher.getLossFloat((AbstractInstance)ins, goldStructure, structure);
	}

	@Override
	public IStructure getLossAugmentedBestStructure(WeightVector wv, IInstance input, IStructure gold) throws Exception {
		SearchResult result = getEdgeOnlyInferenceResult(wv, (AbstractInstance)input, (AbstractOutput)gold, true, false);
		AbstractOutput resultOutput = result.predState.structOutput;
		return ((IStructure)(resultOutput));
	}

	@Override
	public SearchResult runSearchInference(WeightVector wv, EInferencer eifr, IInstance input, IStructure gold) {
		SearchResult result = getEdgeOnlyInferenceResult(wv, (AbstractInstance)input, (AbstractOutput)gold, false, false);
		return (result);
	}

	@Override
	public SearchResult runSearchInferenceMaybeLossAug(WeightVector wv, EInferencer eifr, IInstance input, IStructure gold, boolean losAug) {
		SearchResult result = getEdgeOnlyInferenceResult(wv, (AbstractInstance)input, (AbstractOutput)gold, losAug, false);
		return (result);
	}


	@Override
	public AbstractInferenceSolver clone() {
		HwSearchInferencer cp = new  HwSearchInferencer(gsearcher);
		return cp;
	}

	@Override
	public IStructure getBestLatentStructure(WeightVector weight, IInstance ins, IStructure gold) throws Exception {
		SearchResult result = getEdgeOnlyInferenceResult(weight, (AbstractInstance)ins, (AbstractOutput)gold, false, true);
		AbstractOutput resultOutput = result.predState.structOutput;
		return ((IStructure)(resultOutput));
	}

	
	public SearchResult getEdgeOnlyInferenceResult(WeightVector weight, AbstractInstance ins, AbstractOutput gold, boolean doLossAug, boolean useGold) {
		
		
		if (doLossAug) {
			assert (gold != null);
			assert (useGold != true);
		} else {
			
		}
		
		double[] w = weight.getDoubleArray();
		AceCorefInstance cinst = (AceCorefInstance)ins;
		
		HwOutput pred = new HwOutput(ins.size(), AceCorefInstance.dummyDomain(ins.size()));
		double totalScore = 0;
		for (int i = 0; i < ins.size(); i++) {
			int[] domain = cinst.getDomainGivenIndex(i);
			if (useGold) {
				domain = cinst.getGoldDomainGivenIndex(i);
			}
			
			if (domain.length == 0) {
				pred.setOutput(i, i);
			} else {
				double bestScore = Double.NEGATIVE_INFINITY;
				int bestAnte = -1;
				for (int j = 0; j < domain.length; j++) {
					int ante = domain[j];
					//////////////////////
					double edgeLoss = 0;
					if (doLossAug) {
						if (!cinst.isCorrectAnteIndex(i, ante)) {
							edgeLoss = 1.0;
						}
					}
					int[] edgeFeat = cinst.getMentPairFeature(i, ante);
					double score = AceCorefFactorGraph.scoreWithOneHotFeature(edgeFeat, w);
					double sc = score + edgeLoss;
					//////////////////////
					
					if (bestScore < sc) {
						bestScore = sc;
						bestAnte = ante;
					}
				}
				
				assert (bestAnte != -1);
				assert (bestScore != Double.NEGATIVE_INFINITY);
				pred.setOutput(i, bestAnte);
				totalScore += bestScore;
			}
		}
		
		double bestTruAcc = 0;
		double bestScore = totalScore;
		
		////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////
		
		//System.out.println(pred.toString());
		
		SearchResult finalRe = new SearchResult();
		finalRe.accuarcy = bestTruAcc;
		finalRe.predScore = bestScore;
		finalRe.predState = new SearchState(pred);
		finalRe.trajectories = new ArrayList<SearchTrajectory>();
		finalRe.e_trajs = null;
		finalRe.bestRank = 0;
		return finalRe;
	}
}
