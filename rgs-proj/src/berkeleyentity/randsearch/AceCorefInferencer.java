package berkeleyentity.randsearch;

import edu.illinois.cs.cogcomp.sl.core.AbstractInferenceSolver;
import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.latentsvm.AbstractLatentInferenceSolver;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearning.EInferencer;
import general.AbstractInstance;
import general.AbstractOutput;
import search.GreedySearcher;
import search.SearchResult;
import sequence.hw.HwOutput;
import sequence.hw.HwSearchInferencer;

public class AceCorefInferencer extends AbstractLatentInferenceSolver {
	
	private static final long serialVersionUID = 4051169362098835972L;
	
	GreedySearcher gsearcher;
	GreedySearcher gold_searcher;

	public AceCorefInferencer(GreedySearcher gschr, GreedySearcher goldschr) {
		gsearcher = gschr;
		gold_searcher = goldschr;
	}


	@Override
	public IStructure getBestStructure(WeightVector wv, IInstance input) throws Exception {
		SearchResult result = gsearcher.runSearchWithRestarts(wv, gsearcher.getEvalInferencer(), (AbstractInstance)input, null, false);
		AbstractOutput resultOutput = result.predState.structOutput;
		return ((IStructure)(resultOutput));
	}
	@Override
	public float getLoss(IInstance ins, IStructure goldStructure,  IStructure structure) {
		return gsearcher.getLossFloat((AbstractInstance)ins, goldStructure, structure);
	}

	@Override
	public IStructure getLossAugmentedBestStructure(WeightVector wv, IInstance input, IStructure gold) throws Exception {
		SearchResult result = gsearcher.runSearchWithRestarts(wv, gsearcher.getEvalInferencer(), (AbstractInstance)input, (AbstractOutput)gold, true);
		AbstractOutput resultOutput = result.predState.structOutput;
		return ((IStructure)(resultOutput));
	}

	// result more-informative result rather than just output
	public SearchResult runSearchInference(WeightVector wv, EInferencer eifr, IInstance input, IStructure gold) {
		SearchResult result = gsearcher.runSearchWithRestarts(wv, eifr, (AbstractInstance)input, (AbstractOutput)gold, false);
		return (result);
	}

	public SearchResult runSearchInferenceMaybeLossAug(WeightVector wv, EInferencer eifr, IInstance input, IStructure gold, boolean losAug) {
		SearchResult result = gsearcher.runSearchWithRestarts(wv, eifr, (AbstractInstance)input, (AbstractOutput)gold, losAug);
		return (result);
	}


	@Override
	public AbstractInferenceSolver clone() {
		HwSearchInferencer cp = new  HwSearchInferencer(gsearcher);
		return cp;
	}

	public GreedySearcher getSearcher() {
		return gsearcher;
	}

	@Override
	public IStructure getBestLatentStructure(WeightVector weight, IInstance ins, IStructure gold) throws Exception {
		
		SearchResult result = gold_searcher.runSearchWithRestarts(weight, gold_searcher.getEvalInferencer(), (AbstractInstance)ins, null, false);
		AbstractOutput resultOutput = result.predState.structOutput;
		return ((IStructure)(resultOutput));

	}

	
	///////////// for system test only
	
	public IStructure getRndPerfect(AceCorefInstance ins) {
		
		String[] lbs = new String[ins.size()];
		HwOutput structOut = new HwOutput(ins.size(), lbs);
		for (int i = 0; i < ins.size(); i++) {
			int[] goldDomain = ins.getGoldDomainGivenIndex(i);
			//int[] goldDomain = ins.getDomainGivenIndex(i);
			if (goldDomain.length == 0) {
				structOut.setOutput(i, i);
			} else {
				structOut.setOutput(i, goldDomain[0]);
			}
			//System.out.println("output(" + i + ") = " + structOut.getOutput(i) + " " + ins.getAnteIndex(i, structOut.getOutput(i)));
		}
		
		return ((IStructure)(structOut));
	}
}
