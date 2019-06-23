package elearnnew;

import java.util.List;

import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearning.AbstractRegressionLearner;
import elearning.EInferencer;
import elearning.einfer.ESamplingInferencer;
import elearning.einfer.SearchStateScoringFunction;
import elearning.einfer.XgbSingleInstanceOnlineScorer;
import general.AbstractFeaturizer;
import general.FactorGraphBuilder.FactorGraphType;
import init.RandomStateGenerator;
import search.GreedySearcher;
import search.ZobristKeys;
import sequence.hw.HwInstance;

public class DummyELearning {

	public static EInferencer learnEFunction(RandomStateGenerator randomGenr,
			List<HwInstance> instances,
			AbstractFeaturizer efeaturizer,
			FactorGraphType fgt,
			ZobristKeys abkeys,

			WeightVector cost_weight, 
			GreedySearcher gsearcher, 

			AbstractRegressionLearner regressionTrainer,
			int iteration,
			boolean applyInstWght,
			
			List<HwInstance> tstInsts){
		
		SearchStateScoringFunction emd = new XgbSingleInstanceOnlineScorer(randomGenr, cost_weight, gsearcher, regressionTrainer, iteration);
		
		EInferencer einfr = new ESamplingInferencer(randomGenr, cost_weight, gsearcher.getFeaturizer(), emd, efeaturizer, 100);
		
		return einfr;
	}
	

	
}
