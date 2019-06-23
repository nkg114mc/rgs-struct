package elearning;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearnnew.SamplingELearning;
import general.AbstractFeaturizer;
import general.AbstractInstance;
import general.AbstractLossFunction;
import general.AbstractOutput;
import general.FactorGraphBuilder.FactorGraphType;
import init.RandomStateGenerator;
import search.GreedySearcher;
import search.SearchResult;
import search.SearchState;
import search.SearchTrajectory;
import search.ZobristKeys;
import search.loss.GoldPredPair;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;

public class EFunctionLearning {
	
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
			
			List<HwInstance> tstInsts) {


		//EInferencer einfr = ERegressionLearning.learnEFunction(randomGenr, instances,  efeaturizer, fgt, abkeys,  cost_weight,gsearcher,  regressionTrainer,iteration,applyInstWght);
		//EInferencer einfr = LowLevelCostLearning.learnEFunction(randomGenr, instances,  efeaturizer, fgt, abkeys,  cost_weight,gsearcher,  regressionTrainer,iteration,applyInstWght, StopType.ITER_STOP);
		
		//EInferencer einfr = AlterElearning.learnEFunction(randomGenr, instances,  efeaturizer, fgt, abkeys,  cost_weight,gsearcher,  regressionTrainer,iteration,applyInstWght, StopType.ITER_STOP,null);
		EInferencer einfr = SamplingELearning.learnEFunction(randomGenr, tstInsts,  efeaturizer, fgt, abkeys,  cost_weight,gsearcher,  regressionTrainer,iteration,applyInstWght);
		
		return einfr;
	}
	
	
	// test evaluation function speedup

	public static void testEvaluationSpeedupSeqLabeling(List<HwInstance> tstInsts,
                                                        WeightVector cost_weight, 
                                                        //WeightVector e_weight, 
                                                        GreedySearcher gsearcher,
                                                        EInferencer einfr,
                                                        int restartNum,
														String namePrefix) {

		TrajectoryPloter ploter = new TrajectoryPloter(restartNum, 32767, gsearcher.getLossFunc());
		
		List<GoldPredPair> results = new ArrayList<GoldPredPair>();
		
		
		int[] bestRank = new int[restartNum];
		Arrays.fill(bestRank, 0);

		for (int i = 0; i < tstInsts.size(); i++) {
			HwInstance inst = tstInsts.get(i);
			HwOutput gold = inst.getGoldOutput();
			SearchResult infrRe = gsearcher.runSearchWithRestarts(cost_weight, einfr, restartNum, inst, gold, false);
			checkResult(infrRe);
			SamplingELearning.checkRestartScores(infrRe);
			
			int bestIdx = infrRe.bestRank;
			bestRank[bestIdx]++;
			
			GoldPredPair re = new GoldPredPair((AbstractInstance)inst, (AbstractOutput)gold, (AbstractOutput)(infrRe.predState.structOutput));
			results.add(re);
			
			if (infrRe.e_trajs != null) {
				// replace the predict score (evaluation score) to cost score
				recomputeCostForETraj(infrRe.e_trajs, inst, gsearcher, cost_weight);
			}
			
			//////////////////////////////////
			if (gsearcher.getLossFunc().getClass().getSimpleName().equals("SearchLossHamming")) {
				//System.out.println("Normalize accuracy...");
				normalizeHammingLoss(infrRe, inst, gsearcher.getLossFunc());
			}
			
			ploter.addOneResult(infrRe);
		}
		

		// local optima analysis
		SamplingELearning.printLocalOptimaMeasure(bestRank, tstInsts.size(), restartNum);
		

		// naming prefix
		String useInstWght = "noinstwght";
		//if (einfr != null) {
		//	if (einfr.considerInstWeight) {
		//		useInstWght = "withinstwght";
		//	}
		//}
		String useEfunc = "noEfunc";
		if (einfr != null) {
			useEfunc = "withEfunc";
		}
		String initTyp = gsearcher.getInitGenerName();
		
		String finalPrefix = namePrefix + "_" + initTyp +  "_" + useInstWght + "_" + useEfunc + "_" + "restr" + String.valueOf(gsearcher.randInitSize);
		ploter.plotToFile(finalPrefix);
		
		AbstractLossFunction lossfunc = gsearcher.getLossFunc();
		double lossFuncAcc = lossfunc.computeMacroFromResults(results);
		System.out.println("LossFunction computed accuracy = " + lossFuncAcc);
		System.out.println("");
	}
	
	public static void checkResult(SearchResult re) {
		for (SearchTrajectory traj : re.trajectories) {
			List<SearchState> states = traj.getStateList();
			for (int d = 0; d < states.size(); d++) {
				SearchState state = states.get(d);
				assert (state.trueAccFrac != null);
			}
		}
	}
	
	public static void recomputeCostForETraj(List<SearchTrajectory> e_trajs, HwInstance inst, GreedySearcher gsearcher, WeightVector cost_weight) {
		
		//System.out.println("Turn evaluation to cost...");
		
		for (SearchTrajectory e_traj : e_trajs) {
			List<SearchState> states = e_traj.getStateList();
			for (int d = 0; d < states.size(); d++) {
				SearchState state = states.get(d);
				double cost = gsearcher.scoring(cost_weight, (IInstance)inst, (IStructure)state.structOutput);
				//float eval = state.score;
				state.score = (float) cost;
				//System.out.println("Replace eval to cost:" + eval + " --> " + cost);
			}
		}
		
	}
	
	public static void normalizeHammingLoss(SearchResult re, AbstractInstance inst, AbstractLossFunction lossFunc) {
		
		double den = lossFunc.getAccuracyFullScore(inst);
		
		List<SearchTrajectory> c_trajs = re.trajectories;
		for (int i = 0; i < c_trajs.size(); i++) {
			SearchTrajectory traj = c_trajs.get(i);
			List<SearchState> states = traj.getStateList();
			for (int d = 0; d < states.size(); d++) {
				states.get(d).trueAcc /= den;
			}
		}
	}

}
