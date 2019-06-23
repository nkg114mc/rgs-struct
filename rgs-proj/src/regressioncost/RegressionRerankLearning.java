package regressioncost;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.core.SLModel;
import edu.illinois.cs.cogcomp.sl.core.SLProblem;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearning.AbstractRegressionLearner;
import elearning.RegressionInstance;
import elearning.einfer.SearchStateScoringFunction;
import experiment.ExperimentResult;
import experiment.TestingAcc;
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
import search.loss.LossScore;
import sequence.hw.HwFeaturizer;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;
import sequence.hw.HwSearchInferencer;

public class RegressionRerankLearning {
	
	public static SearchStateScoringFunction regressionCostRerank(List<HwInstance> instances,
											GreedySearcher searcher,
			                                WeightVector cost_weight, 
			                                int restarts,
			                                AbstractRegressionLearner regressionTrainer) {
		
		RandomStateGenerator initStateGener = searcher.getInitGenerator();
		AbstractFeaturizer featurizer = searcher.getFeaturizer();
		AbstractLossFunction lossfunc = searcher.getLossFunc();

		ArrayList<RegressionInstance> regrDataIter = new ArrayList<RegressionInstance>();


		for (AbstractInstance ainst : instances) {
			
			HwOutput goutput = ainst.getGoldOutput();
			
			HashSet<SearchState> initStateSets = initStateGener.generateRandomInitState(ainst, restarts);
			
			for (SearchState iniState : initStateSets) {
				
				AbstractOutput y_start = iniState.structOutput; //(initStateGener.generateSingleRandomInitState(ainst).structOutput);
				SearchResult cResult = searcher.runSearchGivenInitState(cost_weight, ainst, y_start, null, false);

				// a regression instance: y
				SearchTrajectory traj = cResult.getUniqueTraj(); // should be just one
				List<SearchState> states = traj.getStateList();
				for (int d =  0; d < states.size(); d++) {

					SearchState dstate = states.get(d);

					// one regression training data point
					HashMap<Integer, Double> phi = featurizer.featurize(ainst, dstate.structOutput);
					double lost_value = lossfunc.computeZeroOneAcc(ainst, (IStructure)goutput, (IStructure)dstate.structOutput).getVal();

					// aggregate data
					RegressionInstance dp = new RegressionInstance(phi, lost_value);
					regrDataIter.add(dp);

				}
			}
		}
		
		// run regression learning
		SearchStateScoringFunction regFunc = regressionTrainer.regressionTrain(regrDataIter, -1, -1);
		return regFunc;
		
	}

	public static HwOutput predictWithRegressionCost(AbstractInstance instance,
													 GreedySearcher searcher,
													 WeightVector cost_weight,
													 int restarts,
													 SearchStateScoringFunction regFunc,
													 ExperimentResult re) { // return
		
		RandomStateGenerator initStateGener = searcher.getInitGenerator();
		AbstractFeaturizer featurizer = searcher.getFeaturizer();
		AbstractLossFunction lossfunc = searcher.getLossFunc();
		
		HashSet<SearchState> initStateSets = initStateGener.generateRandomInitState(instance, restarts);
		
		AbstractOutput bestOutput = null;
		double bestRegSc = -Double.MAX_VALUE;
		AbstractOutput bestLossOutput = null;
		double bestTrueAcc = -Double.MAX_VALUE;
		
		HwOutput goutput = instance.getGoldOutput();
		
		for (SearchState iniState : initStateSets) {
			
			AbstractOutput y_start = iniState.structOutput; 
			SearchResult cResult = searcher.runSearchGivenInitState(cost_weight, instance, y_start, null, false);

			// a regression instance: y
			SearchTrajectory traj = cResult.getUniqueTraj(); // should be just one
			List<SearchState> states = traj.getStateList();
			for (int d =  0; d < states.size(); d++) {

				SearchState dstate = states.get(d);

				// one regression training data point
				HashMap<Integer, Double> phi = featurizer.featurize(instance, dstate.structOutput);
				double rerankNewScore = regFunc.getScoring(instance, phi);
				double trueAcc = lossfunc.computeZeroOneAcc(instance, (IStructure)goutput, (IStructure)dstate.structOutput).getVal();
				if (bestRegSc < rerankNewScore) {
					bestRegSc = rerankNewScore;
					bestOutput = dstate.structOutput;
				}
				if (bestTrueAcc < trueAcc) {
					bestTrueAcc = trueAcc;
					bestLossOutput = dstate.structOutput;
				}
				
			}
		}
		
		assert (bestOutput != null);
		re.bestCostOutput = bestOutput;
		re.bestLossOutput = bestLossOutput;
		
		return (HwOutput)bestOutput;
	}
	
	
	public static ExperimentResult evaluate(SLProblem sp, SLModel model, SearchStateScoringFunction regFunc) throws Exception {
		double total = 0;
		double acc = 0;
		List<LossScore> trueBestLossAccs = new ArrayList<LossScore>();
		
		HwSearchInferencer searchInfr = (HwSearchInferencer)(model.infSolver);
		AbstractLossFunction lossfunc = searchInfr.getSearcher().getLossFunc();
		
		System.err.println("TestRestart = " + searchInfr.getSearcher().randInitSize);
		int restarts = searchInfr.getSearcher().randInitSize;
		
		List<GoldPredPair> results = new ArrayList<GoldPredPair>();
		
		for (int i = 0; i < sp.instanceList.size(); i++) {
			ExperimentResult er = new ExperimentResult();
			HwOutput gold = (HwOutput) sp.goldStructureList.get(i);
			HwOutput prediction = (HwOutput)predictWithRegressionCost((AbstractInstance)(sp.instanceList.get(i)), searchInfr.getSearcher(), model.wv, restarts, regFunc, er);
			for (int j = 0; j < prediction.output.length; j++) {
				total += 1.0;
				if (prediction.output[j] == gold.output[j]){
					acc += 1.0;
				}
			}
			
			LossScore lsc = lossfunc.computeZeroOneAcc((AbstractInstance)(sp.instanceList.get(i)), (IStructure)gold, (IStructure)er.bestLossOutput);
			trueBestLossAccs.add(lsc);
			
			GoldPredPair re = new GoldPredPair((AbstractInstance)(sp.instanceList.get(i)), (AbstractOutput)gold, (AbstractOutput)prediction);
			results.add(re);
			
		}
		
		double totalBestLossAccScore = lossfunc.computeMacro(trueBestLossAccs);
		
		//avgTruAcc = avgTruAcc / total;
		double accuracy = acc / total;
		System.out.println("Accuracy = " + acc + " / " + total + " = " + accuracy);
		System.out.println("Accuracy = " + accuracy);
		System.out.println("Gen Accuracy = " + totalBestLossAccScore);
		
		/*
		double genAcc = avgTruAcc;
		double selAcc = genAcc - accuracy;
		
		if (genAcc < accuracy) {
			throw new RuntimeException("[ERROR]Generation accuracy is less than final output accuracy: " + genAcc + " < " + accuracy);
		}
		
		System.out.println("Generation Acc = " + genAcc);
		System.out.println("Selection AccDown = " + selAcc);
		
		double lossFuncAcc = lossfunc.computeMacroFromResults(results);
		System.out.println("LossFunction computed accuracy = " + lossFuncAcc);
		
		
		ExperimentResult expRslt = new ExperimentResult();
		expRslt.lossName = "";
		expRslt.overallAcc = accuracy;
		expRslt.generationAcc = genAcc;
		expRslt.selectionAcc = selAcc;
		
		
		expRslt.addAcc(new TestingAcc("HammingAcc", expRslt.overallAcc));
		expRslt.addAcc(new TestingAcc("GenerationAcc", expRslt.generationAcc));
		
		return expRslt;*/
		
		return null;
	}
}
