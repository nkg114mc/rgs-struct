package elearning;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearning.einfer.ELinearSearchInferencer;
import elearning.einfer.ESearchInferencer;
import general.AbstractFeaturizer;
import general.AbstractInstance;
import general.AbstractOutput;
import general.FactorGraphBuilder.FactorGraphType;
import init.RandomStateGenerator;
import search.GreedySearcher;
import search.SearchResult;
import search.SearchState;
import search.SearchTrajectory;
import search.ZobristKeys;
import sequence.hw.HwInstance;

public class LowLevelCostLearning {
	
	public static enum StopType {
		PERC_STOP, ITER_STOP
	}


	//WeightVector cost_weight;
	GreedySearcher gsearcher;
	
	AbstractFeaturizer efeaturizer;
	AbstractRegressionLearner regressionTrainer;
	
	int iteration;
	boolean applyInstWght;
	StopType howToStop;
	
	static double unchangedRateThresh;
	
	
	public LowLevelCostLearning(GreedySearcher gschr,
	
								AbstractFeaturizer efr,
								AbstractRegressionLearner regrTr,
	
								int iter,
								boolean useInstWght,
								StopType hstop,
								double unchgRt) {
		
		//cost_weight = cwght;
		gsearcher = gschr;
		
		efeaturizer = efr;
		regressionTrainer = regrTr;
		
		iteration = iter;
		applyInstWght = useInstWght;
		howToStop = hstop;
		
		unchangedRateThresh = unchgRt;
		assert (unchangedRateThresh >= 0 && unchangedRateThresh <= 1);
		
	}
	
	public EInferencer trainEvalFunc(List<HwInstance> instances, WeightVector cwght) { // no static training
		
		RandomStateGenerator randomGenr = gsearcher.getInitGenerator();
		FactorGraphType fgt = gsearcher.getFactorGraphType();
		ZobristKeys abkeys = gsearcher.getZobKeys();
		
		EInferencer e_model = LowLevelCostLearning.learnEFunction(randomGenr, instances, efeaturizer,
																   fgt,
																   abkeys,
																   cwght,//cost_weight, 
				                                                   gsearcher, 
                                                                   regressionTrainer,
                                                   				   iteration,
                                                   				   applyInstWght,
                                                   				   howToStop);
		
		return e_model;
	}
	
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////

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
			StopType howStop) {
		
		final double learnRate = 0.1;
		final double lambda = 1.0E-3;

		//UniformRndGenerator randomGenr = new UniformRndGenerator(new Random());
		GreedySearcher esearcher = new GreedySearcher(fgt, efeaturizer, 1, gsearcher.getActionGener(), randomGenr,  gsearcher.getLossFunc(), abkeys);

		
		PerceptronUpdater pupdater = new PerceptronUpdater(efeaturizer.getFeatLen(), learnRate, lambda);
		pupdater.reset();
		
		WeightVector e_weight = null;// new WeightVector(cost_weight.getLength()); // init e_weight with all zero
		//ArrayList<WeightVector> e_ws_iter = new ArrayList<WeightVector>();


		////////////reference depth ///////////////////////////////////////
		int orginalDepthSum = 0;
		for (AbstractInstance ainst : instances) {
			AbstractOutput y_start = (randomGenr.generateSingleRandomInitState(ainst).structOutput);
			SearchResult result0 = gsearcher.runSearchGivenInitState(cost_weight, ainst, y_start, null, false);
			SearchTrajectory traj = result0.getUniqueTraj();
			orginalDepthSum += traj.getStateList().size();
		}
		System.out.println("Original sum depth: " + orginalDepthSum);
		/////////////////////////////////////////////////////////////////////

		if (howStop == StopType.PERC_STOP) { // default is iteration stopping
			int newIter = 10;//Integer.MAX_VALUE - 1;
			System.out.println("Change iteration from " + iteration + " to " + newIter);
			iteration = newIter;
		}


		List<AbstractOutput> currCResults = null;//new ArrayList<AbstractOutput>();
		List<AbstractOutput> lastCResults = null;
		
		for (int iter = 0; iter < iteration; iter++) {
			
			e_weight = doubleArrtoWght(pupdater.getCurAvg());

			ArrayList<StarHatPair> regrDataIter = new ArrayList<StarHatPair>();

			System.out.println("E-Learning iteration " + iter + ":");


			currCResults = new ArrayList<AbstractOutput>();
			
			int sumSteps = 0;
			for (AbstractInstance ainst : instances) {

				// E-function inference to find y_end

				AbstractOutput y_start = (randomGenr.generateSingleRandomInitState(ainst).structOutput);

				SearchResult endResult = esearcher.runSearchGivenInitState(e_weight, ainst, y_start, null, false);
				AbstractOutput y_end = endResult.predState.structOutput;


				////////////////////////////////////

				// Regular inference with C-function
				AbstractOutput y_star = ainst.getGoldOutput();

				//SearchResult cResult = gsearcher.runSearchGivenInitState(cost_weight, ainst, y_end, null, false);
				SearchResult cResult = gsearcher.runSearchGivenInitState(cost_weight, ainst, y_start, null, false);
				AbstractOutput y_real = cResult.predState.structOutput;
				currCResults.add(y_real);
				
				//if (endResult.predScore != cResult.predScore) { // update W_e or 
					
				
				SearchTrajectory traj = cResult.getUniqueTraj(); // should be just one
				List<SearchState> states = traj.getStateList();
				sumSteps += states.size();
				
				HashMap<Integer, Double> phi_end = efeaturizer.featurize(ainst, y_end); // hat
				HashMap<Integer, Double> phi_real = efeaturizer.featurize(ainst, y_real); // real
				HashMap<Integer, Double> phi_star = efeaturizer.featurize(ainst, y_star); // star

				// StarHatPair dp = new StarHatPair(phi_end,phi_real);
				StarHatPair dp = new StarHatPair(phi_end,phi_star);
				regrDataIter.add(dp);

				//}
			}

			// train to get new e_function
			pupdater.runQuickPerceptron(regrDataIter, efeaturizer.getFeatLen());
			
			System.out.println("Iter sum steps = " + sumSteps);
			
			if (iter > 0) {
				double noChgRate = checkInferenceResultChange(currCResults,lastCResults);
				System.out.println("iter = "+iter+" UnChanged rate = " + noChgRate);
			}
			
			if (howStop == StopType.PERC_STOP) { // default is iteration stopping
				if (iter > 0) {
					double noChgRate = checkInferenceResultChange(currCResults,lastCResults);
					//System.out.println("UnChanged rate = " + noChgRate);
					if (noChgRate >= unchangedRateThresh) {
						System.out.println("" + noChgRate + " >= " + unchangedRateThresh);
						break;// stop iteration
					}
				}
			}
			
			lastCResults = currCResults;
		}

		e_weight = doubleArrtoWght(pupdater.getCurAvg());
		
		EInferencer einfr = new ELinearSearchInferencer(esearcher, e_weight);
		return einfr;
		//return e_weight;
	}
	

	public static WeightVector doubleArrtoWght(double[] arr) {
		WeightVector wv = new WeightVector(arr.length);
		for (int i = 0; i < arr.length; i++) {
			assert (!Double.isNaN(arr[i]));
			assert (!Double.isInfinite(arr[i]));
			wv.setElement(i, (float)arr[i]);
		}
		return wv;
	}
	
	public static double checkInferenceResultChange(List<AbstractOutput> thisIterResults,
			                                        List<AbstractOutput> lastIterResults) {

		double noChangeCnt = 0;
		double totalCnt = 0;
		
		if (thisIterResults.size() == lastIterResults.size()) {
			int n = thisIterResults.size();
			for (int i = 0; i < n; i++) {
				if (thisIterResults.get(i).isEqual(lastIterResults.get(i))) {
					noChangeCnt++;
				}
				totalCnt++;
			}
		} else {
			throw new RuntimeException("Instance inference result change between two iteration, instance number unequal:" + thisIterResults.size() + "!=" + lastIterResults.size());
		}
		
		double noChangeRate = noChangeCnt / totalCnt;
		return noChangeRate;
	}

}
