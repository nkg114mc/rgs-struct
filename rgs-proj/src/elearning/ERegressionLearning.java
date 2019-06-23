package elearning;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearning.einfer.ELinearSearchInferencer;
import elearning.einfer.ESearchInferencer;
import elearning.einfer.SearchStateScoringFunction;
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

public class ERegressionLearning {
	
/**

Let us say E(x,y) = W_e \times \Phi_e(x,y), where W_e are the weights and \Phi_e(x,y) is the evaluation function features.

If y_{start) is the starting solution of local search and y_{end} is the ending solution of local search (i.e., local optima), then we want to learn E such that,

E(x, y_{start}) = C(x, y_{end})

This is a simple regression problem:
Input = \Phi_e(x, y_{start})
Output = C(x, y_{end})

We need to learn the weights of E via a regression learner: you can use linear regression or SVM regression from Weka if your code is in Java -- Online learner will be better.

High-level algorithmic pseudo-code will look as follows:

Input:
D = {x, y*}, structured input-output training examples
C(x,y), learned cost function to score candidate structured outputs for a given input

Initialize weights of E
repeat
	For each training example (x, y*) \in D
		// Meta Search (Search for good starting states using the current evaluation function)
		Select a random starting solution y_start and perform local search guided by E.
		Say y_end is the local optima and E(x, y_end) is the predicted cost
		// Base Search from the selected starting point
		Start from y_end and perform local search guided by C to reach the local optimal y_real
		// Generate training data to improve E
		If E(x, y_end) != C(x, y_real) // E is erroneous
			For each output y on the search trajectory from y_end to y_real, generate one regression example as follows:
				\Phi_e(x,y) as input and C(x,y_real) as output
				Update the weights of E based on the regression training example // Online learning
				(You can aggregate the regression examples and Run a batch regression learning algorithm)
			End For
		End If
	End For
until convergence or maximum iterations

In the above algorithm, we can also consider importance weights for regression examples, where importance of regression example \Phi_e(x,y) as input and C(x,y_real) as output is equal to:
L(x, y*, y_real)

	 */
	

	/*
	Recall that there are three parts to the AAAI paper:

	1. Empirical evaluation of RGS on diverse benchmark structured prediction tasks

	2. Learning to improve the speed of RGS inference for a given cost function

	3. Learning to Improve the speed of training cost function with RGS inference

	We want to employ the solution for (2) as a procedure to solve (3).

	Therefore, I suggest we explore the following concrete setting to make progress on (2) and (3). There are other aspects for (2), but let us address them later.

	Setting:

	Given a set of inputs D = {x_i}_{i=1 to m} and a scoring function C(x,y) = W. \Phi(x,y), we want to improve the speed of making predictions using RGS.

	Baseline: For each input x_i from D, perform RGS from random starting solution for R_{max} iterations guided by the scoring function, and select the best local optima over all the iterations.

	Time complexity = number of inputs in D * R_{max} * T, where T is the average number of greedy search steps to converge to local optima.

	Question: Can we get to the same result as above with lower time complexity? Yes.

	 ******
	Speedup Algorithm:

	For each input x \in D
	        Initialize \hat{y} randomly
			y_{start} = random output
	End For

	- Initialize weights of evaluation function E
	- Set of regression examples R = \emptyset

	Repeat

	For each input x \in D

		// Base Search
		- Start from y_start and perform local search guided by C to reach the local optimal y_end

		// Collection of regression examples to improve evaluation function E
		If E(x, y_start) != C(x, y_end) // E is erroneous
				For each output y on the search trajectory from y_start to y_end, generate one regression example as follows:
					- Add the following regression example to R: \Phi_e(x,y) as input and C(x,y_end) as output
				End For
		End If

		// Improve the evaluation function E based on new training data
		- E = Regression-Learner(R) OR E = Online-Learn(\delta R)

		// Meta Search (Search for good starting states using the current evaluation function)
		- Perform greedy search from y_end guided by C to reach the local optimal y_restart

		If y_restart = y_start then
			y_start = random output
		else
			y_start = y_restart
		End If	

		// Update the prediction based on new outputs explored in this iteration
		- Update \hat{y} (the best scoring output) based on all the outputs seen in base search (and meta search?)

	End For

	Until convergence or maximum iterations
	 **********

	You can set maximum iterations as R_{max}/2 for now.

	Convergence criteria can be the predictions for p% inputs don't change in two consecutive iterations. (p=70, 80, 80, 100).

	You can also measure the cumulative performance over all the inputs after each iteration (anytime performance) to see how many iterations does it take to reach the performance of baseline.

	3. Learning to Speedup Training

	Standard SSVM Training:

	 *****
	- Initialize weights of scoring function C
	- Initialize the constraint set S = \emptyset
	Repeat
		For each training example (x, y*) \in D
			- \hat{y} = Inference(input x, scoring-function C)
			- Add constraint C(x, y*) > C(x, \hat{y}) to S
		End For
		- Re-learn cost function from aggregated constraint set S
	Until convergence or maximum iterations
	 *****

	Algorithm to SpeedUp Training:

	 ******
	- Initialize weights of scoring function C
	- Initialize the constraint set S = \emptyset
	Repeat
		// Collect all predictions with the current scoring function C using the above algorithm
		YHAT = SpeedUp-RGS-Inference(D ={x_i}, C)
		For each training example (x, y*) \in D
			Add constraint C(x, y*) > C(x, \hat{y}) to S
		End For
		- Re-learn cost function from aggregated constraint set S
	Until convergence or maximum iterations
	 ******
	 
**/	
	
	public static EInferencer learnEFunction(RandomStateGenerator randomGenr,
											  List<HwInstance> instances,
			                                  AbstractFeaturizer efeaturizer,
			                                  FactorGraphType fgt,
			                                  ZobristKeys abkeys,
			                          
			                                  WeightVector cost_weight, 
			                                  GreedySearcher gsearcher, 
			                          
			                                  AbstractRegressionLearner regressionTrainer,
			                                  int iteration,
			                                  boolean applyInstWght) {
		
		//UniformRndGenerator randomGenr = new UniformRndGenerator(new Random());
		GreedySearcher esearcher = new GreedySearcher(fgt, efeaturizer, 1, gsearcher.getActionGener(), randomGenr,  gsearcher.getLossFunc(), abkeys);
		
		WeightVector e_weight = new WeightVector(efeaturizer.getFeatLen()); // init e_weight with all zero
		ArrayList<WeightVector> e_ws_iter = new ArrayList<WeightVector>();
		

//////////// reference depth ///////////////////////////////////////
		int orginalDepthSum = 0;
		for (AbstractInstance ainst : instances) {
			AbstractOutput y_start = (randomGenr.generateSingleRandomInitState(ainst).structOutput);
			SearchResult result0 = gsearcher.runSearchGivenInitState(cost_weight, ainst, y_start, null, false);
			SearchTrajectory traj = result0.getUniqueTraj();
			orginalDepthSum += traj.getStateList().size();
		}
		System.out.println("Original sum depth: " + orginalDepthSum);
/////////////////////////////////////////////////////////////////////
		
		for (int iter = 0; iter < iteration; iter++) { // iteration is restarts
			
			//Instances regrData = new Instances();
			WeightVector e_aggregated_w = aggregateWeights(e_ws_iter, efeaturizer.getFeatLen());
			ArrayList<RegressionInstance> regrDataIter = new ArrayList<RegressionInstance>();
			
			System.out.println("E-Learning iteration " + iter + ":");
			
			int sumSteps = 0;
			for (AbstractInstance ainst : instances) {

				// E-function inference to find y_end

				AbstractOutput y_start = (randomGenr.generateSingleRandomInitState(ainst).structOutput);

				//SearchResult endResult = esearcher.runSearchGivenInitState(e_weight, ainst, y_start, null, false);
				SearchResult endResult = esearcher.runSearchGivenInitState(e_aggregated_w, ainst, y_start, null, false);
				AbstractOutput y_end = endResult.predState.structOutput;

				
				////////////////////////////////////
				
				// Regular inference with C-function


				SearchResult cResult = gsearcher.runSearchGivenInitState(cost_weight, ainst, y_end, null, false);
				//HwOutput y_real = (HwOutput) cResult.predState.structOutput;

				if (endResult.predScore != cResult.predScore) { // update W_e or 

					// a regression instance: y
					SearchTrajectory traj = cResult.getUniqueTraj(); // should be just one
					List<SearchState> states = traj.getStateList();
					sumSteps += states.size();
					for (int d =  0; d < states.size(); d++) {
						
						SearchState dstate = states.get(d);

						// one regression training data point
						// featurize data point
						HashMap<Integer, Double> phi_e = efeaturizer.featurize(ainst, dstate.structOutput);
						double cost_value = cResult.predScore;// dstate.score;
						// aggregate data
						
						double instWght = 1;
						RegressionInstance dp = new RegressionInstance(phi_e, cost_value);
						regrDataIter.add(dp);
						
						//break;
					}

				}
			}
			
			System.out.println("Iter sum steps = " + sumSteps);

			// train to get new e_function
			SearchStateScoringFunction wghtModel = regressionTrainer.regressionTrain(regrDataIter, efeaturizer.getFeatLen(), iter);	
			e_weight = (WeightVector)wghtModel.getModel();
			
			//e_ws_iter
			e_ws_iter.add(e_weight);
		}
		
		WeightVector e_aggregated_w = aggregateWeights(e_ws_iter,efeaturizer.getFeatLen());
		
		EInferencer einfr = new ELinearSearchInferencer(esearcher, e_aggregated_w);//e_weight);
		return einfr;
		//return e_weight;
	}
	
	public static WeightVector aggregateWeights(List<WeightVector> e_ws_iter, int wlen) {
		
		if (e_ws_iter.size() == 0) {
			WeightVector empt = (new WeightVector(wlen));
			for (int j = 0; j < wlen; j++) {
				empt.setElement(j, 0);
			}
			return empt;
		} else {
			/*
			float[] w_sum = new float[wlen];
			Arrays.fill(w_sum, 0);
			
			for (int i = 0; i < e_ws_iter.size(); i++) {
				for (int j = 0; j < wlen; j++) {
					w_sum[j] += e_ws_iter.get(i).get(j);
				}
			}
			
			WeightVector aggw = (new WeightVector(wlen));
			
			double den = (double)(e_ws_iter.size());
			for (int j = 0; j < wlen; j++) {
				w_sum[j] /= den;
				aggw.setElement(j, w_sum[j]);
			}
			*/
			return e_ws_iter.get(e_ws_iter.size() - 1);
			
			//return aggw;
		}

	}


}
