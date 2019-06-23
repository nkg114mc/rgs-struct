package essvm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.illinois.cs.cogcomp.sl.core.AbstractFeatureGenerator;
import edu.illinois.cs.cogcomp.sl.core.AbstractInferenceSolver;
import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.core.SLParameters;
import edu.illinois.cs.cogcomp.sl.core.SLProblem;
import edu.illinois.cs.cogcomp.sl.learner.Learner;
import edu.illinois.cs.cogcomp.sl.util.IFeatureVector;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearning.EInferencer;
import elearning.LowLevelCostLearning;
import sequence.hw.HwSearchInferencer;

public class SSVMGurobiSolverWithEval extends Learner {

	private static Logger log = LoggerFactory.getLogger(SSVMGurobiSolverWithEval.class);

	public static Random random = new Random();

	protected GurobiQpSolverWrapper gurobiQpSolver = new GurobiQpSolverWrapper();
	
	protected HwSearchInferencer inference;
	protected AbstractFeatureGenerator featureGenerator;
	protected int epochUpdateCount;
	
	protected LowLevelCostLearning evalLearner;
	
	//// About E-learning

	/**
	 * @param infSolver 
	 * @param featureGenerator
	 */
	public SSVMGurobiSolverWithEval(AbstractInferenceSolver infSolver, AbstractFeatureGenerator fg, SLParameters params, LowLevelCostLearning eTrainer){
		super(infSolver, fg, params);
		this.featureGenerator = fg;
		if (!infSolver.getClass().getSimpleName().equals("HwSearchInferencer")) {
			throw new RuntimeException("SearchInferencer Required!");
		}
		this.inference = (HwSearchInferencer)infSolver;
		this.evalLearner = eTrainer;
	}

	/**
	 * To train with the default choice(zero vector) of initial weight vector. 
	 * Often this suffices.
	 * @param problem	The structured problem on which the perceptron should be trained
	 * @return w	The weight vector learnt from the training
	 * @throws Exception
	 */
	public WeightVector trainLocal(SLProblem problem, SLParameters params) throws Exception {

		WeightVector init = new WeightVector(10000);
		if (params.TOTAL_NUMBER_FEATURE > 0) {
			init = new WeightVector(params.TOTAL_NUMBER_FEATURE);
			init.setExtendable(false);
		}

		return trainLocal(problem, params, init);
	}
	/**
	 * To train with a custom (possibly non-zero) initial weight vector
	 * @param problem	The structured problem on which the perceptron should be trained
	 * @param init	The initial weightvector to be used
	 * @return w	The weight vector learnt from the training data	
	 * @throws Exception
	 */
	public WeightVector trainLocal(SLProblem problem, SLParameters params, WeightVector init)
			throws Exception {

		log.info("Starting Structured Perceptron learner");

		long start = System.currentTimeMillis();

		WeightVector w = init;

		int epoch = 0;
		boolean done = false;
		
		SSVMWorkSet workset = new SSVMWorkSet();;

		int count = 1;

		while (!done) {

			if (epoch % params.PROGRESS_REPORT_ITER == 0) {
				log.info("Starting epoch {}", epoch);
				if(f!=null)
					f.run(w, this.inference);
			}

			count = doOneIteration(w, null, problem, epoch, count, params, workset);
			
			
			double sumls = testOnTrainSumLoss(w, problem);
			System.out.println("SumLoss = " + sumls);

			if (epoch % params.PROGRESS_REPORT_ITER == 0){
				log.info("End of epoch {}. {} updates made", epoch,
						epochUpdateCount);
			}

			epoch++;
			done = !reachedStoppingCriterion(w, epoch, params);
			if (params.PROGRESS_REPORT_ITER > 0 && (epoch+1) % params.PROGRESS_REPORT_ITER == 0 && this.f != null)
				f.run(w, inference);

		}

		long end = System.currentTimeMillis();

		log.info("Learning complete. Took {}s", "" + (end - start) * 1.0
				/ 1000);

		return w;

	}
	
	/**
	 * Checks if stopping criterion has been met. We will stop if either no mistakes were made 
	 * during this iteration, or the maximum number of passes over the training data has been made.
	 * @param w
	 * @param epoch
	 * @return
	 */
	protected boolean reachedStoppingCriterion(WeightVector w, int epoch, SLParameters params) {

		//if (epochUpdateCount == 0) {
		//	log.info("No errors made. Stopping outer loop because learning is complete!");
		//	return false;
		//}
		return epoch < params.MAX_NUM_ITER;
	}
	/**
	 * Performs one pass over the entire training data.
	 * @param w		The weight vector to use during this iteration
	 * @param problem
	 * @param epoch
	 * @param count
	 * @param params
	 * @return
	 * @throws Exception
	 */
	protected int doOneIteration(WeightVector w, EInferencer einferencer, SLProblem problem, 
			                     int epoch, int count, SLParameters params, SSVMWorkSet wkset) throws Exception {
		
		int numExamples = problem.size();
		double epsilon = 0;
		
		epochUpdateCount = 0;
		problem.shuffle(random); // shuffle your training data after every iteration
		
		// Stage 1: Do Inference
		SSVMWorkSet addedWorkset = new SSVMWorkSet();
		
		for (int exampleId = 0; exampleId < numExamples; exampleId++) {
			IInstance example = problem.instanceList.get(exampleId);	// the input "x"
			IStructure gold = problem.goldStructureList.get(exampleId);	// the gold output structure "y"

			IStructure prediction = null;
			double loss = 0;
			prediction = this.inference.getLossAugmentedBestStructure(w, example, gold);	// the predicted structure
			//prediction = (IStructure) this.inference.runSearchInferenceMaybeLossAug(w, einferencer, example, gold, true).predState.structOutput;	// the predicted structure
			loss = this.inference.getLoss(example, gold, prediction);	// we will update if the loss is non-zero for this example

			assert prediction != null;

			IFeatureVector goldFeatures = featureGenerator.getFeatureVector(example, gold); 
			IFeatureVector predictedFeatures = featureGenerator.getFeatureVector(example, prediction);
			IFeatureVector GoldMinusPred = goldFeatures.difference(predictedFeatures);
			
			if (loss > 0) {
				
				if (!wkset.existOutput(example, prediction)) {
					// insert workset
					SSVMConstraint constraint = new SSVMConstraint(prediction.hashCode());
					constraint.goldMinusPredFeatures = GoldMinusPred;
					constraint.loss = loss;
					
					addedWorkset.insert(example, prediction, constraint);
					wkset.insert(example, prediction, constraint);
					
					//doSolveQP(problem, addedWorkset, w);
				}
			}
			
		}

		
		System.out.println("Add " + addedWorkset.computeSize() + " to workset!");

		// Stage 2: Optimization
		doSolveQP(problem, addedWorkset, w);
		
		
		// Stage 0
		// E-learning
		//EInferencer eInfRe = evalLearner.trainEvalFunc(List<AbstractInstance> instances);
		

		return count;
	}
	
	public double testOnTrainSumLoss(WeightVector w, SLProblem problem) {
		try {

			double sumLoss = 0;
			for (int exampleId = 0; exampleId < problem.size(); exampleId++) {
				IInstance example = problem.instanceList.get(exampleId);	// the input "x"
				IStructure gold = problem.goldStructureList.get(exampleId);	// the gold output structure "y"
				IStructure prediction = this.inference.getBestStructure(w, example);
				// the predicted structure
				double loss = (double)this.inference.getLoss(example, gold, prediction);	// we will update if the loss is non-zero for this example
				sumLoss += loss;
			}

			return sumLoss;

		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return -1;
	}
	
/*
	private SSVMConstraint getOrCreate(IInstance inst, HashMap<IInstance, SSVMConstraint> histConstrs) {
		SSVMConstraint constraint = histConstrs.get(inst);
		if (constraint == null) {
			SSVMConstraint newConstr = new SSVMConstraint();
			histConstrs.put(inst, newConstr);
			constraint = newConstr;
		}
		return constraint;
	}
*/
	
	private void doSolveQP(SLProblem problem, SSVMWorkSet wkset, WeightVector w) {
		gurobiQpSolver.doSolveQP(problem, wkset, w);
	}
	
	/**
	 * 
	 * @param epoch
	 * @param count
	 * @return
	 */
	protected double getLearningRate(int epoch, int count, SLParameters params) {
		if (params.DECAY_LEARNING_RATE)
			return params.LEARNING_RATE / count;
		else
			return params.LEARNING_RATE;
	}
	protected ProgressReportFunction f;

	public void runWhenReportingProgress(ProgressReportFunction f) {
		this.f = f;
	}

	@Override
	public WeightVector train(SLProblem arg0) throws Exception {
		return trainLocal(arg0, parameters);
	}

	@Override
	public WeightVector train(SLProblem arg0, WeightVector arg1) throws Exception {
		return trainLocal(arg0, parameters, arg1);
	}
}
