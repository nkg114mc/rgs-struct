package essvm;

import java.util.ArrayList;
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
import experiment.TrainSpeedResult;
import sequence.hw.HwInstance;
import sequence.hw.HwSearchInferencer;

public class SSVMSGDSolverWithEval extends Learner {

	private static Logger log = LoggerFactory.getLogger(SSVMSGDSolverWithEval.class);

	public static Random random = new Random();


	
	protected HwSearchInferencer inference;
	protected AbstractFeatureGenerator featureGenerator;
	protected int epochUpdateCount;
	
	protected LowLevelCostLearning evalLearner;
	
	//// About E-learning

	/**
	 * @param infSolver 
	 * @param featureGenerator
	 */
	public SSVMSGDSolverWithEval(AbstractInferenceSolver infSolver, AbstractFeatureGenerator fg, SLParameters params, LowLevelCostLearning eTrainer){
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

		int count = 1;

		while (!done) {

			if (epoch % params.PROGRESS_REPORT_ITER == 0) {
				log.info("Starting epoch {}", epoch);
				if(f!=null)
					f.run(w, this.inference);
			}

			count = doOneIteration(w, null, problem, epoch, count, params);

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
	
/*
	public TrainSpeedResult trainWithEvalResultTiming(SLProblem problem, List<HwInstance> instances, SLParameters params, int featLen,  LowLevelCostLearning eTrainer)  {

		
		try {
			
			WeightVector init = (new WeightVector(featLen));

			MyTimeCounter timtCntr = new MyTimeCounter("train");
			TrainSpeedResult trresult = new TrainSpeedResult();

			evalLearner = eTrainer;

			///////////////////////////////////////////////////

			log.info("Starting Structured Perceptron learner");

			long start = System.currentTimeMillis();
			timtCntr.start();

			WeightVector w = init;
			EInferencer einfr = null;

			int epoch = 0;
			boolean done = false;

			int count = 1;

			while (!done) {
				
				double norm2 = w.getSquareL2Norm();

				if (epoch % params.PROGRESS_REPORT_ITER == 0) {
					log.info("Starting epoch {}", epoch);
					if(f!=null)
						f.run(w, this.inference);
				}


				count = doOneIteration(w, einfr, problem, epoch, count, params);

				if (eTrainer != null) {
					einfr = eTrainer.trainEvalFunc(instances, w); // update the evaluation function!
				}


				if (epoch % params.PROGRESS_REPORT_ITER == 0){
					log.info("End of epoch {}. {} updates made", epoch, epochUpdateCount);
				}

				epoch++;
				done = !reachedStoppingCriterion(w, epoch, params);
				if (params.PROGRESS_REPORT_ITER > 0 && (epoch+1) % params.PROGRESS_REPORT_ITER == 0 && this.f != null)
					f.run(w, inference);
				
				double norm22 = w.getSquareL2Norm();
				
				double ndiff = norm22 - norm2;
				//System.out.println(norm22 +" - "+ norm2 + " = " + ndiff);

			}

			timtCntr.end();
			long end = System.currentTimeMillis();

			double timeDur = (end - start) * 1.0 / 1000;
			log.info("Learning complete. Took {}s", "" + timeDur);
			log.info("Timer count: " + timtCntr.getSecondCount() + " seconds.");


			////////////////////////

			trresult.timeConsum = timeDur;//timtCntr.getSecondCount();
			trresult.c_weight = w;
			
			//System.out.println(w.toString());

			// last
			if (eTrainer != null) {
				trresult.einfr = eTrainer.trainEvalFunc(instances, w);
			}

			return trresult;

		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}
*/
	/**
	 * Checks if stopping criterion has been met. We will stop if either no mistakes were made 
	 * during this iteration, or the maximum number of passes over the training data has been made.
	 * @param w
	 * @param epoch
	 * @return
	 */
	protected boolean reachedStoppingCriterion(WeightVector w, int epoch, SLParameters params) {

		if (epochUpdateCount == 0) {
			log.info("No errors made. Stopping outer loop because learning is complete!");
			return false;
		}
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
			                     int epoch, int count, SLParameters params) throws Exception {
		int numExamples = problem.size();

		epochUpdateCount = 0;
		problem.shuffle(random); // shuffle your training data after every iteration

		
		// Stage 1
		// Do Inference
		ArrayList<SSVMUpdateItem> cachedUpdates = new ArrayList<SSVMUpdateItem>();
		
		for (int exampleId = 0; exampleId < numExamples; exampleId++) {
			IInstance example = problem.instanceList.get(exampleId);	// the input "x"
			IStructure gold = problem.goldStructureList.get(exampleId);	// the gold output structure "y"

			IStructure prediction = null;
			double loss = 0;
			//prediction = this.inference.getLossAugmentedBestStructure(w, example, gold);	// the predicted structure
			prediction = (IStructure) this.inference.runSearchInferenceMaybeLossAug(w, einferencer, example, gold, true).predState.structOutput;	// the predicted structure
			loss = this.inference.getLoss(example, gold, prediction);	// we will update if the loss is non-zero for this example

			assert prediction != null;

			IFeatureVector goldFeatures = featureGenerator.getFeatureVector(example, gold); 
			IFeatureVector predictedFeatures = featureGenerator.getFeatureVector(example, prediction);
			IFeatureVector update = goldFeatures.difference(predictedFeatures);
			
			
			SSVMUpdateItem upd = new SSVMUpdateItem();
			upd.loss = loss;
			upd.predDiff = w.dotProduct(update);//Double.NEGATIVE_INFINITY;
			upd.goldFeatures = goldFeatures; 
			upd.predictedFeatures = predictedFeatures;
			upd.update = update;
			cachedUpdates.add(upd);
			
			/*
			double loss_term = 1.0;//loss - w.dotProduct(update);
			//System.out.println(loss_term);
			
			double learningRate = getLearningRate(epoch, count, params);
			w.scale(1.0f-learningRate);
			double sclr = 2*learningRate*params.C_FOR_STRUCTURE*loss_term;
			w.addSparseFeatureVector(update, sclr);
			
			epochUpdateCount++;

			count++;
			*/
		}
		
		
		// Stage 2
		// Update weight
		for (SSVMUpdateItem upd : cachedUpdates) {
			
			IFeatureVector updFeat = upd.update;
			
			double loss_term = 1.0E-3;//loss - w.dotProduct(update);
			//double loss_term = upd.loss - upd.predDiff;//w.dotProduct(update);
			//System.out.println(loss_term);
			
			double learningRate = getLearningRate(epoch, count, params);
			w.scale(1.0f-learningRate);
			double sclr = 2*learningRate*params.C_FOR_STRUCTURE*loss_term;
			w.addSparseFeatureVector(updFeat, sclr);
			
			epochUpdateCount++;

			count++;
		}
		
		

		// Stage 0
		// E-learning
		//EInferencer eInfRe = evalLearner.trainEvalFunc(List<AbstractInstance> instances);
		

		return count;
	}
	
	private WeightVector doSolveQP() {
		return null;
	}
	
	/*
	private double normalizeLossTerm(double loss_term) {
		if (loss_term > 1E-3) {
			loss_term = 1#-;
		} else if (loss_term <  ) {
			
		} else {
			return loss_term;
		}
	}
	*/
	
/*
	protected int doOneIteration(WeightVector w,
			SLProblem problem, int epoch, int count, SLParameters params) throws Exception {
		int numExamples = problem.size();

		epochUpdateCount = 0;
		problem.shuffle(random); // shuffle your training data after every iteration

		// Stage 0
		// E-learning
		
		
		
		
		// Stage 1
		// inferece
		
		
		for (int exampleId = 0; exampleId < numExamples; exampleId++) {
			IInstance example = problem.instanceList.get(exampleId);	// the input "x"
			IStructure gold = problem.goldStructureList.get(exampleId);	// the gold output structure "y"

			IStructure prediction = null;
			double loss = 0;
			prediction = this.inference.getLossAugmentedBestStructure(w, example, gold);	// the predicted structure
			loss = this.inference.getLoss(example, gold, prediction);	// we will update if the loss is non-zero for this example

			assert prediction != null;

			IFeatureVector goldFeatures = featureGenerator.getFeatureVector(example, gold); 
			IFeatureVector predictedFeatures = featureGenerator.getFeatureVector(example, prediction);
			IFeatureVector update = goldFeatures.difference(predictedFeatures);
			double loss_term = loss - w.dotProduct(update);

			double learningRate = getLearningRate(epoch, count, params);
			w.scale(1.0f-learningRate);
			w.addSparseFeatureVector(update, 2*learningRate*params.C_FOR_STRUCTURE*loss_term);
			//System.out.println(loss_term);
			epochUpdateCount++;

			count++;
		}
		
		
		
		// Stage 2
		for () {
			
		}
		
		return count;
	}
*/
	
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
