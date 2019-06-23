package essvm;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import berkeleyentity.MyTimeCounter;
import edu.illinois.cs.cogcomp.sl.core.AbstractFeatureGenerator;
import edu.illinois.cs.cogcomp.sl.core.AbstractInferenceSolver;
import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.core.SLParameters;
import edu.illinois.cs.cogcomp.sl.core.SLProblem;
import edu.illinois.cs.cogcomp.sl.learner.Learner;
import edu.illinois.cs.cogcomp.sl.learner.l2_loss_svm.L2LossSSVMDCDSolver;
import edu.illinois.cs.cogcomp.sl.learner.l2_loss_svm.StructuredInstanceWithAlphas;
import edu.illinois.cs.cogcomp.sl.util.Pair;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearning.AbstractRegressionLearner;
import elearning.EInferencer;
import elearning.einfer.ESamplingInferencer;
import elearning.einfer.SearchStateScoringFunction;
import elearnnew.SamplingELearning;
import general.AbstractFeaturizer;
import general.AbstractInstance;
import search.GreedySearcher;
import sequence.hw.HwSearchInferencer;

public class SSVMCopyDCDLearner extends Learner {
	
	static Logger logger = LoggerFactory.getLogger(L2LossSSVMDCDSolver.class);
	static Random random = new Random(0);
	
	//AbstractInferenceSolver infSolver; 
	//AbstractFeatureGenerator featureGenerator;
	
	protected static final SLProblem emptyStructuredProblem;

	static {
		emptyStructuredProblem = new SLProblem();
		emptyStructuredProblem.instanceList = new ArrayList<IInstance>();
		emptyStructuredProblem.goldStructureList = new ArrayList<IStructure>();
	}
	
	protected ProgressReportFunction f;
	
	public void runWhenReportingProgress(ProgressReportFunction f) {
		this.f = f;
	}

	public SSVMCopyDCDLearner(AbstractInferenceSolver infSolver, AbstractFeatureGenerator fg, SLParameters para){
		super(infSolver, fg, para);
	}
	
	
	@Override
	public WeightVector train(SLProblem sp) throws Exception {
		return trainLocal(sp, parameters);
	}


	@Override
	public WeightVector train(SLProblem sp, WeightVector init) throws Exception {
		return trainLocal(sp, parameters, init);
	}
	
	
	/**
	 * The function for the users to call for the structured SVM
	 * 
	 * @param sp
	 *            Structured Labeled Dataset
	 * @param params
	 *            parameters
	 * @return
	 * @throws Exception
	 */
	public WeightVector trainLocal(final SLProblem sp, SLParameters params) throws Exception {
		WeightVector wv = null;
		
		// +1 because wv.u[0] stores the bias term
		if(params.TOTAL_NUMBER_FEATURE >0){
			wv = new WeightVector(params.TOTAL_NUMBER_FEATURE + 1);
			wv.setExtendable(false);
		} else {
			wv = new WeightVector(8192);
			wv.setExtendable(true);
		}
		return trainLocal(sp,params,wv);
	}
	
	public WeightVector trainLocal(SLProblem sp, SLParameters parameters, WeightVector init) throws Exception {
		TrainResult trnRes = new TrainResult(null);
		return DCDForL2LossSSVM(init, null, infSolver, sp, parameters, trnRes, false, null, 0);
	}
	
	// our local train function
	public WeightVector trainNormail(SLProblem sp, SLParameters params, GreedySearcher gsearcher,// AbstractInferenceSolver infr, 
			                         TrainResult trnResult, boolean useEval,
			             			AbstractRegressionLearner regressionTrainer,
			            			int iteration) throws Exception {
		
		if (trnResult == null) {
			trnResult = new TrainResult(null);
		}
		
		// init weight
		WeightVector init = null;
		if(params.TOTAL_NUMBER_FEATURE >0){
			init = new WeightVector(params.TOTAL_NUMBER_FEATURE + 1);
			init.setExtendable(false);
		} else {
			init = new WeightVector(8192);
			init.setExtendable(true);
		}
		
		// search inferencer
		AbstractInferenceSolver infr = new HwSearchInferencer(gsearcher);
		
		return DCDForL2LossSSVM(init, gsearcher, infr, sp, parameters, trnResult, useEval, regressionTrainer, iteration);
	}

	protected WeightVector DCDForL2LossSSVM(WeightVector oldWv, GreedySearcher gsearcher, final AbstractInferenceSolver infrer,
			SLProblem sp, SLParameters parameters, TrainResult trainResult, boolean useEval,
			AbstractRegressionLearner regressionTrainer,
			int iteration) throws Exception {
		
		int size = sp.size();
		float dualObj = 0;

		WeightVector wv = new WeightVector(oldWv);

		StructuredInstanceWithAlphas[] alphaInsList = initArrayOfInstances(sp, parameters.C_FOR_STRUCTURE, size);


		boolean finished = false;
		boolean resolved = false;

		// train the inner loop

		resolved = false;
		finished = false;
		
		MyTimeCounter myTimer = new MyTimeCounter("ssvm");
		myTimer.start();

		for(int iter=0; iter < parameters.MAX_NUM_ITER; iter++){
			
			if (gsearcher != null) {
				if (useEval) {
					
					//if (iter > 0) {
					//}
					
					
				} else {
					gsearcher.setEvalInferencer(null); // no eval inferencer
				}
			}

			////=============================================================================================
			//double[] wCopy1 = wv.getDoubleArray().clone();
			int NumOfNewStructures = updateWorkingSet(alphaInsList, wv, infrer, gsearcher, parameters, useEval, regressionTrainer, iteration);
			//double[] wCopy2 = wv.getDoubleArray().clone();
			//compareVector(wCopy1, wCopy2);
			////=============================================================================================
			
			
			// no more update is necessary, exit the internal loop
			if (NumOfNewStructures == 0) {
				if (finished == false)
					resolved = true;
				else{
					logger.info("Met the stopping condition; Exit Inner loop");
					logger.info("negative dual obj = " + dualObj);
					break;
				}
			}

			// update weight vector and alphas based on the working set
			Pair<Float, Boolean> res = updateWvWithWorkingSet(alphaInsList, wv, parameters);
			
			////==============================================================================================
			long curTimeSpend = myTimer.getMilSecondSnapShot();
			//// done one iteration
			trainResult.addSnapshot((iter + 1), curTimeSpend, wv);
			// update result
			trainResult.iterNum = (iter + 1);
			trainResult.totalMilSeconds = curTimeSpend;
			////==============================================================================================
			logger.info("time = " + curTimeSpend);
			
			//if(iter % parameters.PROGRESS_REPORT_ITER == 0) {
				logger.info("Iterationn: " + iter
						+ ": Add " + NumOfNewStructures
						+ " candidate structures into the working set.");
				logger.info("negative dual obj = " + res.getFirst());
				if(f!=null)
					f.run(wv, infSolver);
			//}
			
			if (resolved) {
				finished = true;
				logger.info("(Resolved) Met the stopping condition; Exit Inner loop");
				logger.info("negative dual obj = " + res.getFirst());
				break;
			} else {
				finished = res.getSecond();
				dualObj = res.getFirst();
			}
			
			if(logger.isTraceEnabled()){
				printTotalNumberofAlphas(alphaInsList);
			}
			
			// remove unused candidate structures from working set
			if (parameters.CLEAN_CACHE && (iter+1) % parameters.CLEAN_CACHE_ITER == 0) {
				for (int i = 0; i < size; i++) {
					alphaInsList[i].cleanCache(wv);
				}
				if (logger.isInfoEnabled()) {
					logger.info("Cleaning cache....");	
					printTotalNumberofAlphas(alphaInsList);
				}
			}
			

			
			
			
			
			//wv.checkFloatDoubleConsistency();
			
		}
		
		return wv;
	}
	
	private void compareVector(double[] v1, double[] v2) {
		assert (v1.length == v2.length);
		for (int i = 0; i < v1.length; i++) {
			assert (v1[i] == v2[i]);
		}
	}
	
	// this code runs line 3 to 8 from algorithm 3 (DCD-SSVM) in the paper
	protected static Pair<Float, Boolean> updateWvWithWorkingSet(
			StructuredInstanceWithAlphas[] alphaInsList, WeightVector wv,
			SLParameters parameters) {
		// initialize w: w = sum alpha_i x_i
		int numIns = alphaInsList.length;

		logger.trace("STOPPING criteria:" + parameters.INNER_STOP_CONDITION);
		// coordinate descent

		List<Integer> indices = new ArrayList<Integer>();

		for (int i = 0; i < numIns; i++)
			indices.add(i);

		int t = 0;
		
		boolean finished = false;

		for (t = 0; t < parameters.MAX_ITER_INNER; t++) {

			StructuredInstanceWithAlphas.L2SolverInfo si = new StructuredInstanceWithAlphas.L2SolverInfo();

			// shuffle the indices
			Collections.shuffle(indices, random);

			// coordinate descent
			for (int idx : indices) {
				alphaInsList[idx].solveSubProblemAndUpdateW(si, wv);
			}

			if (si.PGMaxNew - si.PGMinNew <= parameters.INNER_STOP_CONDITION) {
				finished = true;
				break;
			}
		}

		float obj = getDualObjective(alphaInsList, wv);
		obj = -obj;
		return new Pair<Float, Boolean>(obj, finished);
	}
	
	/**
	 * returns number of newly added items in the working set
	 * @param alphaInsList
	 * @param wv
	 * @param infSolver
	 * @param parameters
	 * @return
	 * @throws Exception
	 */
	private int updateWorkingSet(StructuredInstanceWithAlphas[] alphaInsList, WeightVector wv,
			 					 AbstractInferenceSolver infSolver, 
								 GreedySearcher gsearcher,
			 					 SLParameters parameters,
			 					 boolean useEval,
			 					 AbstractRegressionLearner regressionTrainer,
			 					 int iteration) throws Exception {
		int numNewStructures = 0;

		// update working set for each training example
		// line 10-16 in Algorithm 3 DCD-SSVM
		for (int i = 0; i < alphaInsList.length; i++) {
			
			StructuredInstanceWithAlphas alphaInst = alphaInsList[i];
			AbstractInstance inst = (AbstractInstance) alphaInst.ins;
			
			
			///////////////////////////////////////////////////////////////////////////////////
			if (useEval) {
				AbstractFeaturizer efeaturizer = gsearcher.getFeaturizer();
				SearchStateScoringFunction emd = SamplingELearning.trainOneInstanceXgb(gsearcher.getInitGenerator(), inst, wv, gsearcher, regressionTrainer, iteration);
				EInferencer einfr = new ESamplingInferencer(gsearcher.getInitGenerator(), wv, gsearcher.getFeaturizer(), emd, efeaturizer, iteration);
				gsearcher.setEvalInferencer(einfr);
				
			} else {
				gsearcher.setEvalInferencer(null);
			}
			///////////////////////////////////////////////////////////////////////////////////
			
			float score = alphaInst.updateRepresentationCollection(wv, infSolver, parameters);
			//alphaInsList[i].solveSubProblemAndUpdateW(null, wv);
			if (score > parameters.STOP_CONDITION)
				numNewStructures += 1;	
		}
		return numNewStructures;
	}
	
	// returns the objective as written in equation (4) in the paper
	protected static float getDualObjective(
			StructuredInstanceWithAlphas[] alphaInsList, WeightVector wv) {
		float obj = 0;

		obj += wv.getSquareL2Norm() * 0.5;

		for (int i = 0; i < alphaInsList.length; i++) {
			StructuredInstanceWithAlphas instanceWithAlphas = alphaInsList[i];
			float w_sum = instanceWithAlphas.getLossWeightAlphaSum();
			float sum = instanceWithAlphas.alphaSum;
			float C = instanceWithAlphas.getC();
			obj -= w_sum;
			obj += (1.0 / (4.0 * C)) * sum * sum;
		}
		return obj;
	}
	
	
	protected static void printTotalNumberofAlphas(
			StructuredInstanceWithAlphas[] alphaInsList) {
		int n_total_alphas = 0;
		int n_ex = alphaInsList.length;
		for (int i = 0; i < n_ex; i++) {
			StructuredInstanceWithAlphas alphaIns = alphaInsList[i];
			//n_total_alphas += alphaIns.alphaFeatureVectorList.size();
			n_total_alphas += alphaIns.candidateAlphas.size();
		}

		logger.trace("Number of ex: " + alphaInsList.length);
		logger.trace("Number of alphas: " + n_total_alphas);
	}


	protected StructuredInstanceWithAlphas[] initArrayOfInstances(
			SLProblem sp, final float CStructure,
			int size) {
		// create the dual variables for each example
		StructuredInstanceWithAlphas[] alphInsList = new StructuredInstanceWithAlphas[size];

		// initialization: structure
		if (sp.instanceWeightList == null) {
			for (int i = 0; i < sp.size(); i++) {
				alphInsList[i] = new StructuredInstanceWithAlphas(
						sp.instanceList.get(i), sp.goldStructureList.get(i),
						CStructure,featureGenerator);
			}
		} else {
			for (int i = 0; i < sp.size(); i++) {
				alphInsList[i] = new StructuredInstanceWithAlphas(
						sp.instanceList.get(i), sp.goldStructureList.get(i),
						CStructure * sp.instanceWeightList.get(i),featureGenerator);
			}
		}
		return alphInsList;
	}
	
	////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////


	
	/**
	 * Get primal objective function value with respect to the weight vector wv
	 * @param sp
	 * @param wv
	 * @param infSolver
	 * @param C
	 * @return
	 * @throws Exception
	 */
	public float getPrimalObjective(
			SLProblem sp, WeightVector wv,
			AbstractInferenceSolver infSolver, float C) throws Exception {
		float obj = 0;

		obj += wv.getSquareL2Norm() * 0.5;
		List<IInstance> input_list = sp.instanceList;
		List<IStructure> output_list = sp.goldStructureList;
		for (int i = 0; i < input_list.size(); i++) {
			IInstance ins = input_list.get(i);
			IStructure gold_struct = output_list.get(i);
			float sC= C;
			IStructure h = infSolver
					.getLossAugmentedBestStructure(wv, ins, gold_struct);
			float loss = infSolver.getLoss(ins, gold_struct, h)
					+ this.featureGenerator.decisionValue(wv, ins, h)
					- this.featureGenerator.decisionValue(wv, ins, gold_struct);
			obj += sC * loss * loss;
		}
		return obj;
	}




}
