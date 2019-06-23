package essvm;

import java.util.ArrayList;
import java.util.Collections;
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
import edu.illinois.cs.cogcomp.sl.util.Pair;
import edu.illinois.cs.cogcomp.sl.util.SparseFeatureVector;
import edu.illinois.cs.cogcomp.sl.util.TimeCnter;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import multilabel.instance.Label;
import sequence.hw.HwOutput;

public class SSVMBookMarksDCDLearner extends Learner {
	
	static Logger logger = LoggerFactory.getLogger(SSVMBookMarksDCDLearner.class);
	static Random random = new Random(0);
	
	static int labelSize = 0;
	
	public SSVMBookMarksDCDLearner(AbstractInferenceSolver infSolver, AbstractFeatureGenerator fg, SLParameters param, int lbsz) {
		super(infSolver, fg, param);
		labelSize = lbsz;
	}

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

	@Override
	public WeightVector train(SLProblem sp) throws Exception {
		return trainLocal(sp, parameters);
	}

	@Override
	public WeightVector train(SLProblem sp, WeightVector init) throws Exception {
		return trainLocal(sp, parameters,init);
	}


	//////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////


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

	public WeightVector trainLocal(SLProblem sp, SLParameters parameters,
			WeightVector init) throws Exception {
		return DCDForL2LossSSVM(init, infSolver, sp, parameters);
	}

	protected WeightVector DCDForL2LossSSVM(WeightVector oldWv,
			final AbstractInferenceSolver infSolver, SLProblem sp,
			SLParameters parameters) throws Exception {
		int size = sp.size();
		float dualObj = 0;

		WeightVector wv = new WeightVector(oldWv);

		logger.info("Construct instances with alpha.");
		BookMarksAlphaStruct[] alphaInsList = initArrayOfInstances(sp,
				parameters.C_FOR_STRUCTURE, size);
		BookMarksAlphaStruct.CACHE_ALPHA_FEATURE = parameters.CACHE_ALPHA_FEATURE_VECTOR;
		logger.info("CACHE_ALPHA_FEATURE = " + BookMarksAlphaStruct.CACHE_ALPHA_FEATURE);
		logger.info("Done alpha-instance construction.");

		boolean finished = false;
		boolean resolved = false;

		// train the inner loop

		resolved = false;
		finished = false;

		//for(int iter=0; iter < parameters.MAX_NUM_ITER; iter++){
		//for(int iter=0; iter < 131; iter++){
		for(int iter=0; iter < 40; iter++){

			TimeCnter timr1 = new TimeCnter();
			timr1.startTimer();
			int NumOfNewStructures = updateWorkingSet(alphaInsList, wv, infSolver, parameters);
			logger.info("Time spend on loss-aug inference: " + timr1.getElpseTime());

			// no more update is necessary, exit the internal loop
			if (NumOfNewStructures == 0) {
				if (finished == false){
					///////////resolved = true;
				}else{
					logger.info("Met the stopping condition; Exit Inner loop");
					logger.info("negative dual obj = " + dualObj);
					///////////break;
				}
			}

			TimeCnter timr2 = new TimeCnter();
			timr2.startTimer();
			// update weight vector and alphas based on the working set
			Pair<Float, Boolean> res = updateWvWithWorkingSet(alphaInsList, wv, parameters);
			logger.info("Time spend on weight update: " + timr2.getElpseTime());

			if(iter % parameters.PROGRESS_REPORT_ITER == 0) {
				logger.info("Iteration: " + iter
						+ ": Add " + NumOfNewStructures
						+ " candidate structures into the working set.");
				logger.info("negative dual obj = " + res.getFirst());
				if(f!=null)
					f.run(wv, infSolver);
			}

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

	// this code runs line 3 to 8 from algorithm 3 (DCD-SSVM) in the paper
	protected static Pair<Float, Boolean> updateWvWithWorkingSet(
			BookMarksAlphaStruct[] alphaInsList, WeightVector wv,
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

		int totalCall = 0;
		int totalLoop = 0;
		int interIter = 0;
		double avgWorkSetSize = 0;
		

		for (t = 0; t < parameters.MAX_ITER_INNER; t++) { //for (t = 0; t < 1; t++) {
			interIter++;
			avgWorkSetSize = 0;

			BookMarksAlphaStruct.L2SolverInfo si = new BookMarksAlphaStruct.L2SolverInfo();

			// shuffle the indices
			Collections.shuffle(indices, random);

			// coordinate descent
			for (int idx : indices) {
				alphaInsList[idx].solveSubProblemAndUpdateW(si, wv);
				totalCall++;
				totalLoop += alphaInsList[idx].solveCnt;
				avgWorkSetSize += ((double)alphaInsList[idx].candidateAlphas.size());
			}
			avgWorkSetSize = avgWorkSetSize / ((double)indices.size());

			
			if (si.PGMaxNew - si.PGMinNew <= parameters.INNER_STOP_CONDITION) {
				finished = true;
				break;
			}
		}
		
		////
		int totalWorkSety = 0;
		int totalWorkSetOnes = 0;
		int totalWorkSetFeatSpar = 0;
		for (int idx : indices) {
			totalWorkSety += (alphaInsList[idx].candidateAlphas.size());
			totalWorkSetOnes += (alphaInsList[idx].countWorkSetStructOnes());
			totalWorkSetFeatSpar += (alphaInsList[idx].countFeatureSparsities());
		}
		double avgOnes = ((double)totalWorkSetOnes) / ((double)totalWorkSety);
		double avgfspar = ((double)totalWorkSetFeatSpar) / ((double)totalWorkSety);
		
		logger.info("solveSubProblemAndUpdateW call " + totalCall);
		logger.info("solveSubProblemAndUpdateW loop " + totalLoop);	
		logger.info("avergeWorksetSize " + avgWorkSetSize);
		logger.info("avergeStructSparsity " + avgOnes);
		logger.info("avergeFeatureSparsity " + avgfspar);	
		logger.info("innerIteration " + interIter);
		

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
	private int updateWorkingSet(
			BookMarksAlphaStruct[] alphaInsList, WeightVector wv,
			AbstractInferenceSolver infSolver, SLParameters parameters)
					throws Exception {
		int numNewStructures = 0;

		// update working set for each training example
		// line 10-16 in Algorithm 3 DCD-SSVM
		for (int i = 0; i < alphaInsList.length; i++) {
			float score = alphaInsList[i].updateRepresentationCollection(wv, infSolver, parameters);
			alphaInsList[i].solveSubProblemAndUpdateW(null, wv);
			if (score > parameters.STOP_CONDITION)	// 
				numNewStructures += 1;	
		}
		
		//////////////////////////////////////////////////////////////////////
		int totalWorkSety = 0;
		int totalWorkSetOnes = 0;
		for (int i = 0; i < alphaInsList.length; i++) {
			totalWorkSety += (alphaInsList[i].candidateAlphas.size());
			totalWorkSetOnes += (alphaInsList[i].countWorkSetStructOnes());
		}
		double avgOnes = ((double)totalWorkSetOnes) / ((double)totalWorkSety);
		logger.info("avergeStructSparsity " + avgOnes);	
		//////////////////////////////////////////////////////////////////////
		
		return numNewStructures;
	}

	// returns the objective as written in equation (4) in the paper
	protected static float getDualObjective(
			BookMarksAlphaStruct[] alphaInsList, WeightVector wv) {
		float obj = 0;

		obj += wv.getSquareL2Norm() * 0.5;

		for (int i = 0; i < alphaInsList.length; i++) {
			BookMarksAlphaStruct instanceWithAlphas = alphaInsList[i];
			float w_sum = instanceWithAlphas.getLossWeightAlphaSum();
			float sum = instanceWithAlphas.alphaSum;
			float C = instanceWithAlphas.getC();
			obj -= w_sum;
			obj += (1.0 / (4.0 * C)) * sum * sum;
		}
		return obj;
	}


	protected static void printTotalNumberofAlphas(
			BookMarksAlphaStruct[] alphaInsList) {
		int n_total_alphas = 0;
		int n_ex = alphaInsList.length;
		for (int i = 0; i < n_ex; i++) {
			BookMarksAlphaStruct alphaIns = alphaInsList[i];
			n_total_alphas += alphaIns.candidateAlphas.size();
		}

		logger.trace("Number of ex: " + alphaInsList.length);
		logger.trace("Number of alphas: " + n_total_alphas);
	}

	protected BookMarksAlphaStruct[] initArrayOfInstances(
			SLProblem sp, final float CStructure,
			int size) {
		// create the dual variables for each example
		BookMarksAlphaStruct[] alphInsList = new BookMarksAlphaStruct[size];
		
		//SparseFeatureVector constantFv = computeConstantFeatureVector(sp.instanceList.get(0));

		// initialization: structure
		if (sp.instanceWeightList == null) {
			//double totalDense = 0;
			for (int i = 0; i < sp.size(); i++) {
				alphInsList[i] = new BookMarksAlphaStruct(
						sp.instanceList.get(i), sp.goldStructureList.get(i),
						CStructure,featureGenerator);
				//SparseFeatureVector gf = (SparseFeatureVector)featureGenerator.getFeatureVector(sp.instanceList.get(i), sp.goldStructureList.get(i));
				//SparseFeatureVector fdiff = null;
				//totalDense += ((double)gf.getNumActiveFeatures());
				if ((i % 10000) == 0) {
					logger.info("Finished alphaInstance " + i + " construction.");
				}
			}
			//totalDense = totalDense / ((double)sp.size());
			//logger.info("average gold feature sparsity = " + totalDense);
		} else {
			for (int i = 0; i < sp.size(); i++) {
				alphInsList[i] = new BookMarksAlphaStruct(
						sp.instanceList.get(i), sp.goldStructureList.get(i),
						CStructure * sp.instanceWeightList.get(i),featureGenerator);
				if ((i % 10000) == 0) {
					logger.info("Finished alphaInstance " + i + " construction.");
				}
			}
		}
		return alphInsList;
	}
	
	//private SparseFeatureVector featureMinusConstantVector(SparseFeatureVector fv, SparseFeatureVector cv) {
	//	
	//}
	
	private SparseFeatureVector computeConstantFeatureVector(IInstance inst) {
		HwOutput y0 = new HwOutput(labelSize, Label.MULTI_LABEL_DOMAIN);
		for (int i = 0; i < y0.size(); i++) {
			y0.setOutput(i, 0);
		}
		SparseFeatureVector zfv = (SparseFeatureVector)featureGenerator.getFeatureVector(inst, y0);
		return zfv;
	}

}