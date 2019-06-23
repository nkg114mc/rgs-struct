package essvm;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.illinois.cs.cogcomp.sl.core.AbstractFeatureGenerator;
import edu.illinois.cs.cogcomp.sl.core.AbstractInferenceSolver;
import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.core.SLParameters;
import edu.illinois.cs.cogcomp.sl.util.IFeatureVector;
import edu.illinois.cs.cogcomp.sl.util.SparseFeatureVector;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import general.AbstractInstance;
import general.AbstractOutput;
import multilabel.MultiLabelFeaturizer;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;

public class BookMarksAlphaStruct {

	static Logger logger = LoggerFactory.getLogger(BookMarksAlphaStruct.class);
	public static int MAX_DCD_INNNER_ITER = 10;
	public static float DCD_INNNER_STOP = 0.1f;

	public static boolean CACHE_ALPHA_FEATURE = true;

	public int solveCnt = 0;

	public final static float UPDATE_CONDITION = 1e-8f;

	public IInstance ins = null;
	public float sC = 0.0f;
	public float alphaSum;

	public static class L2SolverInfo {
		public float PGMaxNew = Float.NEGATIVE_INFINITY;
		public float PGMinNew = Float.POSITIVE_INFINITY;
	}

	public static class AlphaStruct{
		float alpha;
		float loss;
		IFeatureVector alphaFeactureVector;
		IStructure struct;
		//MultiLabelFactorizedFeature mlfacfeat;
	}

	public AbstractFeatureGenerator featGenerator;
	public IStructure goldStructure;
	//public IFeatureVector goldFeatureVector;
	//private IFeatureVector goldFeatureVector;
	public List<AlphaStruct> candidateAlphas;
	// the following two arrays are designed for avoiding ConcurrentModificationException
	// when using DEMI-DCD.

	public List<AlphaStruct> newCandidateAlphas;
	public Set<IStructure> candidateSet = Collections.newSetFromMap(new HashMap<IStructure, Boolean>());
	
	public static MultiLabelFactorizedFeatureGenerator factorFeaturizer = null;

	
	public BookMarksAlphaStruct(IInstance ins, IStructure goldStruct, float C, AbstractFeatureGenerator featGenr) {
		this.goldStructure = goldStruct;
		this.ins = ins;
		//this.goldFeatureVector = null;//featGenr.getFeatureVector(ins, goldStruct);	// 
		candidateAlphas = new ArrayList<AlphaStruct>();
		newCandidateAlphas = Collections.synchronizedList(new ArrayList<AlphaStruct>());
		sC = C;
		featGenerator = featGenr;
		if (factorFeaturizer == null) {
			factorFeaturizer = new MultiLabelFactorizedFeatureGenerator((MultiLabelFeaturizer)featGenr);
		}
	}

	public void recomputeAlphaFeatures() {
		for(AlphaStruct as: candidateAlphas) {
			//IFeatureVector best_features = featGenerator.getFeatureVector(ins, as.struct);
			//IFeatureVector gFeatureVector = featGenerator.getFeatureVector(ins, goldStructure);
			//as.alphaFeactureVector = gFeatureVector.difference(best_features);  // clear stored vector
			as.alphaFeactureVector = featGenerator.getFeatureVectorDiff(ins, goldStructure, as.struct);
		}
	}

	public void clearAlphaFeatures() {
		for(AlphaStruct as: candidateAlphas) {
			as.alphaFeactureVector = null; // clear stored vector
		}
	}

	/*
	 * Update an alpha element and w.
	 */
	public void solveSubProblemAndUpdateW(L2SolverInfo si, WeightVector wv) {
		// solve sub-problem over alphas associated with an instance.
		solveCnt = 0;
		int i = 0;
		float stop;
		candidateAlphas.addAll(newCandidateAlphas);
		newCandidateAlphas.clear();

		// re-compute alpha features
		if (!CACHE_ALPHA_FEATURE) {
			recomputeAlphaFeatures();
		}
		
		//MultiLabelFactorizedWeight factorizedWeight = new MultiLabelFactorizedWeight((HwInstance)ins, wv, factorFeaturizer);

		while(true){
			i++;
			float inner_PGmax_new = Float.NEGATIVE_INFINITY;
			float inner_PGmin_new = Float.POSITIVE_INFINITY;

			// this loop performs the update for each alpha separately
			for(AlphaStruct as: candidateAlphas){
				float alpha = as.alpha;
				float loss = as.loss;

				//MultiLabelFactorizedWeight factorizedWeight = new MultiLabelFactorizedWeight((HwInstance)ins, wv, factorFeaturizer);

				IFeatureVector fv = as.alphaFeactureVector;	// get the difference vector 
				//float dot_product = wv.dotProduct(fv);
				//float xij_norm2 = fv.getSquareL2Norm();
				float dot_product = computeDotProd(wv, fv);
				//float dot_product_fast = factorizedWeight.doProductDebug(as.mlfacfeat, (AbstractInstance)ins, (AbstractOutput)goldStructure, (AbstractOutput)as.struct);
				float xij_norm2 = computeNorm2(fv);
				
				//if (Math.abs(dot_product - dot_product_fast) > 0.00001) {
				//	throw new RuntimeException(dot_product + " != " + dot_product_fast);
				//}

				float NG = (loss - dot_product) - alphaSum / (2.0f * sC);

				float PG = -NG;	// projected gradient
				if (alpha == 0)
					PG = Math.min(-NG, 0);

				inner_PGmax_new = Math.max(inner_PGmax_new, PG);
				inner_PGmin_new = Math.min(inner_PGmin_new, PG);

				if (Math.abs(PG) > UPDATE_CONDITION) {

					float step = NG / (xij_norm2 + 1.0f / (2.0f * sC));
					float new_alpha = Math.max(alpha + step, 0);
					alphaSum += (new_alpha - alpha);
					wv.addSparseFeatureVector(fv, (new_alpha - alpha));
					//factorizedWeight.addFactorizedFeatureVector(as.mlfacfeat, (new_alpha - alpha));
					as.alpha = new_alpha;
				}

				solveCnt++;
			}

			stop = inner_PGmax_new - inner_PGmin_new;

			// satisfied inner stopping condition
			if (stop < DCD_INNNER_STOP || i >= MAX_DCD_INNNER_ITER){
				if(si != null){
					si.PGMaxNew = Math.max(si.PGMaxNew, inner_PGmax_new);
					si.PGMinNew = Math.min(si.PGMinNew, inner_PGmin_new);
				}
				break;
			}

		}

		// clear alpha features
		if (!CACHE_ALPHA_FEATURE) {
			clearAlphaFeatures();
		}
	}

	public  void cleanCache(WeightVector wv) {
		Iterator<AlphaStruct> iterator = candidateAlphas.iterator();
		while(iterator.hasNext()){
			AlphaStruct as = iterator.next();
			if(as.alpha <=1e-8){
				iterator.remove();
				candidateSet.remove(as.struct);

			}
		}
	}

	public float computeDotProd(WeightVector wv, IFeatureVector fv) {
		float pd = wv.dotProduct(fv);
		return pd;
	}

	public float computeNorm2(IFeatureVector fv) {
		float nm = fv.getSquareL2Norm();
		return nm;
	}
	
/*
	public IFeatureVector getGoldFeatureVector() {
		if (goldFeatureVector == null) {
			SparseFeatureVector computedGfv = (SparseFeatureVector) featGenerator.getFeatureVector(ins, goldStructure);
			if (computedGfv.getNumActiveFeatures() < 20000) {
				goldFeatureVector = computedGfv;// store gold feature
			}
			return computedGfv;
		} else {
			return goldFeatureVector;
		}
	}
*/
	
	/**
	 * changes the working set, returns 1 if the instance is added, 0 otherwise
	 * @param wv
	 * synchronized@param infSolver
	 * @param parameters
	 * @return
	 * @throws Exception
	 */
	public int updateRepresentationCollection(WeightVector wv,
			AbstractInferenceSolver infSolver, SLParameters parameters) throws Exception {

		float C = sC;

		IStructure h = infSolver.getLossAugmentedBestStructure(wv, ins, goldStructure);
		// already in candidateSet
		if(candidateSet.contains(h))
			return 0;

		float loss = infSolver.getLoss(ins, goldStructure, h);

		IFeatureVector diff = featGenerator.getFeatureVectorDiff(ins, goldStructure, h);
		//MultiLabelFactorizedFeature mlff = factorFeaturizer.getFactorizedFeatureVectorDiff((AbstractInstance)ins, (AbstractOutput)goldStructure, (AbstractOutput)h);
		/*
		IFeatureVector best_features = featGenerator.getFeatureVector(ins, h);
		IFeatureVector gFeatureVector = featGenerator.getFeatureVector(ins, goldStructure);
		IFeatureVector diffOrig = gFeatureVector.difference(best_features);
		float dp1 = wv.dotProduct(diff);
		float nm1 = computeNorm2(diff);
		float dp2 = wv.dotProduct(diffOrig);
		float nm2 = computeNorm2(diffOrig);
		assert (dp1 == dp2);
		assert (nm1 == nm2);
		*/
		float xi = alphaSum / (2.0f * C);
		float dotProduct = wv.dotProduct(diff);
		float score = (loss - dotProduct) - xi;	// line 12 in DCD-SSVM (algorithm 3 in paper)

		if(parameters.CHECK_INFERENCE_OPT) {
			float max_score_in_cache = Float.NEGATIVE_INFINITY;
			for(AlphaStruct as:new ArrayList<AlphaStruct>(candidateAlphas)){
				if(as !=null){
					float s = as.loss - wv.dotProduct(as.alphaFeactureVector) - xi;
					if (max_score_in_cache < s)
						max_score_in_cache = s;
				}
			}

			if (score < max_score_in_cache - 1e-4) {
				if(logger.isErrorEnabled()){
					printErrorLogForIncorrectInference(wv, loss, h, xi, dotProduct, score,
							max_score_in_cache);
				}
				throw new Exception(
						"The inference procedure obtains a sub-optimal solution!"
								+ "If you want to use an approximate inference solver, set SLParameter.check_inference_opt = false.");
			}
		}
		 

		if (score < parameters.STOP_CONDITION) // not enough contribution
			return 0;

		AlphaStruct as = new AlphaStruct();
		as.alpha = 0.0f;
		as.loss = loss;
		if (parameters.CACHE_ALPHA_FEATURE_VECTOR) { // store the alpha feature vector
			as.alphaFeactureVector = diff;
		} else {
			as.alphaFeactureVector = null;
		}
		as.struct = h;
		//as.mlfacfeat = mlff;
		newCandidateAlphas.add(as);
		candidateSet.add(h);
		return 1;
	}

	private void printErrorLogForIncorrectInference(WeightVector wv, float loss, IStructure h,
			float xi, float dotProduct, float score,
			float max_score_in_cache) {
		logger.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
		logger.error("The inference procedure finds a sub-optimal solution.");
		logger.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
		logger.error("If you want to use an approximate inference solver, set SLParameter.check_inference_opt = false.");

		logger.error("score: " + score);
		logger.error("max score of the cached structure: " + max_score_in_cache);
	}

	/**
	 * returns the sum of alphas, weighed by the corresponding losses (the last term of the dual objective in equation (4))
	 * @return
	 */
	public float getLossWeightAlphaSum() {
		float sum_alpha = 0f;
		for(AlphaStruct as: candidateAlphas){
			sum_alpha += as.loss*as.alpha;
		}
		return sum_alpha;

	}

	public float getC() {
		return sC;
	}
	
	
	public int countWorkSetStructOnes() {
		int worksetOnes = 0;
		for(AlphaStruct as: candidateAlphas) {
			HwOutput y = (HwOutput)as.struct;
			for (int i = 0; i < y.size(); i++) {
				if (y.getOutput(i) > 0) {
					worksetOnes++;
				}
			}
		}
		return worksetOnes;
	}
	
	public int countFeatureSparsities() {
		int worksetOnes = 0;
		for(AlphaStruct as: candidateAlphas) {
			SparseFeatureVector f = (SparseFeatureVector)as.alphaFeactureVector;
			worksetOnes += f.getNumActiveFeatures();
		}
		return worksetOnes;
	}

}