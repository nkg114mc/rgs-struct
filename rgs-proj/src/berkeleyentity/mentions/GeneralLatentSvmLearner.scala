package berkeleyentity.mentions

import edu.illinois.cs.cogcomp.sl.core.SLParameters
import edu.illinois.cs.cogcomp.sl.util.WeightVector
import edu.illinois.cs.cogcomp.sl.core.SLProblem
import edu.illinois.cs.cogcomp.sl.learner.Learner
import edu.illinois.cs.cogcomp.sl.latentsvm.AbstractLatentInferenceSolver
import edu.illinois.cs.cogcomp.sl.core.AbstractFeatureGenerator
import edu.illinois.cs.cogcomp.sl.core.IInstance

class GeneralLatentSvmLearner(val baseLearner: Learner, 
                              val fg: AbstractFeatureGenerator, 
                              val params: SLParameters, 
                              val solver: AbstractLatentInferenceSolver) extends Learner(solver,fg, params) {
  
  def train(problem: SLProblem) = {
		train(problem, new WeightVector(100000));
	}
  
	def train(problem: SLProblem, w_init: WeightVector): WeightVector = {

		var w = w_init; // new WeightVector(100000);//baseLearner.train(problem); // init w

		//for (int outerIter = 0; outerIter < params.MAX_NUM_ITER; outerIter++) {
		for (outerIter <- 0 until params.MAX_NUM_ITER) {

		  val new_prob = runLatentStructureInference(problem, w, solver); // returns structured problem with (x_i,h_i)
			w = baseLearner.train(new_prob, w); // update weight vector
			
			//w.checkFloatDoubleConsistency();

			if (params.PROGRESS_REPORT_ITER > 0 && (outerIter+1) % params.PROGRESS_REPORT_ITER == 0 && this.f != null) {
				f.run(w, solver);
			}
		}

		return w;
	}
	
  def train(problem: SLProblem,
            w_init: WeightVector,
            testExs: Seq[IInstance],
            testFunc: (Seq[IInstance], WeightVector) => Any): WeightVector = {

		var w = w_init; // init w

		//for (int outerIter = 0; outerIter < params.MAX_NUM_ITER; outerIter++) {
		for (outerIter <- 0 until params.MAX_NUM_ITER) {

		  val new_prob = runLatentStructureInference(problem, w, solver); // returns structured problem with (x_i,h_i)
			w = baseLearner.train(new_prob, w); // update weight vector
			
			//w.checkFloatDoubleConsistency();

			if (params.PROGRESS_REPORT_ITER > 0 && (outerIter+1) % params.PROGRESS_REPORT_ITER == 0 && this.f != null) {
				f.run(w, solver);
			}
			
			
			// have a check the performance
      //searcher.beamSearchQuickTest(testExs, beamSize, w.getDoubleArray, pruner, restart);
			//SingletonDetection.runSingletonEvaluation(testExs, w.getDoubleArray)
			testFunc(testExs, w)
		}

		return w;
	}

	def runLatentStructureInference(problem: SLProblem,
														      w: WeightVector, 
                                  inference: AbstractLatentInferenceSolver): SLProblem = {
	  
		val p = new SLProblem();
		for (i <- 0 until problem.size()) {
			val x = problem.instanceList.get(i);
			val gold = problem.goldStructureList.get(i);
			val y = inference.getBestLatentStructure(w, x, gold); // best
			// explaining latent structure
			p.instanceList.add(x);
			p.goldStructureList.add(y);
		}

		return p;
	}

}