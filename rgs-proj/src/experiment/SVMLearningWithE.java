package experiment;

import edu.illinois.cs.cogcomp.sl.core.AbstractFeatureGenerator;
import edu.illinois.cs.cogcomp.sl.core.AbstractInferenceSolver;
import edu.illinois.cs.cogcomp.sl.core.SLParameters;
import edu.illinois.cs.cogcomp.sl.learner.Learner;
import essvm.SSVMSGDSolverWithEval;
import experiment.RndLocalSearchExperiment.DataSetName;

public class SVMLearningWithE {

	public static void main(String[] args) {

	}
	
	public static Learner getSvmWithELearner(DataSetName dname,
			                       AbstractInferenceSolver infSolver,
			                       AbstractFeatureGenerator fg, 
			                       SLParameters parameters) {
		
		parameters.DECAY_LEARNING_RATE = false;
		//Learner esvm = new SSVMSGDSolverWithEval(infSolver, fg, parameters);
		//return esvm;
		return null;
	}
	
	

}
