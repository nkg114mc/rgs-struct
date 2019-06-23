package essvm;


import edu.illinois.cs.cogcomp.sl.core.AbstractFeatureGenerator;
import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.learner.l2_loss_svm.StructuredInstanceWithAlphas;

public class AlphaInstanceWithEval extends StructuredInstanceWithAlphas {

	public AlphaInstanceWithEval(IInstance ins, IStructure goldStruct, float C, AbstractFeatureGenerator featGenr) {
		super(ins, goldStruct, C, featGenr);
	}

}
