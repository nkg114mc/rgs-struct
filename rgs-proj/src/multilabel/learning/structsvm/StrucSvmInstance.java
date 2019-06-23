package multilabel.learning.structsvm;

import multilabel.instance.Example;
import edu.illinois.cs.cogcomp.sl.core.IInstance;

public class StrucSvmInstance implements IInstance {
	
	public Example example;
	
	public StrucSvmInstance(Example ex) {
		example = ex;
	}

}
