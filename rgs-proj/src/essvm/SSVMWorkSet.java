package essvm;

import java.util.HashMap;
import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;

public class SSVMWorkSet {
	
	int worksetSize = 0;
	public HashMap<IInstance, HashMap<IStructure, SSVMConstraint>> cachedConstrs;
	
	public SSVMWorkSet() {
		cachedConstrs = new HashMap<IInstance, HashMap<IStructure, SSVMConstraint>>();
	}
	
	private HashMap<IStructure, SSVMConstraint> getOrCreate(IInstance exmp) {
		if (cachedConstrs.containsKey(exmp)) {
			return cachedConstrs.get(exmp);
		} else {
			HashMap<IStructure, SSVMConstraint> newSlot = new HashMap<IStructure, SSVMConstraint>();
			cachedConstrs.put(exmp,newSlot);
			return newSlot;
		}
	}
	
	
	public boolean existOutput(IInstance x, IStructure y) {
		if (cachedConstrs.containsKey(x)) {
			HashMap<IStructure, SSVMConstraint> instanceSet = cachedConstrs.get(x);
			if (instanceSet.containsKey(y)) {
				return true;
			} else {
				return false;
			}
		} else {
			return false;
		}
	}
	
	public boolean existInstance(IInstance x) {
		if (cachedConstrs.containsKey(x)) {
			HashMap<IStructure, SSVMConstraint> instanceSet = cachedConstrs.get(x);
			if (instanceSet.size() > 0) {
				return true;
			}
		}
		return false;
	}
	
	// return set increasement
	public int insert(IInstance x, IStructure y, SSVMConstraint yinfo) {
		HashMap<IStructure, SSVMConstraint> instanceSet = getOrCreate(x);
		if (instanceSet.containsKey(y)) {
			// not need to add
			return 0;
		} else {
			instanceSet.put(y, yinfo);
			worksetSize++;
			return 1;
		}
	}
	
	public int computeSize() {
		int sz = 0;
		for (HashMap<IStructure, SSVMConstraint> instSet : cachedConstrs.values()) {
			sz += instSet.size();
		}
		return sz;
	}
	
	public int size() {
		
		int compSz = computeSize();
		assert (compSz == worksetSize);
		
		return worksetSize;
	}
}