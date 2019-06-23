package multilabel.pruner;

import java.util.List;

import general.AbstractActionGenerator;
import general.AbstractInstance;
import search.SearchAction;
import search.SearchState;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;

/**
 * This pruner works for MultiLabel problem only
 * 
 * @author mc
 *
 */
public class PrunedActionGenerator extends AbstractActionGenerator {

	private class OutputSlot {
		
	}
	
	@Override
	public List<SearchAction> genAllAction(AbstractInstance instace, SearchState currState) {
		
		
		HwOutput gold = instace.getGoldOutput();
		
		
		
		// TODO Auto-generated method stub
		return null;
	}
	

}
