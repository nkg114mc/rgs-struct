package berkeleyentity.randsearch;

import java.util.ArrayList;
import java.util.List;

import general.AbstractInstance;
//import ims.hotcoref.oregonstate.HotCorefDocInstance;
import search.SeachActionGenerator;
import search.SearchAction;
import search.SearchState;

public class AceCorefActionGenerator  extends SeachActionGenerator {
	
	boolean useGold = false;
	
	public AceCorefActionGenerator(boolean gd) {
		useGold = gd;
	}
	

	@Override
	public List<SearchAction> genAllAction(AbstractInstance instace, SearchState currState) {
	
		AceCorefInstance hotInst = (AceCorefInstance)instace;
	
		List<SearchAction> actions = new ArrayList<SearchAction>();

		assert (currState.structOutput.size() == instace.size());
		
		for (int i = 0; i < currState.structOutput.size(); i++) {
			
			int[] domains = hotInst.getDomainGivenIndex(i); // all actions
			if (useGold) {
				domains = hotInst.getGoldDomainGivenIndex(i); // gold actions only
			}
			for (int j = 0; j < domains.length; j++) {
				int oldv = currState.structOutput.getOutput(i);
				int newv = domains[j];
				if (oldv != newv) {
					SearchAction act = new SearchAction(i, newv, oldv);
					actions.add(act);
				}
			}
		}

		return actions;
	}
}
