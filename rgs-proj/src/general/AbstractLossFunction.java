package general;

import java.util.ArrayList;
import java.util.List;

import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import search.SearchAction;
import search.loss.GoldPredPair;
import search.loss.LossScore;

public abstract class AbstractLossFunction {
	
	//// ====LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS ====================================================
	public abstract LossScore computeZeroOneLossIncreamentally(SearchAction action, IInstance ins, IStructure goldStructure,  IStructure structure);
	public abstract LossScore computeZeroOneLosss(IInstance ins, IStructure goldStructure,  IStructure structure);
	//// ====LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS ====================================================

	public abstract LossScore computeZeroOneAccIncreamentally(SearchAction action, AbstractInstance ins, IStructure goldStructure,  IStructure structure);
	public abstract LossScore computeZeroOneAcc(AbstractInstance ins, IStructure goldStructure,  IStructure structure);

	public abstract double getAccuracyFullScore(AbstractInstance ins);
	
	public abstract String getName();
	
	public abstract double computeMacro(List<LossScore> scores);
	public double computeMacroFromResults(List<GoldPredPair> results) {

		List<LossScore> scores = new ArrayList<LossScore>();
		for (GoldPredPair pair : results) {
			LossScore sc = computeZeroOneAcc(pair.inst, (IStructure)pair.gold,  (IStructure)pair.pred);
			scores.add(sc);
		}
		
		double res = computeMacro(scores);
		return res;
		
	}

}