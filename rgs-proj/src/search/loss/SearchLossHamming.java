package search.loss;

import java.util.List;

import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import general.AbstractInstance;
import general.AbstractLossFunction;
import general.AbstractOutput;
import search.SearchAction;

public class SearchLossHamming extends AbstractLossFunction {
	
	public String getName() {
		return "SearchLossHamming";
	}
	
	//// ====LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS ====================================================
	public LossScore computeZeroOneLossIncreamentally(SearchAction action, IInstance ins, IStructure goldStructure,  IStructure structure) {
		AbstractOutput goldLabeledSeq = (AbstractOutput) goldStructure;
		double lossChange = 0;
		int goldValue = goldLabeledSeq.getOutput(action.getSlotIdx());
		if (action.getOldVal() == goldValue) { // old is correct
			lossChange += 1.0;
		}
		if (action.getNewVal() == goldValue) {
			lossChange -= 1.0;
		}
		return (new LossScoreHamm(lossChange, goldLabeledSeq.size()));
	}
	
	public LossScore computeZeroOneLosss(IInstance ins, IStructure goldStructure,  IStructure structure) {
		AbstractOutput goldLabeledSeq = (AbstractOutput) goldStructure;
		double loss = 0;
		for (int i = 0; i < goldLabeledSeq.size(); i++) {
			if (((AbstractOutput) structure).getOutput(i) != goldLabeledSeq.getOutput(i)) {
				loss += 1.0;
			}
		}
		return (new LossScoreHamm(loss, goldLabeledSeq.size()));
	}
	//// ====LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS ====================================================
	

	
	public LossScore computeZeroOneAccIncreamentally(SearchAction action, AbstractInstance ins, IStructure goldStructure,  IStructure structure) {
		AbstractOutput goldLabeledSeq = (AbstractOutput) goldStructure;
		double accChange = 0;
		int goldValue = goldLabeledSeq.getOutput(action.getSlotIdx());
		if (action.getOldVal() == goldValue) { // old is correct
			accChange -= 1.0;
		}
		if (action.getNewVal() == goldValue) {
			accChange += 1.0;
		}
		return (new LossScoreHamm(accChange, goldLabeledSeq.size()));
	}
	
	public LossScore computeZeroOneAcc(AbstractInstance ins, IStructure goldStructure,  IStructure structure) {
		AbstractOutput goldLabeledSeq = (AbstractOutput) goldStructure;
		double acc = 0;
		for (int i = 0; i < goldLabeledSeq.size(); i++) {
			if (((AbstractOutput) structure).getOutput(i) == goldLabeledSeq.getOutput(i)) {
				acc += 1.0;
			}
		}
		return (new LossScoreHamm(acc, goldLabeledSeq.size()));
	}

	@Override
	public double getAccuracyFullScore(AbstractInstance ins) {
		double accfull = (double)ins.size();
		return accfull;
	}

	@Override
	public double computeMacro(List<LossScore> scores) {
		
		double total = 0;
		double crr = 0;
		
		for (LossScore sc : scores) {
			LossScoreHamm sc2 = (LossScoreHamm)sc;
			total += sc2.totalCnt;
			crr += sc2.correctCnt;
		}
		
		double acc = crr / total;
		return acc;
	}
}
