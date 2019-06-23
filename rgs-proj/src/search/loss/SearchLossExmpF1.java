package search.loss;

import java.util.List;

import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import general.AbstractInstance;
import general.AbstractLossFunction;
import general.AbstractOutput;
import search.SearchAction;

public class SearchLossExmpF1 extends AbstractLossFunction {
	
	public String getName() {
		return "SearchLossExmpF1";
	}
	
	//// ====LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS ====================================================
	public LossScore computeZeroOneLossIncreamentally(SearchAction action, IInstance ins, IStructure goldStructure,  IStructure structure) {
		AccuracyScoreExmpF1 lscDiff = (AccuracyScoreExmpF1)computeZeroOneAccIncreamentally(action, (AbstractInstance)(ins), goldStructure, structure);
		return (new LossScoreExmpF1(lscDiff.getTp(), lscDiff.getTn(), lscDiff.getFp(), lscDiff.getFn()));
	}
	
	public LossScore computeZeroOneLosss(IInstance ins, IStructure goldStructure,  IStructure structure) {
		AccuracyScoreExmpF1 lsc = (AccuracyScoreExmpF1)computeZeroOneAcc((AbstractInstance)(ins), goldStructure, structure);
		return (new LossScoreExmpF1(lsc.getTp(), lsc.getTn(), lsc.getFp(), lsc.getFn()));
	}
	//// ====LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS LOSS ====================================================
	

	
	public LossScore computeZeroOneAccIncreamentally(SearchAction action, AbstractInstance ins, IStructure goldStructure,  IStructure structure) {
		AbstractOutput goldLabeledSeq = (AbstractOutput) goldStructure;
		double tp = 0;
		double tn = 0;
		double fp = 0;
		double fn = 0;
		int goldValue = goldLabeledSeq.getOutput(action.getSlotIdx());
		if (action.getOldVal() == goldValue) { // old is correct
			int pred = action.getOldVal();
			if (pred == 0) {
				tn--;
			} else {
				tp--;
			}
		} else { // old is wrong
			int pred = action.getOldVal();
			if (pred == 0) {
				fn--;
			} else {
				fp--;
			}
		}
		if (action.getNewVal() == goldValue) { // new is correct
			int pred = action.getNewVal();
			if (pred == 0) {
				tn++;
			} else {
				tp++;
			}
		} else {
			int pred = action.getNewVal();
			if (pred == 0) {
				fn++;
			} else {
				fp++;
			}
		}
		return (new AccuracyScoreExmpF1(tp, tn, fp, fn));
	}
	
	public LossScore computeZeroOneAcc(AbstractInstance ins, IStructure goldStructure,  IStructure structure) {
		AbstractOutput goldLabeledSeq = (AbstractOutput) goldStructure;
		double tp = 0;
		double tn = 0;
		double fp = 0;
		double fn = 0;
		for (int i = 0; i < goldLabeledSeq.size(); i++) {
			int pred = ((AbstractOutput) structure).getOutput(i);
			int gold = goldLabeledSeq.getOutput(i);
			if (pred == gold) {
				if (pred == 0) {
					tn++;
				} else {
					tp++;
				}
			} else {
				if (pred == 0) {
					fn++;
				} else {
					fp++;
				}
			}
		}
		return (new AccuracyScoreExmpF1(tp, tn, fp, fn));
	}
	
	@Override
	public double getAccuracyFullScore(AbstractInstance ins) {
		return 1.0;
	}

	@Override
	public double computeMacro(List<LossScore> scores) {
		//throw new RuntimeException("xxxxxxx.....");
		//return 0;
		
		double total = 0;
		double den = 0;
		
		for (LossScore sc : scores) {
			//LossScoreHamm sc2 = (LossScoreHamm)sc;
			total += sc.getVal();
			den += 1;
		}
		
		double avg = total / den;
		return avg;
		
	}
}
