package berkeleyentity.randsearch;

import java.util.ArrayList;
import java.util.List;

import berkeleyentity.coref.DocumentGraph;
import berkeleyentity.coref.PairwiseIndexingFeaturizer;
import edu.illinois.cs.cogcomp.sl.core.SLProblem;
import sequence.hw.HwOutput;

public class Ace05DataSet {
	
	public final String name = "ace05";
	
	private Ace05CorefInterf binterf;

	private ArrayList<AceCorefInstance> trainInsts = null;
	private ArrayList<AceCorefInstance> testInsts = null;
	
	
	public Ace05DataSet() {
		binterf = new Ace05CorefInterf();
	}
	
	public static Ace05DataSet loadFromScratch() {
		return (new Ace05DataSet());
	}
	
	public PairwiseIndexingFeaturizer getMentionPairFeaturizer() {
		return (binterf.basicFeaturizer());
	}
	
	public List<AceCorefInstance> getTrainInstances() {
		if (trainInsts == null) {
			ArrayList<DocumentGraph> dglist = binterf.getTrainingDocGraphs();
			trainInsts = toAceInsts(dglist);
		}
		return (trainInsts);
	}

	public List<AceCorefInstance> getTestInstances() {
		if (testInsts == null) {
			ArrayList<DocumentGraph> dglist = binterf.getTestingDocGraphs();
			testInsts = toAceInsts(dglist);
		}
		return testInsts;
	}
	
	public ArrayList<AceCorefInstance> toAceInsts(ArrayList<DocumentGraph> dglist) {
		ArrayList<AceCorefInstance> ilist = new ArrayList<AceCorefInstance>();
		for (DocumentGraph dg : dglist) {
			AceCorefInstance ins = new AceCorefInstance(dg, binterf.basicFeaturizer());
			System.out.println("Ments = " + ins.size());
			ilist.add(ins);
		}
		return ilist;
	}
	
	public void clearTrainInstances() {
		if (trainInsts != null) {
			trainInsts.clear();
		}
		System.gc();
	}
	

	
	public static SLProblem ExampleListToSLProblem(List<AceCorefInstance> insts) {
		SLProblem problem = new SLProblem();
		for (int i = 0; i < insts.size(); i++) {
			HwOutput randOutput = new HwOutput(insts.get(i).size(), null);
			problem.addExample(insts.get(i), randOutput);
		}
		return problem;
	}
	
}
