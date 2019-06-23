package multilabel.evaluation;

import java.util.ArrayList;
import java.util.List;

import experiment.TestingAcc;
import multilabel.instance.Example;
import sequence.hw.HwInstance;

public class MultiLabelEvaluator {
	
	Scorer[] scorers;
	
	public  MultiLabelEvaluator() {
		scorers = new Scorer[4]; // 4
		scorers[0] = new HammingAccuracyScorer();
		scorers[1] = new ExampleF1Scorer();
		scorers[2] = new ExampleAccuracyScorer();
		scorers[3] = new ExactMatchScorer();
	}
	
	public  MultiLabelEvaluator(String[] scorerNames) {
		
	}

	
	public List<TestingAcc> evaluationDataSet(String dsname, ArrayList<Example> exs) {
		
		double[] sc = new double[scorers.length];
		for (int i = 0; i < scorers.length; i++) {
			sc[i] = scorers[i].getAccuracyBatch(exs);
		}
		
		ArrayList<TestingAcc> accs = new ArrayList<TestingAcc>();

		System.out.println("==== Evaluating "+ dsname +" ====");
		for (int i = 0; i < scorers.length; i++) {
			System.out.println(scorers[i].name() + ": " + sc[i]);
			accs.add(new TestingAcc(scorers[i].name(), sc[i]));
		}
		System.out.println("===== Done Evaluation =====");
		
		return accs;
	}
	
	
	
	public List<TestingAcc> evaluationHwInstance(String dsname, List<HwInstance> exs) {
		
		double[] sc = new double[scorers.length];
		for (int i = 0; i < scorers.length; i++) {
			sc[i] = scorers[i].getAccuracyBatchHw(exs);
		}
		
		ArrayList<TestingAcc> accs = new ArrayList<TestingAcc>();

		System.out.println("==== Evaluating "+ dsname +" ====");
		for (int i = 0; i < scorers.length; i++) {
			System.out.println(scorers[i].name() + ": " + sc[i]);
			accs.add(new TestingAcc(scorers[i].name(), sc[i]));
		}
		System.out.println("===== Done Evaluation =====");
		
		return accs;
	}
}
