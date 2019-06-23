package elearning;

import java.util.ArrayList;

import elearning.einfer.SearchStateScoringFunction;

public abstract class AbstractRegressionLearner {
	
	//public abstract WeightVector regressionTrain(ArrayList<RegressionInstance> regrDataIter, int featLen, int iterNum);//(Instances dataset);
	public abstract SearchStateScoringFunction regressionTrain(ArrayList<RegressionInstance> regrDataIter, int featLen, int iterNum);

}
