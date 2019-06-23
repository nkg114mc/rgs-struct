package elearning.einfer;

import java.util.HashMap;

import elearning.XgbRegressionLearner;
import general.AbstractInstance;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;


public class XgbSearchStateSingleInstanceScorer extends SearchStateScoringFunction {
	
	HashMap<AbstractInstance, Booster> instModels;
	
	public XgbSearchStateSingleInstanceScorer(HashMap<AbstractInstance, Booster>  mdMap) {
		instModels = mdMap;
	}

	@Override
	public double getScoring(AbstractInstance ainst, HashMap<Integer, Double> feature) {
		try {

			DMatrix mx = XgbRegressionLearner.createSingleMatrix(feature, false);

			Booster booster = instModels.get(ainst);
			if (booster == null) {
				throw new RuntimeException("No model found for current instance!!!");
			}
			
			float[][] predicts = booster.predict(mx);
			return predicts[0][0];

		} catch (XGBoostError e) {
			e.printStackTrace();
		}

		return -9999;
	}

	@Override
	public Object getModel() {
		return instModels;
	}

}


