package elearning.einfer;

import java.util.HashMap;

import elearning.XgbRegressionLearner;
import general.AbstractInstance;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;


public class XgbSearchStateOneInstanceScorer extends SearchStateScoringFunction {
	
	AbstractInstance inst;
	Booster booster;
	
	public XgbSearchStateOneInstanceScorer(AbstractInstance ins, Booster md) {
		inst = ins;
		booster = md;
	}

	@Override
	public double getScoring(AbstractInstance ainst, HashMap<Integer, Double> feature) {
		try {
			
			//if (booster == null) {
			if (inst != ainst) {
				throw new RuntimeException("No model found for current instance!!!");
			}

			DMatrix mx = XgbRegressionLearner.createSingleMatrix(feature, false);

			float[][] predicts = booster.predict(mx);
			return predicts[0][0];

		} catch (XGBoostError e) {
			e.printStackTrace();
		}

		return -9999;
	}

	@Override
	public Object getModel() {
		return booster;
	}

}