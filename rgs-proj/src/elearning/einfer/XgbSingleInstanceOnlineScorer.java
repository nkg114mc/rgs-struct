package elearning.einfer;

import java.util.HashMap;

import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearning.AbstractRegressionLearner;
import elearning.XgbRegressionLearner;
import elearnnew.SamplingELearning;
import general.AbstractInstance;
import init.RandomStateGenerator;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;
import search.GreedySearcher;

public class XgbSingleInstanceOnlineScorer extends SearchStateScoringFunction {
	
	
	RandomStateGenerator randomGenr;
	WeightVector cost_weight;
	GreedySearcher gsearcher; 

	AbstractRegressionLearner regressionTrainer;
	int iteration;
	
	// only cached on instance
	Booster cachedModel = null;
	AbstractInstance cachedInst = null;
	int updtCnt = 0;
	
	public XgbSingleInstanceOnlineScorer(RandomStateGenerator rndg,
			WeightVector cost, 
			GreedySearcher gschr, 

			AbstractRegressionLearner regtnr,
			int iter) {
		
		randomGenr = rndg;
		cost_weight = cost;
		gsearcher = gschr; 

		regressionTrainer = regtnr;
		iteration = iter;
		
		updtCnt = 0;
	}
	
	public Booster trainForSingleInstance(AbstractInstance ainst) {
		
		if (ainst.equals(cachedInst)) {
			// do nothing
			//System.out.println("=================================> use cahced model!");
		} else {
			
			if (cachedModel != null) {
				cachedModel.dispose();
			}
			
			updtCnt++;
			if (updtCnt % 100 == 0) {
				System.out.println("Scored " + updtCnt + " instances...");
			}
			
			// update cache
			Booster instanceBooster = SamplingELearning.workOnOneInstance(randomGenr, ainst, cost_weight, gsearcher, regressionTrainer, iteration);
			cachedInst = ainst;
			cachedModel = instanceBooster;
		}
		return cachedModel;
	}

	@Override
	public double getScoring(AbstractInstance ainst, HashMap<Integer, Double> feature) {
		try {

			DMatrix mx = XgbRegressionLearner.createSingleMatrix(feature, false);

			Booster booster = trainForSingleInstance(ainst);
			
			float[][] predicts = booster.predict(mx);
			float ret = predicts[0][0];
			
			//// release memory
			mx.dispose();
			
			
			return ret;

		} catch (XGBoostError e) {
			e.printStackTrace();
		}

		return -9999;
	}

	@Override
	public Object getModel() {
		return null;
	}

}
