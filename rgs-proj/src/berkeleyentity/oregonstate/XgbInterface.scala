package berkeleyentity.oregonstate

import java.io.File
import java.util

import scala.collection.mutable

import ml.dmlc.xgboost4j.java.{DMatrix => JDMatrix}
import ml.dmlc.xgboost4j.java.example.util.DataLoader
import ml.dmlc.xgboost4j.scala.{XGBoost, DMatrix}
import ml.dmlc.xgboost4j.scala.Booster
import scala.collection.mutable.HashSet
import berkeleyentity.coref.DocumentGraph
import berkeleyentity.xgb.XgbMatrixBuilder
import berkeleyentity.coref.PairwiseIndexingFeaturizer

object XgbInterface {
  
  def predictOneDocBooster(featurizer: PairwiseIndexingFeaturizer, docGraph: DocumentGraph, booster: Booster): Array[Int] = {
    val featsChart = docGraph.featurizeIndexNonPrunedUseCache(featurizer);
    val decisions = CorefTesting.extractDeciExmlOneDoc(docGraph);
    
    val result = new Array[Int](decisions.length);
    for (i <- 0 until decisions.length) {
      result(i) = predictDecision(decisions(i), booster);
      val crr = decisions(i).isCorrect(result(i));
    }
    result;
  }
  
  // return best value 
  def predictDecision(decision: CorefDecisionExample, booster: Booster): Int = {
    
    val (oneInstanMtrx, wdt) = XgbMatrixBuilder.corefExmpToDMatrix(Seq(decision), -1, false);
    val predicts2 = booster.predict(oneInstanMtrx);
    
    var bestLbl = -1;
		var bestScore = -Double.MaxValue;
		for (j <- 0 until decision.values.length) {
			val l = decision.values(j);
			var score = predicts2(j)(0);
			if (score > bestScore) {
				bestScore = score;
				bestLbl = j;
			}
		}
		decision.values(bestLbl);
  }
}

class XgbInterface {

  def runQuickLearning(trainExs: Seq[CorefDecisionExample], testExs: Seq[CorefDecisionExample]) : Booster = {
    
    val (trainMtrx, trWidth) = XgbMatrixBuilder.corefExmpToDMatrix(trainExs, -1, true);
    val (testMtrx, tstWidth) = XgbMatrixBuilder.corefExmpToDMatrix(testExs, trWidth, true);
	  
    //// train
	  val bstr = performLearningGvienTrainTestDMatrix(trainMtrx, testMtrx);
	  bstr;
	}

  
  def performLearningGvienTrainTestDMatrix(trainMax: DMatrix, testMax: DMatrix): Booster = {
    
    //// train
	  
    println("Trainset size: " + trainMax.rowNum);
	  println("Testset size: " + testMax.rowNum);
    
    val params = new mutable.HashMap[String, Any]()
    val round = 370
    //params += "distribution" -> "bernoulli"
    params += "eta" -> 0.1
    params += "max_depth" -> 100
    params += "silent" -> 0
    //params += "colsample_bytree" -> 0.9
    //params += "min_child_weight" -> 10
    params += "objective" -> "rank:pairwise"
    params += "eval_metric" -> "pre@1"
    params += "nthread" -> 4
      
    val watches = new mutable.HashMap[String, DMatrix]
    watches += "train" -> trainMax
    watches += "test" -> testMax


    // train a model
    val booster = XGBoost.train(trainMax, params.toMap, round, watches.toMap)
    
    //val bestScore = booster.
    //val bestIteration = model.best_iteration + 1   // note that xgboost start building tree with index 0
    //print("best_score: %s" % best_score)
    //print("opmital # of trees: %s" % best_iteration)
    
    // predict
    val predicts = booster.predict(testMax)
    // save model to model path
    val file = new File("./model")
    if (!file.exists()) {
      file.mkdirs()
    }
    booster.saveModel(file.getAbsolutePath + "/xgb-coref.model")
    // dump model
    //booster.getModelDump(file.getAbsolutePath + "/dump.raw.txt", true)
    // dump model with feature map
    //booster.getModelDump(file.getAbsolutePath + "/featmap.txt", true)
    // save dmatrix into binary buffer
    trainMax.saveBinary(file.getAbsolutePath + "/dtrain.buffer")
    testMax.saveBinary(file.getAbsolutePath + "/dtest.buffer")
    
    booster;
  }

}
