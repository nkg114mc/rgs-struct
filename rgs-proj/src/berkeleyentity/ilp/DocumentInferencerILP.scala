
/*
package berkeleyentity.ilp


import berkeleyentity.coref._;
import edu.berkeley.nlp.futile.fig.basic.Indexer
import berkeleyentity.Driver;
import berkeleyentity.GUtil
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.futile.util.Counter
import scala.collection.JavaConverters._
import berkeleyentity.coref.DocumentGraph;
import berkeleyentity.coref.PairwiseScorer;

import gurobi._;
import java.io._;
import java.util.ArrayList;
import java.lang.Integer;
import java.lang.String;

class DocumentInferencerILP extends DocumentInferencer {
	def addUnregularizedStochasticGradient(docGraph: berkeleyentity.coref.DocumentGraph,pairwiseScorer: berkeleyentity.coref.PairwiseScorer,lossFcn: (berkeleyentity.coref.CorefDoc, Int, Int) ⇒ Float,gradient: Array[Float]): Unit = {
    ???
  }

	def computeLikelihood(docGraph: berkeleyentity.coref.DocumentGraph,pairwiseScorer: berkeleyentity.coref.PairwiseScorer,lossFcn: (berkeleyentity.coref.CorefDoc, Int, Int) ⇒ Float): Float = {
    ???
  }
	def finishPrintStats(): Unit = {???}
  
	def getInitialWeightVector(featureIndexer: edu.berkeley.nlp.futile.fig.basic.Indexer[String]): Array[Float] = {???}
  
  
  def viterbiDecode(docGraph: berkeleyentity.coref.DocumentGraph,pairwiseScorer: berkeleyentity.coref.PairwiseScorer): Array[Int] = {???}
  
  
  def ilpInferenceAll(docGraphs: Seq[DocumentGraph], pairwiseScorer: PairwiseScorer): Array[Array[Int]] = {
    val allPredBackptrs = new Array[Array[Int]](docGraphs.size);
    for (i <- 0 until docGraphs.size) {
      val docGraph = docGraphs(i);
      Logger.logs("Decoding " + i);
      val predBackptrs = ilpInference(docGraph, pairwiseScorer);
      allPredBackptrs(i) = predBackptrs;
    }
    allPredBackptrs;
  }
  
  def ilpInferenceAllFormClusterings(docGraphs: Seq[DocumentGraph], pairwiseScorer: PairwiseScorer): (Array[Array[Int]], Array[OrderedClustering]) = {
    val allPredBackptrs = ilpInferenceAll(docGraphs, pairwiseScorer);
    val allPredClusteringsSeq = (0 until docGraphs.size).map(i => OrderedClustering.createFromBackpointers(allPredBackptrs(i)));
    (allPredBackptrs, allPredClusteringsSeq.toArray)
  }
  
  
  def ilpInference(docGraph: DocumentGraph, pairwiseScorer: PairwiseScorer): Array[Int] = {
    
    val (featsChart, scoresChart) = docGraph.featurizeIndexAndScoreNonPrunedUseCache(pairwiseScorer);
    val probFcn = (idx: Int) => {
      val probs = scoresChart(idx);
      GUtil.expAndNormalizeiHard(probs);
      probs;
    }
    ilpArgMax(scoresChart.size, probFcn);
    
  }
  
  /*
  def ilpArgMax(size: Int, probFcn: Int => Array[Float]): Array[Int] = {
    
    val backpointers = new Array[Int](size);
    for (i <- 0 until size) {
      val allProbs = probFcn(i);
      var bestIdx = -1;
      var bestProb = Float.NegativeInfinity;
      for (j <- 0 to i) {
        val currProb = allProbs(j);
        if (bestIdx == -1 || currProb > bestProb) {
          bestIdx = j;
          bestProb = currProb;
        }
      }
      backpointers(i) = bestIdx;
    }
    backpointers;
  }
  */
  def ilpArgMax(size: Int, probFcn: Int => Array[Float]): Array[Int] = {
    val backpointers = new Array[Int](size);
    for (i <- 0 until size) {
      val allProbs = probFcn(i);
      backpointers(i) = ilpGorubiInferece(i, allProbs);
    }
    backpointers;
  }
  
  def ilpGorubiInferece(curId: Int, allProbs: Array[Float]) : Int = {

		  val env = new GRBEnv();
		  val model = new GRBModel(env);
		  val vars = new ArrayList[GRBVar]();//new GRBVar(curId + 1);
		  /////////////////////////////////////////
      
      // Add variables
		  for (j <- 0 to curId) {
			  val currProb = allProbs(j);
			  val vName = "V_" + String.valueOf(j);
			  val newVar = model.addVar(0.0, 1.0, currProb, GRB.BINARY, vName);
        vars.add(newVar);
		  }
      
       model.update();

      // Add constraints
      val expr = new GRBLinExpr();
      val varsArr = vars.toArray(new Array[GRBVar](0));
      expr.addTerms(null, varsArr);
      val constraintName = "Constr_Val";
      model.addConstr(expr, GRB.EQUAL, 1.0, constraintName);

      model.set(GRB.IntAttr.ModelSense, -1);
      
      model.update();


      // Optimize model
      model.optimize();
      

      var bestIdx = -1;
      var bestProb = Double.NegativeInfinity;
      for (j2 <- 0 until varsArr.length) {
        
        val vari = varsArr(j2);
        val idx: Int = parseVariableIndex(vari.get(GRB.StringAttr.VarName));
        val currProb = vari.get(GRB.DoubleAttr.X);
        println("var(" + idx + ") = " + currProb);
        if (bestIdx == -1 || currProb > bestProb) {
          bestIdx = idx;
          bestProb = currProb;
        }
      }
      bestIdx;
  }
  
  def parseVariableIndex(varName: String) : Int = {
    val strArr = varName.split("_");
    val id = Integer.parseInt(strArr(1));
    id;
  }

}
 */