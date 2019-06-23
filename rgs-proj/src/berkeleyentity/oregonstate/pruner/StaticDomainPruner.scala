package berkeleyentity.oregonstate.pruner

import scala.collection.mutable.ArrayBuffer
import scala.math._
import berkeleyentity.coref.CorefPruner
import berkeleyentity.coref.PairwiseScorer
import berkeleyentity.coref.CorefPrunerFolds
import berkeleyentity.GUtil
import java.io.PrintWriter
import java.io.BufferedReader
import java.io.FileReader
import berkeleyentity.ilp.HistgramRecord
import edu.berkeley.nlp.futile.fig.basic.Indexer
import scala.collection.mutable.HashMap
import berkeleyentity.oregonstate.SingleTaskStructTesting
import berkeleyentity.oregonstate.IndepVariable
import berkeleyentity.oregonstate.VarValue
import berkeleyentity.oregonstate.AceMultiTaskExample
import berkeleyentity.oregonstate.AceJointTaskExample
import berkeleyentity.oregonstate.QueryWikiValue



/**
 * Prune the domain of each variable in the structure
 */

class DomainElement(val vIndex: Int,//val value: VarValue[T],
                    val rankingWeight: Double) {
  
}

class StaticDomainPruner { // base pruner class for all pruners
  
  def pruneDomainBatch(testStructs: Seq[AceJointTaskExample]) {
    for (ex <- testStructs) {
      pruneDomainOneExmp(ex);
    }
  }

  def pruneDomainOneExmp(ex: AceJointTaskExample): Unit = { throw new RuntimeException("Unimplemented!"); };
  
  // scoring a variable value
  def scoreCorefValue(cvalue: VarValue[Int]): Double = { throw new RuntimeException("Unimplemented!"); 0;};
  def scoreNerValue(nvalue: VarValue[String]): Double = { throw new RuntimeException("Unimplemented!"); 0;};
  def scoreWikiValue(wvalue: VarValue[QueryWikiValue]): Double = { throw new RuntimeException("Unimplemented!"); 0;};
  
  
  // return the value index
  def getCorefVarBestValue(cvar: IndepVariable[Int], ignorePruning: Boolean): Int = {
    if (ignorePruning) {
      return getVarBestValueIgnorePruning[Int](cvar, scoreCorefValue);
    } else {
      return getVarBestValue[Int](cvar, scoreCorefValue);
    }
  }
  def getNerVarBestValue(nvar: IndepVariable[String], ignorePruning: Boolean): Int = {
    if (ignorePruning) {
      return getVarBestValueIgnorePruning[String](nvar, scoreNerValue);
    } else {
      return getVarBestValue[String](nvar, scoreNerValue);
    }
  }
  def getWikiVarBestValue(wvar: IndepVariable[QueryWikiValue], ignorePruning: Boolean): Int = {
    if (ignorePruning) {
      return getVarBestValueIgnorePruning[QueryWikiValue](wvar, scoreWikiValue);
    } else {
      return getVarBestValue[QueryWikiValue](wvar, scoreWikiValue);
    }
  }
  
    // return the value index!
  def getVarBestValue[T](variable: IndepVariable[T], scoreFunc: VarValue[T] => Double): Int = {
      var bestLbl = -1;
      var bestScore = -Double.MaxValue;
      for (l <- 0 until variable.values.size) {
    	  if (!variable.values(l).isPruned) {
    		  var score = scoreFunc(variable.values(l));
    		  if (score > bestScore) {
    			  bestScore = score;
    			  bestLbl = l;
    		  }
    	  }
      }
      bestLbl;
  }
  
  def getVarBestValueIgnorePruning[T](variable: IndepVariable[T], scoreFunc: VarValue[T] => Double): Int = {
      var bestLbl = -1;
      var bestScore = -Double.MaxValue;
      for (l <- 0 until variable.values.size) {
    	  //if (!variable.values(l).isPruned) {
    		  var score = scoreFunc(variable.values(l));
    		  if (score > bestScore) {
    			  bestScore = score;
    			  bestLbl = l;
    		  }
    	  //}
      }
      bestLbl;
  }
  
  
  def getCorefVarCorrectBestValue(cvar: IndepVariable[Int]): Int = {
    getVarCorrectBestValue[Int](cvar, scoreCorefValue);
  }
  def getNerVarCorrectBestValue(nvar: IndepVariable[String]): Int = {
    getVarCorrectBestValue[String](nvar, scoreNerValue);
  }
  def getWikiVarCorrectBestValue(wvar: IndepVariable[QueryWikiValue]): Int = {
    getVarCorrectBestValue[QueryWikiValue](wvar, scoreWikiValue);
  }
  
  // return the correct value index!
  def getVarCorrectBestValue[T](variable: IndepVariable[T], scoreFunc: VarValue[T] => Double): Int = {
      var bestLbl = -1;
      var bestScore = -Double.MaxValue;
      var bestCorrectLbl = -1; // latent best
      var bestCorrectScore = -Double.MaxValue;
      for (l <- 0 until variable.values.size) {
        var score = scoreFunc(variable.values(l));
        if (score > bestScore) {
          bestScore = score;
          bestLbl = l;
        }
        if (variable.values(l).isCorrect) {
          if (score > bestCorrectScore) {
            bestCorrectScore = score;
            bestCorrectLbl = l;
          }
        }
      }
      bestCorrectLbl;
  }

}

class BerkeleyCorefDomainPruner(val berkCorefPruner: CorefPruner) extends StaticDomainPruner {
  
  //def getBerkeleyCorefPruner(strategy: String, trainFolder: String) = {
  //  berkCorefPruner = CorefPruner.buildPrunerArguments(strategy, trainFolder, -1);
  //  berkCorefPruner;
  //}

  // use berkeley coref pruner
  def pruneDomainIndepWithBerkCorefPrunerBatch(testStructs: Seq[AceMultiTaskExample]) {
	  
    if (berkCorefPruner == null) {
      throw new RuntimeException("Berkeley pruner is not constructed yet!");
    }
    /*
    // start pruning
    for (ex <- testStructs) {
    	doBerkCorefPrune(ex, berkCorefPruner);
    	val nerVars = ex.nerOutput.variables;
    	for (i <- 0 until nerVars.size) {
    		variableUnaryScoring[String](nerVars(i).values, weight, topK2);
    	}
    	val wikiVars = ex.wikiOutput.variables;
    	for (i <- 0 until wikiVars.size) {
    		variableUnaryScoring[QueryWikiValue](wikiVars(i).values, weight, topK3);
    	}
	  }
	  */
  }
    
  def doBerkCorefPrune(ex: AceMultiTaskExample, berkPruner: CorefPruner) {

	  val foldPruner = berkPruner.asInstanceOf[CorefPrunerFolds];
	  //CorefPrunerFolds(val docsToFolds: HashMap[UID,Int],
	  //                   val foldModels: ArrayBuffer[PairwiseScorer],
	  //                   val logPruningThreshold: Double) 

	  val doc = ex.docGraph;
    val threshold = foldPruner.logPruningThreshold;
	  val prunerScorer = if (foldPruner.docsToFolds.contains(doc.corefDoc.rawDoc.uid)) {
		  foldPruner.foldModels(foldPruner.docsToFolds(doc.corefDoc.rawDoc.uid));
	  } else {
		  foldPruner.foldModels.head;
	  }

	  val corefVars = ex.corefOutput.variables;
	  for (i <- 0 until corefVars.size) {
		  //doCorefPruneVariable(i, corefVars(i).values, prunerScorer, foldPruner.logPruningThreshold);
      val scores = (0 to i).map(j => prunerScorer.score(doc, i, j, false));
      val bestIdx = GUtil.argMaxIdxFloat(scores);
      
      val varValues = corefVars(i).values;
      for (jj <- 0 until varValues.length) {
    	  val j = varValues(jj).value;
        varValues(jj).unaryScore = scores(j);
    	  if (scores(j) < scores(bestIdx) + threshold) {
    		  //prunedEdges(i)(j) = true;
    		  //cachedFeats(i)(j) = emptyIntArray;
    		  varValues(jj).isPruned = true;
    	  } else {
          varValues(jj).isPruned = false;
        }
      }
      
      /*
      for (j <- 0 to i) {
        if (scores(j) < scores(bestIdx) + threshold) {
          //prunedEdges(i)(j) = true;
          //cachedFeats(i)(j) = emptyIntArray;
          varValues(i).isPruned = true;
      varValues(i).unaryScore = score;
        }
      }
      */
	  }
  }
  
  //def doCorefPruneVariable(idx: Int, varValues: Array[VarValue[Int]], model: PairwiseScorer, logPruningThreshold: Double) {
  //  val i = idx;
  //}
  
}


class DocElement(val docName: String,
                 val rankingWeight: Double) {
  
}


object StaticDomainPruner {
  
  def savePrunerWeight(weight: Array[Double], filePath: String) {
    val writer = new PrintWriter(filePath);
    writer.println(weight.length);
    for (i <- 0 until weight.length) {
      writer.println(i + " " + weight(i));
    }
    writer.close();
  }
  

  def loadPrunerWeight(weight: Array[Double], filePath: String) {
	  var lineCnt = 0;
    val br = new BufferedReader(new FileReader(filePath));
		  var line: String = "";
      line = br.readLine();
		  while (line != null) {
        
        if (lineCnt == 0) { // length
          val len = Integer.parseInt(line);
          if (len != weight.length) {
            throw new RuntimeException("Unequal length of weights: " + len + " " + weight.length);
          }
        } else { // others
          val tks = line.split("\\s+");
          val idx = tks(0).toInt;
          val value = tks(1).toDouble;
          weight(idx) = value;
        }
        
        lineCnt += 1;
			  line = br.readLine();
		  }
  }

  def loadPrunerWeightUnknownLen(filePath: String) = {
    var weight: Array[Double] = null;
	  var lineCnt = 0;
    val br = new BufferedReader(new FileReader(filePath));
    var line: String = "";
    line = br.readLine();
    while (line != null) {

    	if (lineCnt == 0) { // length
    		val len = Integer.parseInt(line);
    		if (len < 0) {
    			throw new RuntimeException("Invalide weights length: " + len);
    		}
    		weight = new Array[Double](len);
    	} else { // others
    		val tks = line.split("\\s+");
    		val idx = tks(0).toInt;
    		val value = tks(1).toDouble;
    		weight(idx) = value;
    	}

    	lineCnt += 1;
    	line = br.readLine();
    }
    weight;
  }
	//// utils ////
	def mentionCount(trains: Seq[AceMultiTaskExample],
			             tests: Seq[AceMultiTaskExample]) {

		val trExs = new ArrayBuffer[DocElement]();
		val tsExs = new ArrayBuffer[DocElement]();
		for (extr <- trains) {
			trExs += (new DocElement(extr.docGraph.corefDoc.rawDoc.docID, extr.numMentions.toDouble));
		}

		for (exts <- tests) {
			tsExs += (new DocElement(exts.docGraph.corefDoc.rawDoc.docID, exts.numMentions.toDouble));
		}
    
    // sort!
    val sortTr = (trExs.toSeq.sortWith(_.rankingWeight > _.rankingWeight)).toArray;
    val sortTs = (tsExs.toSeq.sortWith(_.rankingWeight > _.rankingWeight)).toArray;
    
    println("==== Train Mention Count ====");
    for (i <- 0 until sortTr.length) {
      //println(sortTr(i).docName + ": " + sortTr(i).rankingWeight.toInt);
      if (sortTr(i).rankingWeight > 200) {
        println("cp train/" + sortTr(i).docName + " train_large/");
      }
    }
    println("==== Test Mention Count ====");
    for (j <- 0 until sortTs.length) {
      if (sortTs(j).rankingWeight > 200) {
        println("cp test/" + sortTs(j).docName + " test_large/");
      }
      //println(sortTs(j).docName + ": " + sortTs(j).rankingWeight.toInt);
    }  
    
	}
  
    def mentionCount2(trains: Seq[AceMultiTaskExample],
                   tests: Seq[AceMultiTaskExample]) {

    	println("==== Train Mention Count in Order ====");
    	for (extr <- trains) {
    		println(extr.docGraph.corefDoc.rawDoc.docID + ": " + extr.numMentions.toDouble);
    	}
    	println("==== Test Mention Count in Order====");
    	for (exts <- tests) {
    		println(exts.docGraph.corefDoc.rawDoc.docID + ": " + exts.numMentions.toDouble);
    	}
  }
  
	def noCorrectCount(exs: Seq[AceMultiTaskExample]) {
		var corefCnt = 0;
		var nerCnt = 0;
		var wikiCnt = 0;
		var total = 0;
		var corefTotalValues = 0;
		var nerTotalValues = 0;
		var wikiTotalValues = 0;

		for (ex <- exs) {
			total += ex.docGraph.getMentions().size;
			for (cvar <- ex.corefOutput.variables) {
				val pruned = cvar.getCorrectNonPruningValueIndices();
				val remain = cvar.getAllNonPruningValueIndices();
				corefTotalValues += remain.length;
				if (pruned.length <= 0) {
					corefCnt += 1;
				}
			}
			for (nvar <- ex.nerOutput.variables) {
				val pruned = nvar.getCorrectNonPruningValueIndices();
				val remain = nvar.getAllNonPruningValueIndices();
				nerTotalValues += remain.length;
				if (pruned.length <= 0) {
					nerCnt += 1;
				}
			}
			for (wvar <- ex.wikiOutput.variables) {
				val pruned = wvar.getCorrectNonPruningValueIndices();
				val remain = wvar.getAllNonPruningValueIndices();
				wikiTotalValues += remain.length;
				if (pruned.length <= 0) {
					wikiCnt += 1;
				}
			}
		}

		println("Coref_no_correct: " + corefCnt  + "/" + total+ "/" + corefTotalValues);
		println("Ner_no_correct: " + nerCnt  + "/" + total+ "/" +nerTotalValues);
		println("Wiki_no_correct: " + wikiCnt  + "/" + total+ "/" + wikiTotalValues);
	}
	def checkPruningLoss(trains: Seq[AceMultiTaskExample],
                       tests: Seq[AceMultiTaskExample]) {
		println("==== Train Pruning Loss ====");
		noCorrectCount(trains);
		println("==== Test Pruning Loss ====");
		noCorrectCount(tests)
	}
	
	def noCorrectCountJoint(exs: Seq[AceJointTaskExample]) {
		var corefCnt = 0;
		var nerCnt = 0;
		var wikiCnt = 0;
		var total = 0;
		var corefTotalValues = 0;
		var nerTotalValues = 0;
		var wikiTotalValues = 0;

		for (ex <- exs) {
			total += ex.docGraph.getMentions().size;
			val corefVars = ex.corefVars;
			for (cvar <- corefVars) {
				val pruned = cvar.getCorrectNonPruningValueIndices();
				val remain = cvar.getAllNonPruningValueIndices();
				corefTotalValues += remain.length;
				if (pruned.length <= 0) {
					corefCnt += 1;
				}
			}
			val nerVars = ex.nerVars;
			for (nvar <- nerVars) {
				val pruned = nvar.getCorrectNonPruningValueIndices();
				val remain = nvar.getAllNonPruningValueIndices();
				nerTotalValues += remain.length;
				if (pruned.length <= 0) {
					nerCnt += 1;
				}
			}
			val wikiVars = ex.wikiVars;
			for (wvar <- wikiVars) {
				val pruned = wvar.getCorrectNonPruningValueIndices();
				val remain = wvar.getAllNonPruningValueIndices();
				wikiTotalValues += remain.length;
				if (pruned.length <= 0) {
					wikiCnt += 1;
				}
			}
		}

		println("Coref_no_correct: " + corefCnt  + "/" + total+ "/" + corefTotalValues);
		println("Ner_no_correct: " + nerCnt  + "/" + total+ "/" +nerTotalValues);
		println("Wiki_no_correct: " + wikiCnt  + "/" + total+ "/" + wikiTotalValues);
	}
}