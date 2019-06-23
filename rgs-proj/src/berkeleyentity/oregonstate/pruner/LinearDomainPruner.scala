package berkeleyentity.oregonstate.pruner

import scala.collection.mutable.ArrayBuffer

import berkeleyentity.oregonstate.SingleTaskStructTesting
import berkeleyentity.oregonstate.VarValue
import berkeleyentity.oregonstate.AceMultiTaskExample
import berkeleyentity.ilp.HistgramRecord
import berkeleyentity.oregonstate.AceJointTaskExample
import berkeleyentity.oregonstate.QueryWikiValue
import edu.berkeley.nlp.futile.fig.basic.Indexer


class LinearDomainPruner(val weight: Array[Double],
                         val topK1: Int,
                         val topK2: Int,
                         val topK3: Int) extends StaticDomainPruner {
  
	// special process?
	val topKCoref = topK1;
	val topKNer = topK2;
	val topKWiki = topK3
  
  def pruneDomainIndepBatch(testStructs: Seq[AceMultiTaskExample], topK1: Int, topK2: Int, topK3: Int) {
    for (ex <- testStructs) {
      pruneDomainIndepOneExmp(ex, weight, topK1, topK2, topK3);
    }
  }

  def pruneDomainIndepOneExmp(ex: AceMultiTaskExample, wght: Array[Double], topK1: Int, topK2: Int, topK3: Int) {

    //if (ex.numMentions >= 100) {
    //  topKCoref = topK1 - 1;
    //  topKNer = topK2 - 1;
    //}
    
    val corefVars = ex.corefOutput.variables;
    for (i <- 0 until corefVars.size) {
      val t2 = (corefVars.size.toDouble * 0.1).toInt;
      val topc = if (t2 <= topKCoref) {
        topKCoref
      } else {
        t2;
      }
      variableUnaryScoring[Int](corefVars(i).values, wght, topc);// topKCoref);
    }
    val nerVars = ex.nerOutput.variables;
    for (i <- 0 until nerVars.size) {
      variableUnaryScoring[String](nerVars(i).values, wght, topKNer);
    }
    val wikiVars = ex.wikiOutput.variables;
    for (i <- 0 until wikiVars.size) {
      variableUnaryScoring[QueryWikiValue](wikiVars(i).values, wght, topKWiki);
    }
  }

/*
  def pruneDomainBatch(testStructs: Seq[AceJointTaskExample], topK1: Int, topK2: Int, topK3: Int) {
    for (ex <- testStructs) {
      pruneDomainOneExmp(ex, weight, topK1, topK2, topK3);
    }
  }
*/
  //def pruneDomainOneExmp(ex: AceJointTaskExample, wght: Array[Double], topK1: Int, topK2: Int, topK3: Int) {
  override def pruneDomainOneExmp(ex: AceJointTaskExample) {
    for (i <- 0 until ex.corefVars.size) {
      variableUnaryScoring[Int](ex.corefVars(i).values, weight, topK1);
    }
    for (i <- 0 until ex.nerVars.size) {
      variableUnaryScoring[String](ex.nerVars(i).values, weight, topK2);
    }
    for (i <- 0 until ex.wikiVars.size) {
      variableUnaryScoring[QueryWikiValue](ex.wikiVars(i).values, weight, topK3);
    }
  }
  
  override def scoreCorefValue(cvalue: VarValue[Int]): Double = {
    cvalue.computeScore(weight);
  };
  
  override def scoreNerValue(nvalue: VarValue[String]): Double = {
		nvalue.computeScore(weight);
  };
  
  override def scoreWikiValue(wvalue: VarValue[QueryWikiValue]): Double = {
    wvalue.computeScore(weight);
  };
  

  // do pruning!
  def variableUnaryScoring[T](varValues: Array[VarValue[T]], wght: Array[Double], topK: Int) {
    val valueElements = new ArrayBuffer[DomainElement]();
    for (i <- 0 until varValues.size) {
      val score = varValues(i).computeScore(wght);
      val ve = new DomainElement(i, score);
      valueElements += ve;
      varValues(i).isPruned = true;
      varValues(i).unaryScore = score;
    }
    
    // sort!
    val sortv = (valueElements.toSeq.sortWith(_.rankingWeight > _.rankingWeight)).toArray;
    //for (decs <- sortc) {
    //  println("c: " + decs.score + " " + decs.isCorrect);
    //}

    val nonPrunedValuesNumber = if (sortv.size > topK) topK else sortv.size
    for (j <- 0 until nonPrunedValuesNumber) {
      val idx = sortv(j).vIndex;
      varValues(idx).isPruned = false; // keep top K values! 
    }
  }

}

object LinearDomainPruner {

  def runIndependentTaskTraining(trains: Seq[AceMultiTaskExample],
                               tests: Seq[AceMultiTaskExample],
      featIndexer: Indexer[String]) = {
      
      //trainExs: Seq[AceJointTaskExample],
       //                          testExs: Seq[AceJointTaskExample],
       //                          featIndexer: Indexer[String]) = { //,
                                 //goldWikification: HashMap[String, DocWikiAnnots]) = {
    
    val trainStructs = (new ArrayBuffer[AceMultiTaskExample]()) ++ (trains);//(trainExs.map(struct => { struct.toMultiTaskStructs() }));
    val testStructs = (new ArrayBuffer[AceMultiTaskExample]()) ++ (tests);//(testExs.map(struct => { struct.toMultiTaskStructs() }));

    // Learning!
    val weight = SingleTaskStructTesting.structurePerceptrion(trainStructs, featIndexer, testStructs);

    // test as oracle?
    //testStructuraAllTaskOracle(testStructs, featIndexer.size());
    
    // test!
    val histgram = new HistgramRecord();
    SingleTaskStructTesting.testStructuralAllTasks(testStructs, weight, histgram);
    histgram.printHistgram();
    
    // evaluate
    //SingleTaskStructTesting.evaluateAceStructs(testStructs, goldWikification);
    
    // return 
    weight;
  }
}
