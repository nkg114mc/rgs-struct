package berkeleyentity.oregonstate.pruner

import scala.collection.mutable.ArrayBuffer

import berkeleyentity.coref.CorefPrunerFolds
import berkeleyentity.coref.PairwiseScorer
import berkeleyentity.oregonstate.IndepVariable
import berkeleyentity.oregonstate.VarValue
import berkeleyentity.oregonstate.AceMultiTaskExample
import berkeleyentity.oregonstate.AceJointTaskExample
import berkeleyentity.oregonstate.QueryWikiValue

/*
class DecisionElement(val dIndex: Int,
                      val initVIdx: Int,
                      val rankScore: Double,
                      val isCrr: Boolean) {
  
}


class SyntheticPruner(val weight: Array[Double]) extends StaticDomainPruner {

  def pruneDomainIndepBatch(testStructs: Seq[AceMultiTaskExample], topK1: Int, topK2: Int, topK3: Int) {
    pruneDomainAndVariableBatchSynthetic(testStructs, topK1, topK2, topK3);
  }

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
  
  /////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////

  def variablePrunning[T](variables: Array[IndepVariable[T]], wght: Array[Double]) {
    
    val crrInitVariables = new ArrayBuffer[DecisionElement]();
    val wngInitVariables = new ArrayBuffer[DecisionElement]();

	  for (i <- 0 until variables.length) {
		  val vrbl = variables(i);
		  val varValues = vrbl.values;

		  val valueElements = new ArrayBuffer[DomainElement]();
		  for (j <- 0 until varValues.size) {
			  val score = varValues(j).computeScore(wght);
			  val ve = new DomainElement(j, score);
			  valueElements += ve;
			  varValues(j).unaryScore = score;
		  }

		  // sort!
		  val sortv = (valueElements.toSeq.sortWith(_.rankingWeight > _.rankingWeight)).toArray;
      val initValue = sortv(0);
      val rankSc = if (sortv.length > 1) {
        sortv(0).rankingWeight - sortv(1).rankingWeight; 
      } else {
        -Double.MaxValue; // only one value, can not be wrong, so will lastly considered
      }
      if (varValues(initValue.vIndex).isCorrect) {
        crrInitVariables += (new DecisionElement(i, initValue.vIndex, rankSc, true));
      } else {
        wngInitVariables += (new DecisionElement(i, initValue.vIndex, rankSc, false));
      }
		  //for (decs <- sortc) {
		  //  println("c: " + decs.score + " " + decs.isCorrect);
		  //}

		  //val nonPrunedValuesNumber = if (sortv.size > topK) topK else sortv.size
		  //for (j <- 0 until nonPrunedValuesNumber) {
		  //  val idx = sortv(j).vIndex;
		  //   varValues(idx).isPruned = false; // keep top K values! 
		  //}
	  }
    
    ////////////////////////////////
    ////////////////////////////////
    ////////////////////////////////
    
    // prune some correct values
    val sortedVariables = (crrInitVariables.toSeq.sortWith(_.rankScore > _.rankScore)).toArray;
    var nmen = (variables.size * 0.5).toInt; //40;//30;
    if (sortedVariables.length < nmen) nmen = sortedVariables.length;
    for (k <- 0 until nmen) {
      val idx = sortedVariables(k).dIndex;
      val vidx = sortedVariables(k).initVIdx;
      
      
      val varValues = variables(idx).values;
      for (j <- 0 until varValues.size) {
        varValues(j).isPruned = true;
        if (j == vidx) {
          varValues(j).isPruned = false; // only leave a initial value
        }
      }
      
    }
  }
  
  
  def pruneDomainAndVariableBatchSynthetic(testStructs: Seq[AceMultiTaskExample], topK1: Int, topK2: Int, topK3: Int) {
    
    
    // special process?
    var topKCoref = topK1;
    var topKNer = topK2;
    var topKWiki = topK3
    
	  for (ex <- testStructs) {

      val corefVars = ex.corefOutput.variables;
		  for (i <- 0 until corefVars.size) {
			  variableUnaryScoring[Int](corefVars(i).values, weight, topKCoref);
		  }
      variablePrunning[Int](corefVars, weight);
      
      
		  val nerVars = ex.nerOutput.variables;
		  for (i <- 0 until nerVars.size) {
			  variableUnaryScoring[String](nerVars(i).values, weight, topKNer);
		  }
      variablePrunning[String](nerVars, weight);
      
      
		  val wikiVars = ex.wikiOutput.variables;
		  for (i <- 0 until wikiVars.size) {
			  variableUnaryScoring[QueryWikiValue](wikiVars(i).values, weight, topKWiki);
		  }
      variablePrunning[QueryWikiValue](wikiVars, weight);
      

	  }
  }
}
*/


//// No pruner

// does not do pruning at all
class NonePruner(val weight: Array[Double]) extends StaticDomainPruner {

  def pruneDomainIndepBatch(testStructs: Seq[AceMultiTaskExample], topK1: Int, topK2: Int, topK3: Int) {
    pruneDomainAndVariableBatch(testStructs, topK1, topK2, topK3);
  }

  override def pruneDomainOneExmp(ex: AceJointTaskExample) {
    /*
    val corefVars = ex.g//ex.corefOutput.variables;
		  for (i <- 0 until corefVars.size) {
			  val varValues = corefVars(i).values;
			  for (j <- 0 until varValues.size) {
				  varValues(j).isPruned = false;
				  varValues(j).unaryScore = 0;
			  }
		  }

		  val nerVars = ex.nerOutput.variables;
		  for (i <- 0 until nerVars.size) {
			  val varValues = nerVars(i).values;
			  for (j <- 0 until varValues.size) {
				  varValues(j).isPruned = false;
				  varValues(j).unaryScore = 0;
			  }
		  }

		  val wikiVars = ex.wikiOutput.variables;
		  for (i <- 0 until wikiVars.size) {
			  val varValues = wikiVars(i).values;
			  for (j <- 0 until varValues.size) {
				  varValues(j).isPruned = false;
				  varValues(j).unaryScore = 0;
			  }
		  }
		  */
  }
  
  override def scoreCorefValue(cvalue: VarValue[Int]): Double = {
    0;
  };
  
  override def scoreNerValue(nvalue: VarValue[String]): Double = {
		0;
  };
  
  override def scoreWikiValue(wvalue: VarValue[QueryWikiValue]): Double = {
    0;
  };
  
  def pruneDomainAndVariableBatch(testStructs: Seq[AceMultiTaskExample], topK1: Int, topK2: Int, topK3: Int) {
	  for (ex <- testStructs) {
		  val corefVars = ex.corefOutput.variables;
		  for (i <- 0 until corefVars.size) {
			  val varValues = corefVars(i).values;
			  for (j <- 0 until varValues.size) {
				  varValues(j).isPruned = false;
				  varValues(j).unaryScore = 0;
			  }
		  }

		  val nerVars = ex.nerOutput.variables;
		  for (i <- 0 until nerVars.size) {
			  val varValues = nerVars(i).values;
			  for (j <- 0 until varValues.size) {
				  varValues(j).isPruned = false;
				  varValues(j).unaryScore = 0;
			  }
		  }

		  val wikiVars = ex.wikiOutput.variables;
		  for (i <- 0 until wikiVars.size) {
			  val varValues = wikiVars(i).values;
			  for (j <- 0 until varValues.size) {
				  varValues(j).isPruned = false;
				  varValues(j).unaryScore = 0;
			  }
		  }

	  }
  }

}