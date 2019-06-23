package berkeleyentity.oregonstate

import scala.collection.mutable.HashSet;
import scala.collection.mutable.ArrayBuffer;

import berkeleyentity.oregonstate.pruner.StaticDomainPruner
import berkeleyentity.oregonstate.pruner.DomainElement


class IndepVariable[VarType](val values: Array[VarValue[VarType]],
                             val correcValues: Array[VarValue[VarType]],
                             var currentValue: VarValue[VarType]) {
  
  val noCorrect = noCorrectValue();
  
  def domainSize(): Int = {
    values.size;
  }
  
  // whether the corect values and value properties is consistent
  def checkConsistency() {
    
  }
  
  def noCorrectValue(): Boolean = {
    var crrctCnt = 0;
    for (v <- values) {
      if (v.isCorrect) crrctCnt += 1;
    }
    (crrctCnt <= 0);
  }
  
  def correctValueNumber(): Int = {
    correcValues.size;
  }
  
  
  def getAllValueIndices(): Array[Int] = {
    var vals = new ArrayBuffer[Int]();
    for (i <- 0 until values.length) {
      vals += i;
    }
    vals.toArray;
  }
  
  def getCorrectValueIndices(): Array[Int] = {
    var crrtVals = new ArrayBuffer[Int]();
    for (i <- 0 until values.length) {
      if (values(i).isCorrect) crrtVals += i;
    }
    crrtVals.toArray;
  }
  
  
  def getAllNonPruningValueIndices(): Array[Int] = {
    var vals = new ArrayBuffer[Int]();
    for (i <- 0 until values.length) {
      if (!values(i).isPruned) { vals += i; }
    }
    vals.toArray;
  }
  def getCorrectNonPruningValueIndices(): Array[Int] = {
    var crrtVals = new ArrayBuffer[Int]();
    for (i <- 0 until values.length) {
      if ((!values(i).isPruned) && (values(i).isCorrect)) crrtVals += i;
    }
    crrtVals.toArray;
  }
  def getSortedNonPruningValueIndices(): Array[Int] = { // sorted indices according unary scores
    val valueElements = new ArrayBuffer[DomainElement]();
    for (i <- 0 until values.length) {
      if (!values(i).isPruned) {
        val ve = new DomainElement(i, values(i).unaryScore);
        valueElements += ve;
      }
    }

    // sort!
    val sortv = (valueElements.toSeq.sortWith(_.rankingWeight > _.rankingWeight)).toArray;
    //for (decs <- sortc) {
    //  println("c: " + decs.score + " " + decs.isCorrect);
    //}
    var vals = new ArrayBuffer[Int]();
    for (j <- 0 until sortv.size ) {
      val idx = sortv(j).vIndex;
      vals += idx;
    }
    vals.toArray;
  }
  
  /*
  def isCorrect(): Boolean = {
    val correValSet = (new HashSet[VarValue[VarType]]()) ++ (correcValues);
    (correValSet.contains(currentValue));
  }
  */
  
  def isCorrectValue(v: VarValue[VarType]): Boolean = {
    val correValSet = (new HashSet[VarValue[VarType]]()) ++ (correcValues);
    (correValSet.contains(v));
  }
  
  // return the value index!
  def getBestValue(wght: Array[Double]): Int = {
      var bestLbl = -1;
      var bestScore = -Double.MaxValue;
      for (l <- 0 until values.size) {
        var score = values(l).computeScore(wght);
        if (score > bestScore) {
          bestScore = score;
          bestLbl = l;
        }
      }
      bestLbl;
  }
  
  // return the correct value index!
  def getCorrectBestValue(wght: Array[Double]): Int = {
      var bestLbl = -1;
      var bestScore = -Double.MaxValue;
      var bestCorrectLbl = -1; // latent best
      var bestCorrectScore = -Double.MaxValue;
      for (l <- 0 until values.size) {
        var score = values(l).computeScore(wght);
        if (score > bestScore) {
          bestScore = score;
          bestLbl = l;
        }
        if (values(l).isCorrect) {
          if (score > bestCorrectScore) {
            bestCorrectScore = score;
            bestCorrectLbl = l;
          }
        }
      }
      bestCorrectLbl;
  }
  
}