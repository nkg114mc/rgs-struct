package berkeleyentity.oregonstate

import scala.collection.immutable.HashSet
import java.util.Arrays

class FeatureSparsity(val featLength: Int) {
  
  val fireCount = new Array[Int](featLength);
  
  def init() {
    Arrays.fill(fireCount, 0); // clear zero
  }
  
  def addOneSample(featVec: Array[Int]) {
    val idxSet = (new HashSet[Int]()) ++ featVec;
    for (i <- idxSet) {
      fireCount(i) += 1;
    }
  }
  
  def printHistogram() {
    val sorted = (fireCount.toSeq.sortWith(_ > _)).toArray;
    val maxFeatCnt = 10;
    for (i <- 0 until maxFeatCnt) {
      println("(" + i + "): " + sorted(i));
    }
    
  }

}