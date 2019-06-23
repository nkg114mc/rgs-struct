package berkeleyentity.oregonstate

class VarValue[T](val idx: Int,
                  val value: T,
                  val feature: Array[Int],
                  val isCorrect: Boolean) {
  
   var isPruned : Boolean = false;
   var unaryScore : Double = 0;
   var cachedScore : Double = Double.NaN; // illegal score
   
   def computeScore(wght: Array[Double]): Double = {
      var result : Double = 0;
      for (idx1 <- feature) {
        result += (wght(idx1));
      }
      result;
   }
   
   def clearCachedScore() = {
      cachedScore = Double.NaN;
   }
   
   def computeScoreAndCached(wght: Array[Double]) = {
      val score = computeScore(wght);
      cachedScore = score;
      score;
   }

   def computeScoreAndCachedNonPruned(wght: Array[Double]) = {
	   if (!isPruned) {
		   val score = computeScore(wght);
		   cachedScore = score;
		   score;
	   } else {
       cachedScore = Double.NaN;
		   -Double.MaxValue
	   }
   }
  
}