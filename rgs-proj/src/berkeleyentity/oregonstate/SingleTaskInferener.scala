package berkeleyentity.oregonstate

object SingleTaskInferener {
  
  
  
  def unaryFactorInference(jointExample: AceJointTaskExample, weight: Array[Double], useGold: Boolean, useLossAugment: Boolean): SearchState = {
    
    val output = new Array[Int](jointExample.totalSize);
    
    for (i <- 0 until jointExample.totalSize) {
      //val sglIdx = jointExample.getSingleTaskIndex(i);
      val variable = jointExample.getVariableGivenIndex(i);
      var bestValIdx = if (!useGold && !useLossAugment) { // regular
        getBestValue(variable, weight);
      } else if (useGold && !useLossAugment) { // gold
        getCorrectBestValue(variable, weight);
      } else if (!useGold && useLossAugment) { // lossAug
        getLossAugmentedBestValue(variable, weight);
      } else {
        throw new RuntimeException("both true!");
        -1;
      }
      
      output(i) = bestValIdx;
    }
    
    val bestState = new SearchState(output);
    bestState;
  }

  
  
  // 0 or 1
  def singleBitLoss[T](variable: IndepVariable[T], valIdx: Int): Double = {
    var loss: Double = 0;
    if (variable.noCorrectValue()) {
      loss = 0; // no correct at all ...
    } else {
    	loss = if (variable.values(valIdx).isCorrect) {
    		0;
    	} else {
    		1;
    	}
    }
    loss;
  }
  
    // return the value index!
  def getBestValue[T](variable: IndepVariable[T], wght: Array[Double]): Int = {
      var bestLbl = 0;
      var bestScore = -Double.MaxValue;
      for (l <- 0 until variable.values.size) {
        var score = variable.values(l).computeScore(wght);
        if (score > bestScore) {
          bestScore = score;
          bestLbl = l;
        }
      }
      bestLbl;
  }
  
  // return the correct value index!
  def getCorrectBestValue[T](variable: IndepVariable[T], wght: Array[Double]): Int = {
      //var bestLbl = 0;
      //var bestScore = -Double.MaxValue;
      var bestCorrectLbl = 0; // latent best
      var bestCorrectScore = -Double.MaxValue;
      for (l <- 0 until variable.values.size) {
        var score = variable.values(l).computeScore(wght);
        /*
        if (score > bestScore) {
          bestScore = score;
          bestLbl = l;
        }
        */
        if (variable.values(l).isCorrect) {
          if (score > bestCorrectScore) {
            bestCorrectScore = score;
            bestCorrectLbl = l;
          }
        }
      }
      bestCorrectLbl;
  }
  
   def getLossAugmentedBestValue[T](variable: IndepVariable[T], wght: Array[Double]): Int = {
      var bestLbl = 0;
      var bestScore = -Double.MaxValue;
      for (l <- 0 until variable.values.size) {
        var score = variable.values(l).computeScore(wght) + singleBitLoss(variable, l);
        if (score > bestScore) {
          bestScore = score;
          bestLbl = l;
        }
      }
      bestLbl;
  }
  
  
}