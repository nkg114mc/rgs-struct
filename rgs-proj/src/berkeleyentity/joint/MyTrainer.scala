package berkeleyentity.joint

import scala.Array.canBuildFrom
import scala.collection.JavaConverters.asScalaBufferConverter
import scala.collection.mutable.HashSet
import edu.berkeley.nlp.futile.math.CachingDifferentiableFunction
import edu.berkeley.nlp.futile.math.LBFGSMinimizer
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.futile.fig.basic.SysInfoUtils
import java.util.Arrays

class MyTrainer[T] {

    
  //var inferenceNanos = 0L;
  //var adagradNanos = 0L;
  
  def train(trainExs: Seq[T],
            computer: LikelihoodAndGradientComputer[T],
            numFeats: Int,
            eta: Float,
            reg: Float,
            batchSize: Int,
            numItrs: Int): Array[Float] = {
    train2(trainExs, computer, numFeats, 0.01F, 10F, batchSize, numItrs);
  }

  def train2(trainExs: Seq[T],
                   computer: LikelihoodAndGradientComputer[T],
                   numFeats: Int,
                   eta: Float,
                   lambda: Float,
                   batchSize: Int,
                   numItrs: Int): Array[Float] = {
//    val weights = Array.fill(pairwiseIndexingFeaturizer.featureIndexer.size)(0.0);
    val weights = Array.fill(numFeats)(0.0F);
    val reusableGradientArray = Array.fill(numFeats)(0.0F);
    val diagGt = Array.fill(numFeats)(0.0F);
    for (i <- 0 until numItrs) {
      Logger.logss("ITERATION " + i);
      val startTime = System.nanoTime();
      Logger.startTrack("Computing gradient");
      learnWeight(trainExs, computer,
                  weights,
                  numFeats,
                  eta,
                  lambda);
      
      
      Logger.endTrack();
      Logger.logss("NONZERO WEIGHTS: " + weights.foldRight(0)((weight, count) => if (Math.abs(weight) > 1e-15) count + 1 else count));
      Logger.logss("WEIGHT VECTOR NORM: " + weights.foldRight(0.0)((weight, norm) => norm + weight * weight));

    }
    weights
  }
  
/*
  def computeObjectiveL1R(trainExs: Seq[T],
                          computer: LikelihoodAndGradientComputer[T],
                          weights: Array[Float],
                          lambda: Float): Float = {
    var objective = computeLikelihood(trainExs, computer, weights);
    for (weight <- weights) {
      objective -= lambda * Math.abs(weight);
    }
    objective;
  }

  def computeLikelihood(trainExs: Seq[T],
                        computer: LikelihoodAndGradientComputer[T],
                        weights: Array[Float]): Float = {
    (trainExs.foldRight(0.0)((ex, likelihood) => likelihood + computer.computeLogLikelihood(ex, weights))).toFloat;
  }
*/
  
  def learnWeight(exs: Seq[T],
                 computer: LikelihoodAndGradientComputer[T],
                  weights: Array[Float],
                  numFeats: Int,
                  eta: Float,
                  lambda: Float) {
    
    val reusableGradientArray = Array.fill(numFeats)(0.0F);
    
    for (ex <- exs) {
    	Arrays.fill(reusableGradientArray, 0.0F);
    	computer.addUnregularizedStochasticGradient(ex, weights, reusableGradientArray);

      // update weight
      
    	//var l1norm = getL1Norm(currentWeight);
    	for (i2 <- 0 until weights.length) {
    		//var regularizerNum: Double = Math.max(0, b);
    		//var regularizerDen: Double = Math.max(0, b);
    		var reg: Float = 1.0F - (eta * lambda);
        //println(reg + ", " + eta + ", " + lambda);
    		var curWeightVal = weights(i2) * reg;
    	  weights(i2) = curWeightVal - (reusableGradientArray(i2) * eta);
    	  //currentWeight(i2) += (gradient(i2) * eta);
    	}
    }
  }
  
/*
  def takeAdagradStepL1R(exs: Seq[T],
                         computer: LikelihoodAndGradientComputer[T],
                         weights: Array[Float],
                         reusableGradientArray: Array[Float],
                         diagGt: Array[Float],
                         eta: Float,
                         lambda: Float) {
    Arrays.fill(reusableGradientArray, 0.0F);
    var nanoTime = System.nanoTime();
    for (ex <- exs) {
      computer.addUnregularizedStochasticGradient(ex, weights, reusableGradientArray);
    }
    inferenceNanos += (System.nanoTime() - nanoTime);
    nanoTime = System.nanoTime();
    // Precompute this so dividing by batch size is a multiply and not a divide
    val batchSizeMultiplier = 1.0F/exs.size;
    var i = 0;
    while (i < reusableGradientArray.size) {
      val xti = weights(i);
      // N.B. We negate the gradient here because the Adagrad formulas are all for minimizing
      // and we're trying to maximize, so think of it as minimizing the negative of the objective
      // which has the opposite gradient
      // Equation (25) in http://www.cs.berkeley.edu/~jduchi/projects/DuchiHaSi10.pdf
      // eta is the step size, lambda is the regularization
      val gti = -reusableGradientArray(i) * batchSizeMultiplier;
      // Update diagGt
      diagGt(i) += gti * gti;
      val Htii = 1F + Math.sqrt(diagGt(i)).toFloat;
      // Avoid divisions at all costs...
      val etaOverHtii = eta / Htii;
      val newXti = xti - etaOverHtii * gti;
      weights(i) = Math.signum(newXti) * Math.max(0, Math.abs(newXti) - lambda * etaOverHtii);
      i += 1;
    }
    adagradNanos += (System.nanoTime() - nanoTime);
  }
*/
  
}