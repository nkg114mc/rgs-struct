package berkeleyentity.oregonstate

import java.io.PrintWriter
import java.time.Duration
import java.time.Instant

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import scala.util.Random
import scala.util.control.Breaks.break
import scala.util.control.Breaks.breakable

import berkeleyentity.MyTimeCounter
import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.futile.util.Counter
import java.util.Comparator
import berkeleyentity.oregonstate.pruner.StaticDomainPruner
import berkeleyentity.oregonstate.pruner.DynamicPruner



class SearchState(val output: Array[Int]) {
  
  var cachedPredScore: Double = 0;
  var cachedTrueLoss: Double  = 0;
  var cachedFeatureVector: HashMap[Int, Double] = null;
  var actClass: String = "?";
  
  var slotMusk: Array[Int] = Array.fill[Int](output.length)(1);  // only the non-zero slot is allowed to apply value-changing (default: all 1)
  
  def numMentions() = {
    val nMents = output.length / 3;
    nMents;
  }
  
  def size() = {
    output.length;
  }
  
  /*
  def getHashkey() {
    if (hashkey == 0) {
      hashkey = 
    }
  }*/
  
  def computeHashkey(zobristHashKeys: Array[Array[Int]]) {
    var result: Int = 0;
    for (i <- 0 until output.length) {
      val j = output(i);
      if (j >= 0) {
        result = (result ^ zobristHashKeys(i)(j));
      }
    }
    result;
  }
  
  def isEqualOutput(anotherState: SearchState): Boolean = {
    var isequal = true;
    if (anotherState.size() != output.length) {
      throw new RuntimeException("Not a equvalent length: " + anotherState.size() + "!=" + output.length);
    }
    
    breakable {
      for (i<- 0 until output.length) {
        if (anotherState.output(i) != output(i)) {
          isequal = false;
          break;
        }
      }
    }
    
    isequal;
  }
  
  def copyFrom(anotherState: SearchState) {
    if (anotherState.size() != output.length) {
      throw new RuntimeException("Not a equvalent length: " + anotherState.size() + "!=" + output.length);
    }
    for (i<- 0 until output.length) {
       output(i) = anotherState.output(i);
    }
  }
  
  // return a new state that is the same as the current one
  def getSelfCopy() = {
    SearchState.copyState(this);
  }
  
  // print
  def printState() {
    val len = output.size / 3;
    for (i <- 0 until output.size) {
      print(output(i) + ", ");
    }
    println();
  }
}

object SearchState {
  /**
   * Note: This copy does not copy the feature vector!
   */
	def copyState(src: SearchState) : SearchState = {

			val outputCopy = new Array[Int](src.size()); 
			Array.copy(src.output, 0, outputCopy , 0, src.size());

			val newState = new SearchState(outputCopy);

			newState.cachedPredScore = src.cachedPredScore;
			newState.cachedTrueLoss = src.cachedTrueLoss;
			newState.cachedFeatureVector = null;//src.cachedFeatureVector;

			val muskCopy = new Array[Int](src.slotMusk.length);
			Array.copy(src.slotMusk, 0, muskCopy, 0, src.slotMusk.length);
      newState.slotMusk = muskCopy;

			newState;
	}
}

class SearchAction(val idx: Int, val newValIdx: Int) {
  var score: Double = 0;
  var trueAcc: Double = 0;
  var isCorrect: Boolean = false;

  var actClass = "?";

  var undoIndex: Int = -1;
  var undoOldValue: Int = -1;
  
  //var sparseFeature: HashMap[Int, Double] = null;
  
  override def toString() = {
    val str = "Change variable(" + idx + ") from value("+undoOldValue+") to value(" + newValIdx + ")!";
    str;
  }
}



class ZobristKeys(val maxNmention: Int, val maxNvalue: Int) {
  
  val zobristHashKeys = initialZobrise(maxNmention, maxNvalue);

  def initialZobrise(maxNment: Int, maxNval: Int) = {
    val zhkeys = Array.ofDim[Int](maxNment, maxNval);
    for (i <- 0 until maxNment) {
      for (j <- 0 until maxNval) {
        zhkeys(i)(j) = Random.nextInt(Integer.MAX_VALUE);
      }
    }
    zhkeys;
  }
}

class SearchHashTable(val zrbkeys: ZobristKeys) {
  
  val zobristHashKeys = zrbkeys.zobristHashKeys;//initialZobrise(500);
  val indexedTable = new HashMap[Int, ArrayBuffer[SearchState]]();

  var totalSize = 0;
  
  def computeIndex(state: SearchState): Int = {
    var result: Int = 0;
    for (i <- 0 until state.output.length) {
      val j = state.output(i);
      //if (j >= 0) {
        result = (result ^ zobristHashKeys(i)(j));
      //}
    }
    result;
  }
  
  def computeIndexIncreamental(oldStateIndex: Int, slotIdx: Int, oldValIdx: Int, newValIdx: Int): Int = {
    var result: Int = oldStateIndex;
    result = (result ^ zobristHashKeys(slotIdx)(oldValIdx));
    result = (result ^ zobristHashKeys(slotIdx)(newValIdx));
    result;
  }
  
  
  def insertNewInstance(state: SearchState) {
    val indexValue = computeIndex(state);
    
    val valueList = indexedTable.get(indexValue);
    if (valueList == None) {
      // insert new state
      val newList = new ArrayBuffer[SearchState]();
      newList += state;
      indexedTable += (indexValue -> newList);
    } else {
      val existList = valueList.get;
      existList += state;
      indexedTable += (indexValue -> existList);
    }
    totalSize += 1;
  }
  
  // return true or false
  def probeExistence(state: SearchState): Boolean = {
		  val indexValue = computeIndex(state);
		  val valueList = indexedTable.get(indexValue);

		  var isExist: Boolean = false;
		  if (valueList == None) {
			  isExist = false;
		  } else {
			  val existList = valueList.get;
			  breakable {
				  for (everyEntry <- existList) {
					  if (everyEntry.isEqualOutput(state)) {
						  isExist = true;
						  break;
					  }
				  }
			  }
		  }
		  isExist;
  }
  
  def probeExistenceWithAction(state: SearchState, oldIndex: Int, action: SearchAction): Boolean = {
		  //val indexValue = computeIndex(state);
      val indexValue = computeIndexIncreamental(oldIndex, action.idx, action.undoOldValue, action.newValIdx);
		  val valueList = indexedTable.get(indexValue);

		  var isExist: Boolean = false;
		  if (valueList == None) {
			  isExist = false;
		  } else {
			  val existList = valueList.get;
			  breakable {
				  for (everyEntry <- existList) {
					  if (everyEntry.isEqualOutput(state)) {
						  isExist = true;
						  break;
					  }
				  }
			  }
		  }
		  isExist;
  }
}

/*
class GreedyTrajectoryState(val state: SearchState,
                            val stateScore: Double,
                            //val depth: Int,
                            val actionSize: Int,
                            val trueLoss: Double) {
  
}
*/

class SearchBasedLearner(val zrbkeys: ZobristKeys, val dynamicActionPruner: DynamicPruner = null) {
  
  val ignorePruningAtInit = true;
  
 // return 0-1 loss according to ground truth
  def getTrueAcc(jointExample: AceJointTaskExample, state: SearchState): Double = {
    val zeroOneLoss = jointExample.getWeightedZeroOneError(state.output);//jointExample.getZeroOneError(state.output);
    val acc = jointExample.getErrFullScore() - zeroOneLoss;
    acc.toDouble;
  }
  
  def getTrueLosss(jointExample: AceJointTaskExample, state: SearchState): Double = {
    jointExample.getWeightedZeroOneError(state.output);//jointExample.getZeroOneError(state.output).toDouble;
  }
  
  def getTrueLossIncreamentalByAction(jointExample: AceJointTaskExample, oldTrueLoss: Double, action: SearchAction): Double = {
    var newZeroOneLoss = oldTrueLoss.toDouble;
    val idx = action.idx;
    val ivariable = jointExample.getVariableGivenIndex(idx);
    if (!ivariable.values(action.undoOldValue).isCorrect) {
      newZeroOneLoss -= jointExample.errorWeight(idx);
    }
    if (!ivariable.values(action.newValIdx).isCorrect) {
      newZeroOneLoss += jointExample.errorWeight(idx);;
    }
    //(jointExample.totalSize - newZeroOneLoss);
    newZeroOneLoss.toDouble;
  }
  
  def getTrueAccuracyIncreamentalByAction(jointExample: AceJointTaskExample, oldTrueAcc: Double, action: SearchAction): Double = {
    var newAcc = oldTrueAcc.toDouble;
    val idx = action.idx;
    val ivariable = jointExample.getVariableGivenIndex(idx);
    if (ivariable.values(action.undoOldValue).isCorrect) {
      newAcc -= jointExample.errorWeight(idx);
    }
    if (ivariable.values(action.newValIdx).isCorrect) {
      newAcc += jointExample.errorWeight(idx);
    }
    (newAcc).toDouble;
  }

  ///////////////////////////////////////////////////
  
  def printActionVerbose(jointExample: AceJointTaskExample, action: SearchAction) = {
    val str = "Change variable(" + action.idx + ") from value("+ jointExample.getVariableGivenIndex(action.idx).values(action.undoOldValue).value +") to value(" + jointExample.getVariableGivenIndex(action.idx).values(action.newValIdx).value + ")!";
    str;
  }
  
  def almostEqualDouble(v1: Double, v2: Double, err: Double) = {
    ((v1 - v2).abs <= err);
  }
  def printScoreCheck(v1: Double, v2: Double, v3: Double) = {
    val err = 0.0000001;
    //if (v1 == v2 && v2 == v3 && v1 == v3) {
    if (almostEqualDouble(v1, v2, err) && almostEqualDouble(v2, v3, err) && almostEqualDouble(v1, v3, err)) {
      true;
    } else {
      throw new RuntimeException("(" + v1 + ", " + v2 + ", " + v3 + ")");
      false;
    }
  }
  def printScoreCheckTwo(v1: Double, v2: Double) = {
    val err = 0.0000001;
    if (almostEqualDouble(v1, v2, err)) {
      true;
    } else {
      throw new RuntimeException("(" + v1 + ", " + v2 + ")");
      false;
    }
  }

// A faster version of GreedySearch
  
  def hillClimbing(jointExample: AceJointTaskExample, initState: SearchState, weight: Array[Double], useGold: Boolean, isLossAugment: Boolean) = {

    val verbose = false;
    
    if (useGold && isLossAugment) {
      // invalid!
      throw new RuntimeException("Can not perform inference with both Gold and LossAugment!");
    } else if (!useGold && !isLossAugment) {
      // predict truth
    } else if (!useGold && isLossAugment) {
      // loss augment
    } else if (useGold && !isLossAugment) {
      // gold
    }
    
    jointExample.updateValueScoreGivenWeight(weight);
    
    val learnRate = 0.1;
    val lambda = 1e-8;
    var updateCnt: Int = 0;

    val hashTb = new SearchHashTable(zrbkeys);
    val trajectory = new Array[SearchState](2000);

    val maxDepth = jointExample.totalSize * 100;
    var depth = 0;
    var flatStep = 0;

    // initialization!
    var currentState = initState.getSelfCopy();
    val computedScore = SingleTaskStructTesting.computeScoreSparse(weight, jointExample.featurize(currentState.output));
    currentState.cachedPredScore = computedScore;
    currentState.cachedTrueLoss = getTrueAcc(jointExample, currentState);
    currentState.cachedFeatureVector = jointExample.featurize(currentState.output);
    if (useGold) {
        currentState.cachedPredScore = currentState.cachedTrueLoss;
    }
    if (isLossAugment) {
      currentState.cachedPredScore = computedScore + getTrueLosss(jointExample, currentState); // w * phi(x,y) + delta(y, y*) 
    }
    
    trajectory(0) = currentState;
    hashTb.insertNewInstance(currentState);

    // start greedy search
    var lastDepth: Int = 1;
    breakable {
      
      //val actions = actionGenerationWithMusk(jointExample, currentState, weight, useGold, true);
      val actCnt = 0;
      
    	for (step <- 1 until maxDepth) {

    	  lastDepth = step;
    		if (step % 50 == 0) {
    			//println("Greedy search depth: " + step);
    		}

    		//var siblingsCnt: Int = 0;
    		var actionCnter = new Counter[String]();

    		//val actions = actionGenerationNormal(jointExample, currentState, weight, useGold, true);
        //val actions = actionGenerationWithMusk(jointExample, currentState, weight, useGold, true);
    		val (bestAct, siblingsCnt, bestActScore) = fastFindGreedyBestAction(jointExample, currentState, hashTb, null, useGold, isLossAugment, weight); // for speedup!
    		//val (bestGoodAct, bestGoodActScore) = getBestCorrectAction(nonRepeatActions);
    		
    		//if (bestAct == null) {
    		//  println("siblingsCnt = " + siblingsCnt);
    		//}
    		
        var bestScore = bestActScore;
    		
    		//var (bestState, bestScore) = getBestChildState(siblingStates);
    		//val (bestGoodState, bestGoodScore) = getBestCorrectChildState(siblingStates);
    		//println("best score = " + bestScore );

    		// shall we continue?
    		if (siblingsCnt == 0) {
    			if (verbose) println("Stop search: no non-repreat actions!" + " depth = " + step + " branch = " + actCnt);
    			break; // no legal action!
    		}
    		if (bestScore < trajectory(step - 1).cachedPredScore) {
    			if (verbose) println("Stop search: reached peak!" + " depth = " + step + " branch = " + actCnt);
    			break; // reach the hill peak!
    		} else if (bestScore == trajectory(step - 1).cachedPredScore){
    			// reach a flat range, if this continues for 5 more step, break (local optimal)
    			flatStep += 1;
    			if (flatStep >= 5) {
    				if (verbose) println("Stop search: No increasing for 5 steps!" + " depth = " + step + " branch = " + actCnt);
    				break;
    			}
    		}
    		
    		
    		// execute the bestAction
    		var bestState = performActionGetNewState(jointExample, currentState, bestAct);
    		bestState.cachedPredScore = bestAct.score;
    		bestState.cachedTrueLoss = bestAct.trueAcc;
    		bestState.actClass = bestAct.actClass;
    		if (useGold) {
    			bestState.cachedPredScore = bestAct.trueAcc;
    		}
    		
    		
    		// store to the trajectory
    		currentState =  bestState;
    		trajectory(step) = currentState;
    		hashTb.insertNewInstance(currentState); // remember the extended state
    	}

    	//println("Greedy search depth: " + lastDepth);
    } // breakable

    
    
    // return current state
    currentState;
	}
  
  def fastFindGreedyBestAction(jointExample: AceJointTaskExample, 
                               currentState: SearchState, 
                               hashTb: SearchHashTable,
                               genActions: ArrayBuffer[SearchAction], 
                               useGold: Boolean,
                               isLossAugment: Boolean,
                               weight: Array[Double]): (SearchAction, Int, Double) = {
    // action generation
    val actions = actionGenerationWithMusk(jointExample, currentState, weight, useGold, true);
    
    var bestAction: SearchAction = null;
    var bestScore: Double = -Double.MaxValue;
    var nonRepeatActCnt: Int = 0;
    //val nonRepeatActions = new ArrayBuffer[SearchAction]();
    
    val corefRightLink = jointExample.computeCorefRightLink(currentState.output);
    val oldScore = jointExample.getScoreFast(currentState.output);
    
    val oldStateHash = hashTb.computeIndex(currentState);
    
    val oldTrueAcc = getTrueAcc(jointExample, currentState);
    val oldTrueLoss = getTrueLosss(jointExample, currentState);
      

      // group actions
      
      //println("Branch factor = " + actions.size);
      for (act <- actions) {
         
    	  val incrementalScore = jointExample.getScoreFastIncrementalByAction(currentState.output, act, corefRightLink);
    	  val newLoss = getTrueLossIncreamentalByAction(jointExample, oldTrueLoss, act);

    	  val lossAugmentScore: Double = if (isLossAugment) {
    	    //oldTrueLoss + increamentalLoss;
    	    newLoss;
    	  } else {
    	    0;
    	  }
    	  
    	  val predScore4 = oldScore + incrementalScore + lossAugmentScore;
        
    	  
    			// do this action
    		  performAction(jointExample, currentState, act);
    		  
    		  //val newStateHash = hashTb.computeIndex(currentState);
    			//val increamentalStateHash = hashTb.computeIndexIncreamental(oldStateHash, act.idx, act.undoOldValue, act.newValIdx);
    			//printScoreCheckTwo(newStateHash.toDouble, increamentalStateHash.toDouble);

    		  
    			//val stateIsOk = (!hashTb.probeExistence(currentState));
    		  val stateIsOk = (!hashTb.probeExistenceWithAction(currentState, oldStateHash, act));
    			if (stateIsOk) {
    			  
    			  nonRepeatActCnt += 1;

    				//println(printActionVerbose(jointExample, act));
    				//val incrementalFeature = stateFeatureIncreamentCompute(jointExample, currentState, act);
    				//val stateFeature = jointExample.featurize(newChildState.output);
    				//SearchBasedLearner.isSameFeature(stateFeature, incrementalFeature); // check!
    				//val stateFeature = incrementalFeature;

    				//val stateFeature = SearchBasedLearner.getRandomFeatureVec();
    				//println(stateFeature.size);
    				//val predScore1 = SingleTaskStructTesting.computeScoreSparse(weight, stateFeature);

    				//val predScore2 = jointExample.getScoreGivenWeight(currentState.output, weight);
    				//val predScore3 = jointExample.getScoreFast(currentState.output);
    			  
    				//printScoreCheckTwo(predScore3, predScore4);
    				//printScoreCheck(predScore1, predScore2, predScore3);
    				
    				//val trueAcc0 = getTrueLoss(jointExample, currentState);
    				val trueAcc1 = getTrueAccuracyIncreamentalByAction(jointExample, oldTrueAcc, act);
    				//printScoreCheckTwo(trueAcc0, trueAcc1);
    				
    				//val accumLoss = newLoss;//oldTrueLoss + increamentalLoss; 		  
    		    //val computedLoss = getTrueLosss(jointExample, currentState);
    	      //printScoreCheckTwo(computedLoss, accumLoss);
    				
    				val trueAcc = trueAcc1;
    				val predScore = predScore4;
    				
    				//var actClass = if (trueAcc > trajectory(step - 1).cachedTrueLoss) {
    				//	"good"
    				//} else if (trueAcc == trajectory(step - 1).cachedTrueLoss) {
    				//	"netral"
    				//} else {
    				//	"bad";
    				//}

    				//act.sparseFeature = stateFeature;
    				act.actClass = "unknown";//actClass;
    				act.score = predScore;
    				act.trueAcc = trueAcc;
    				if (useGold) {
    					act.score = trueAcc;
    				}
    				//actionCnter.incrementCount(actClass, 1.0);
    				//nonRepeatActions += act;

    				// about the state
    				//newChildState.cachedFeatureVector = stateFeature;
    				/*
    				newChildState.cachedPredScore = predScore;
    				if (useGold) {
    					newChildState.cachedPredScore = trueAcc;
    				}
    				newChildState.cachedTrueLoss = trueAcc;
    				newChildState.actClass = actClass;
    				*/
    				//siblingStates += newChildState;
    				
    				//println(useGold + " Act score = " + act.score + ", " + act.trueAcc);
    				if (act.score > bestScore) {
    				  bestAction = act;
              bestScore = act.score
    				}
    				
    			}

    			// undo action
    			cancelAction(jointExample, currentState, act);
    		}

    		//val goodCnt = actionCnter.getCount("good").toInt;
    		//val netralCnt = actionCnter.getCount("netral").toInt;
    		//val badCnt = actionCnter.getCount("bad").toInt;
    		
    		//println("branch factor = " + siblingsCnt + ", output size = " + jointExample.totalSize);
    		//println("(good:" + goodCnt + ", netral:" + netralCnt + ", bad:" + badCnt + ")");
    		//println("HashTb size = " + hashTb.totalSize);
      
    (bestAction, nonRepeatActCnt, bestScore);
  }

  def checkSparsity(weight: Array[Double]) {
    var nz = 0;
    for (i <- 0 until weight.length) {
      if (weight(i) != 0) nz += 1;
    }
    println("NonZero: " + nz + " / " + weight.length);
  }
  
  // generate top-k actions rather than just the best
  
  def fastFindTopKAction(jointExample: AceJointTaskExample, 
                         currentState: SearchState, 
                         hashTb: SearchHashTable,
                         genActions: ArrayBuffer[SearchAction], 
                         topk: Int,
                         useGold: Boolean,
                         isLossAugment: Boolean,
                         weight: Array[Double],
                         depth: Int): (List[SearchAction], SearchAction, List[SearchAction], SearchAction, Int) = {
    
    //checkSparsity(weight);
    
    // action generation
    val applyPrune = (!useGold);
    val actions = actionGenerationWithMusk(jointExample, currentState, weight, useGold, true);
    
    val corefRightLink = jointExample.computeCorefRightLink(currentState.output);
    val oldScore = jointExample.getScoreFast(currentState.output);
    if (java.lang.Double.isNaN(oldScore)) {
      println("Depth = " + depth);
    }
    assert (!java.lang.Double.isNaN(oldScore));
    
    val oldStateHash = hashTb.computeIndex(currentState);
    
    val oldTrueAcc = getTrueAcc(jointExample, currentState);
    val oldTrueLoss = getTrueLosss(jointExample, currentState);
      

    // construct action beam
    val actionComparator = new ActionInversedPredComparator();
    val actionPredBeam = new ActionBeam(topk, actionComparator);
    val actionTruthBeam = new ActionBeam(topk, new ActionInversedOracleComparator());

   
    // remove repeat!
    val noRepeatActs = removeRepeatActions(jointExample, currentState, hashTb, oldStateHash, actions);
    val nonRepeatActCnt = noRepeatActs.size;

    // run pruner here!!!
    val pruningRemainActs = if (dynamicActionPruner != null) {
    	val rma = dynamicActionPruner.pruneStateAndChileActions(jointExample, currentState, noRepeatActs); // send for pruning
    	val keptCnt = rma.size;
    	val prunedCnt = noRepeatActs.size - rma.size;
    	System.out.println("Action total (" + noRepeatActs.size + "): Remain " + rma.size + " (Pruned " + (prunedCnt) + ")");
    	rma;
    } else {
      //throw new RuntimeException("Dynamic pruner not found!");
      noRepeatActs;
    }

    val extendActions = pruningRemainActs;
    //println("Branch factor = " + actions.size);
    for (act <- extendActions) {
        
        val incrementalScore = jointExample.getScoreFastIncrementalByAction(currentState.output, act, corefRightLink);
    	  val newLoss = getTrueLossIncreamentalByAction(jointExample, oldTrueLoss, act);

    	  val lossAugmentScore: Double = if (isLossAugment) {
    	    newLoss;
    	  } else {
    	    0;
    	  }
    	  
    	  val predScore4 = oldScore + incrementalScore + lossAugmentScore;
    	  assert (!java.lang.Double.isNaN(incrementalScore));
    
    	  
    		// do this action
    		performAction(jointExample, currentState, act);
    		  
    		//val newStateHash = hashTb.computeIndex(currentState);
    	  //val increamentalStateHash = hashTb.computeIndexIncreamental(oldStateHash, act.idx, act.undoOldValue, act.newValIdx);
    		//printScoreCheckTwo(newStateHash.toDouble, increamentalStateHash.toDouble);

    		  
    		//val stateIsOk = (!hashTb.probeExistence(currentState));
    		val stateIsOk = (!hashTb.probeExistenceWithAction(currentState, oldStateHash, act));
    		if (stateIsOk) {
    			  
    			  //nonRepeatActCnt += 1;

    				//println(printActionVerbose(jointExample, act));
    				//val incrementalFeature = stateFeatureIncreamentCompute(jointExample, currentState, act);
    				//val stateFeature = jointExample.featurize(newChildState.output);
    				//SearchBasedLearner.isSameFeature(stateFeature, incrementalFeature); // check!
    				//val stateFeature = incrementalFeature;

    				//val stateFeature = SearchBasedLearner.getRandomFeatureVec();
    				//println(stateFeature.size);
    				//val predScore1 = SingleTaskStructTesting.computeScoreSparse(weight, stateFeature);

    				//val predScore2 = jointExample.getScoreGivenWeight(currentState.output, weight);
    				//val predScore3 = jointExample.getScoreFast(currentState.output);
    			  
    				//printScoreCheckTwo(predScore3, predScore4);
    				//printScoreCheck(predScore1, predScore2, predScore3);
    				
    				//val trueAcc0 = getTrueAcc(jointExample, currentState);
    				val trueAcc1 = getTrueAccuracyIncreamentalByAction(jointExample, oldTrueAcc, act);
    				//printScoreCheckTwo(trueAcc0, trueAcc1);
    				
    				//val accumLoss = newLoss;//oldTrueLoss + increamentalLoss; 		  
    		    //val computedLoss = getTrueLosss(jointExample, currentState);
    	      //printScoreCheckTwo(computedLoss, accumLoss);
    				
    				val trueAcc = trueAcc1;
    				val predScore = predScore4;
   

    				//act.sparseFeature = stateFeature;
    				act.actClass = "unknown";//actClass;
    				act.score = predScore;
    				act.trueAcc = trueAcc;
 
    				// insert into beam
    				//println(useGold + " Act score = " + act.score + ", " + act.trueAcc);
    				actionPredBeam.addAndKeepTopK(act);
    				actionTruthBeam.addAndKeepTopK(act);
    		} else {
    			  //repeatActCnt += 1;
    		}

    		// undo action
    		cancelAction(jointExample, currentState, act);
    }

     
    val topkActions = actionPredBeam.getAll();
    val minPredAct = actionPredBeam.getPeek();
    
    val topkTruthActs = actionTruthBeam.getAll();
    val minGoldAct = actionTruthBeam.getPeek();
    
    //if (repeatActCnt > 0) {
      //println("Repeat Actions = " + repeatActCnt);
    //}
    
    
    (topkActions, minPredAct, topkTruthActs, minGoldAct,  nonRepeatActCnt);
  }
  
  def removeRepeatActions(jointExample: AceJointTaskExample, 
                         currentState: SearchState, 
                         hashTb: SearchHashTable,
                         oldStateHash: Int,
                         actions: ArrayBuffer[SearchAction]): ArrayBuffer[SearchAction] = {
    
    val noRepeatActions = new ArrayBuffer[SearchAction]();
    //val oldStateHash = hashTb.computeIndex(currentState);

    var nonRepeatActCnt = 0;
    var repeatActCnt = 0;
    for (act <- actions) {
      // do this action
      performAction(jointExample, currentState, act);
      val stateIsOk = (!hashTb.probeExistenceWithAction(currentState, oldStateHash, act));
      if (stateIsOk) {
        nonRepeatActCnt += 1;
        noRepeatActions += act;
      } else {
        repeatActCnt += 1;
      }
      cancelAction(jointExample, currentState, act);
    }
    noRepeatActions;
  }
  
/*

  def fastFindTopKAction(jointExample: AceJointTaskExample, 
                         currentState: SearchState, 
                         hashTb: SearchHashTable,
                         genActions: ArrayBuffer[SearchAction], 
                         topk: Int,
                         useGold: Boolean,
                         isLossAugment: Boolean,
                         weight: Array[Double]): (List[SearchAction], SearchAction, List[SearchAction], SearchAction, Int) = {
    
    //checkSparsity(weight);
    
    // action generation
    val actions = actionGenerationWithMusk(jointExample, currentState, weight, useGold, true);
    
    //var bestAction: SearchAction = null;
    //var bestScore: Double = -Double.MaxValue;
    var nonRepeatActCnt: Int = 0;
    
    val corefRightLink = jointExample.computeCorefRightLink(currentState.output);
    val oldScore = jointExample.getScoreFast(currentState.output);
    
    val oldStateHash = hashTb.computeIndex(currentState);
    
    val oldTrueAcc = getTrueAcc(jointExample, currentState);
    val oldTrueLoss = getTrueLosss(jointExample, currentState);
      

    // construct action beam
    val actionComparator = new ActionInversedPredComparator();
    val actionPredBeam = new ActionBeam(topk, actionComparator);
    val actionTruthBeam = new ActionBeam(topk, new ActionInversedOracleComparator());
    
      // group actions
      var repeatActCnt = 0;
      
      //println("Branch factor = " + actions.size);
      for (act <- actions) {
        
    	  val incrementalScore = jointExample.getScoreFastIncrementalByAction(currentState.output, act, corefRightLink);
    	  val newLoss = getTrueLossIncreamentalByAction(jointExample, oldTrueLoss, act);

    	  val lossAugmentScore: Double = if (isLossAugment) {
    	    newLoss;
    	  } else {
    	    0;
    	  }
    	  
    	  val predScore4 = oldScore + incrementalScore + lossAugmentScore;
        
    	  
    			// do this action
    		  performAction(jointExample, currentState, act);
    		  
    		  //val newStateHash = hashTb.computeIndex(currentState);
    			//val increamentalStateHash = hashTb.computeIndexIncreamental(oldStateHash, act.idx, act.undoOldValue, act.newValIdx);
    			//printScoreCheckTwo(newStateHash.toDouble, increamentalStateHash.toDouble);

    		  
    			//val stateIsOk = (!hashTb.probeExistence(currentState));
    		  val stateIsOk = (!hashTb.probeExistenceWithAction(currentState, oldStateHash, act));
    			if (stateIsOk) {
    			  
    			  nonRepeatActCnt += 1;

    				//println(printActionVerbose(jointExample, act));
    				//val incrementalFeature = stateFeatureIncreamentCompute(jointExample, currentState, act);
    				//val stateFeature = jointExample.featurize(newChildState.output);
    				//SearchBasedLearner.isSameFeature(stateFeature, incrementalFeature); // check!
    				//val stateFeature = incrementalFeature;

    				//val stateFeature = SearchBasedLearner.getRandomFeatureVec();
    				//println(stateFeature.size);
    				//val predScore1 = SingleTaskStructTesting.computeScoreSparse(weight, stateFeature);

    				//val predScore2 = jointExample.getScoreGivenWeight(currentState.output, weight);
    				//val predScore3 = jointExample.getScoreFast(currentState.output);
    			  
    				//printScoreCheckTwo(predScore3, predScore4);
    				//printScoreCheck(predScore1, predScore2, predScore3);
    				
    				//val trueAcc0 = getTrueAcc(jointExample, currentState);
    				val trueAcc1 = getTrueAccuracyIncreamentalByAction(jointExample, oldTrueAcc, act);
    				//printScoreCheckTwo(trueAcc0, trueAcc1);
    				
    				//val accumLoss = newLoss;//oldTrueLoss + increamentalLoss; 		  
    		    //val computedLoss = getTrueLosss(jointExample, currentState);
    	      //printScoreCheckTwo(computedLoss, accumLoss);
    				
    				val trueAcc = trueAcc1;
    				val predScore = predScore4;
    				
    				//var actClass = if (trueAcc > trajectory(step - 1).cachedTrueLoss) {
    				//	"good"
    				//} else if (trueAcc == trajectory(step - 1).cachedTrueLoss) {
    				//	"netral"
    				//} else {
    				//	"bad";
    				//}

    				//act.sparseFeature = stateFeature;
    				act.actClass = "unknown";//actClass;
    				act.score = predScore;
    				act.trueAcc = trueAcc;
    				//actionCnter.incrementCount(actClass, 1.0);
    				//nonRepeatActions += act;

    				// about the state
    				//newChildState.cachedFeatureVector = stateFeature;
    				//siblingStates += newChildState;
    				
    				
    				// best
    				//if (act.score > bestScore) {
    				//  bestAction = act;
            //  bestScore = act.score
    				//}
    				
    				// insert into beam
    				//println(useGold + " Act score = " + act.score + ", " + act.trueAcc);
    				actionPredBeam.addAndKeepTopK(act);
    				actionTruthBeam.addAndKeepTopK(act);
    			} else {
    			  repeatActCnt += 1;
    			}

    			// undo action
    			cancelAction(jointExample, currentState, act);
    		}

    		//val goodCnt = actionCnter.getCount("good").toInt;
    		//val netralCnt = actionCnter.getCount("netral").toInt;
    		//val badCnt = actionCnter.getCount("bad").toInt;
    		
    		//println("branch factor = " + siblingsCnt + ", output size = " + jointExample.totalSize);
    		//println("(good:" + goodCnt + ", netral:" + netralCnt + ", bad:" + badCnt + ")");
    		//println("HashTb size = " + hashTb.totalSize);
     
    val topkActions = actionPredBeam.getAll();
    val minPredAct = actionPredBeam.getPeek();
    
    val topkTruthActs = actionTruthBeam.getAll();
    val minGoldAct = actionTruthBeam.getPeek();
    
    if (repeatActCnt > 0) {
      //println("Repeat Actions = " + repeatActCnt);
    }
    
    
    (topkActions, minPredAct, topkTruthActs, minGoldAct,  nonRepeatActCnt);
  }

 */
  
  def beamSearch(jointExample: AceJointTaskExample, 
                 initState: SearchState, 
                 beamSize: Int, 
                 weight: Array[Double], 
                 useGold: Boolean, 
                 isLossAugment: Boolean) = {

    val verbose = false;
    
    if (useGold && isLossAugment) {
      // invalid!
      throw new RuntimeException("Can not perform inference with both Gold and LossAugment!");
    } else if (!useGold && !isLossAugment) {
      // predict truth
    } else if (!useGold && isLossAugment) {
      // loss augment
    } else if (useGold && !isLossAugment) {
      // gold
    }
    
    jointExample.updateValueScoreGivenWeight(weight);
    
    val learnRate = 0.1;
    val lambda = 1e-8;
    var updateCnt: Int = 0;

    val hashTb = new SearchHashTable(zrbkeys);
    val trajectory = new Array[SearchState](32767);
    val beam = new SearchBeam(beamSize, new StatePredComparator);
    val maxDepth = jointExample.totalSize * 100;
    var depth = 0;
    var flatStep = 0;

    // initialization!
    var currentState = initState.getSelfCopy();
    val computedScore = SingleTaskStructTesting.computeScoreSparse(weight, jointExample.featurize(currentState.output));
    currentState.cachedPredScore = computedScore;
    currentState.cachedTrueLoss = getTrueAcc(jointExample, currentState);
    if (isLossAugment) {
      currentState.cachedPredScore = computedScore + getTrueLosss(jointExample, currentState); // w * phi(x,y) + delta(y, y*) 
    }
    
    trajectory(0) = currentState;
    hashTb.insertNewInstance(currentState);

    // start greedy search
    var lastDepth: Int = 1;
    breakable {
      
      //val actions = actionGenerationWithMusk(jointExample, currentState, weight, useGold, true);
      val actCnt = 0;
      
    	for (step <- 1 until maxDepth) {

    	  lastDepth = step;
    		if (step % 1 == 0) {
    		  //println(step + ": " + currentState.cachedPredScore + ", " + currentState.cachedTrueLoss);
    		}
    		if (step >= trajectory.length) {
    		  //for (stp <- 0 until (step - 1)) {
    		  //  println(trajectory(stp).cachedTrueLoss + " " + trajectory(stp).cachedPredScore);
    		  //}
    		  println(trajectory(step - 1).cachedTrueLoss + " " + trajectory(step - 1).cachedPredScore);
    		  println("useGold = " + useGold);
    		  println("isLossAugment = " + isLossAugment);
    		  throw new RuntimeException("Depth limit exceed!");
    		}

    		//var siblingsCnt: Int = 0;
    		var actionCnter = new Counter[String]();

    		//val actions = actionGenerationNormal(jointExample, currentState, weight, useGold, true);
        //val actions = actionGenerationWithMusk(jointExample, currentState, weight, useGold, true);
    		//val (bestAct, siblingsCnt, bestActScore) = fastFindGreedyBestAction(jointExample, currentState, hashTb, null, useGold, isLossAugment, weight); // for speedup!
    		val (topkActs, minRankedAct, topTrueActs, minTrueAct, siblingsCnt) = fastFindTopKAction(jointExample, currentState, hashTb, null, beam.beamSize, useGold, isLossAugment, weight, step);
    		//val (bestGoodAct, bestGoodActScore) = getBestCorrectAction(nonRepeatActions);

    		// insert new states
    		val betterActionsThanBeam = {
    		  /*
    			val (tailBeamState) = beam.getWorstInBeam();
    			println("beam size = " + beam.size());
    			println("worst score = " + minBeamScore);
    			val betterActionsThanBeam = topkActs.filter{ act => (act.score >= minBeamScore); }

    			println("better size = " + betterActionsThanBeam.size);
*/
    			topkActs;
    		}
    		
    		//println("better size = " + betterActionsThanBeam.size);
    		for (tkAct <- betterActionsThanBeam) {
    			var newState = performActionGetNewState(jointExample, currentState, tkAct);
    			newState.cachedPredScore = tkAct.score;
    			newState.cachedTrueLoss = tkAct.trueAcc;
    			newState.actClass = tkAct.actClass;
    			beam.insert(newState);
    		}
    		
    		if (betterActionsThanBeam.size == 0) {
    		  if (verbose) println("Stop search: no non-repreat actions!" + " depth = " + step + " useGold = " + useGold);
    			break; // no legal action!
    		}
    		
    		
        val bestState = beam.getBestInBeam();
        if (bestState == null) {
          throw new RuntimeException("No best in beam! + ex_size = " + jointExample.totalSize);
        }
        
        //val bestScore = if (useGold) bestState.cachedTrueLoss else bestState.cachedPredScore
        //val lastStepScore = if (useGold) trajectory(step - 1).cachedTrueLoss else trajectory(step - 1).cachedPredScore
        val bestScore = bestState.cachedPredScore;
        val lastStepScore = trajectory(step - 1).cachedPredScore;


    		// shall we continue?
    		if (siblingsCnt == 0) {
    			//if (verbose) println("Stop search: no non-repreat actions!" + " depth = " + step + " branch = " + actCnt + " useGold = " + useGold);
    		  if (verbose) println("Stop search: no non-repreat actions!" + " depth = " + step + " useGold = " + useGold);
    			break; // no legal action!
    		}
    		if (bestScore < lastStepScore){
    			//if (verbose) println("Stop search: reached peak!" + " depth = " + step + " branch = " + actCnt + " useGold = " + useGold);
    		  if (verbose) println("Stop search: reached peak!" + " depth = " + step + " useGold = " + useGold);
    			break; // reach the hill peak!
    		} else if (bestScore == lastStepScore){
    			// reach a flat range, if this continues for 5 more step, break (local optimal)
    			flatStep += 1;
    			if (flatStep >= 5) {//jointExample.totalSize) {
    				//if (verbose) println("Stop search: No increasing for 5 steps!" + " depth = " + step + " branch = " + actCnt + " useGold = " + useGold);
    			  if (verbose) println("Stop search: No increasing for 5 steps!" + " depth = " + step + " useGold = " + useGold);
    				break;
    			}
    		}

    		// drop the state out of beam size
    		beam.keepTopKOnly();
    			
    		// pop the best
    		currentState = beam.popBest();//bestState;

    		// store to the trajectory
    		trajectory(step) = currentState;
    		hashTb.insertNewInstance(currentState); // remember the extended state

    	}
		
    } // breakable

    // return current state
    currentState;
  }
  
  def actionGenerationNormal(jointExample: AceJointTaskExample, state: SearchState, weight: Array[Double],
                             asGold: Boolean, applyPrune: Boolean) = {
    
    val actionList = new ArrayBuffer[SearchAction]();
    
    for (i <- 0 until state.size) {
      //val domainSize = jointExample.getVariableDomainSizeGivenIndex(i);//0;
      val ivariable = jointExample.getVariableGivenIndex(i);
      val indices = if (asGold) {
        if (applyPrune) { 
          ivariable.getCorrectNonPruningValueIndices();
        } else {
          ivariable.getCorrectValueIndices();
        }
      } else {
        if (applyPrune) { 
          ivariable.getAllNonPruningValueIndices();
        } else {
          ivariable.getAllValueIndices();
        }
      }
      for (vIdx <- indices) {
        //val vIdx = indices(idxIndex);
        if (vIdx != state.output(i)) {
          val newAction = new SearchAction(i, vIdx);
          //println(newAction.toString());
          actionList += newAction;
        }
      }
    }
    
    actionList;
  }
  
  
  // Do not generate actions for the variable whose musk is 0
  def actionGenerationWithMusk(jointExample: AceJointTaskExample, state: SearchState, weight: Array[Double],
                               asGold: Boolean, applyPrune: Boolean) = {
    
    val actionList = new ArrayBuffer[SearchAction]();
    
    for (i <- 0 until state.size) {

    	if (state.slotMusk(i) != 0) {

    		val ivariable = jointExample.getVariableGivenIndex(i);
    		val indices = if (asGold) {
    			if (applyPrune) { 
    				ivariable.getCorrectNonPruningValueIndices();
    			} else {
    				ivariable.getCorrectValueIndices();
    			}
    		} else {
    			if (applyPrune) { 
    				ivariable.getAllNonPruningValueIndices();
    			} else {
    				ivariable.getAllValueIndices();
    			}
    		}
    		for (vIdx <- indices) {
    			//val vIdx = indices(idxIndex);
    			if (vIdx != state.output(i)) {
    				val newAction = new SearchAction(i, vIdx);
    			  newAction.undoOldValue = state.output(i);
    				//println(newAction.toString());
    				actionList += newAction;
    			}
    		}
        
    	} else {
        //println("No action for slot " + i); 
      }
    }
    
    actionList;
  }
  
  
  def performAction(example: AceJointTaskExample, state: SearchState, action: SearchAction) {
    action.undoIndex = action.idx;
    action.undoOldValue = state.output(action.idx);
    state.output(action.idx) = action.newValIdx;
  }
  
  // return a new state after performing action
  def performActionGetNewState(example: AceJointTaskExample, state: SearchState, action: SearchAction) = {
    val newState = state.getSelfCopy();
    
    action.undoIndex = action.idx;
    action.undoOldValue = state.output(action.idx);
    newState.output(action.idx) = action.newValIdx;
    newState;
  }
  
  def cancelAction(example: AceJointTaskExample, state: SearchState, action: SearchAction) {
    state.output(action.idx) = action.undoOldValue;
  }
  
  def getGoldInitState() = {
    
  }
  
  def constructGoldMusk(ex: AceJointTaskExample, initState: SearchState, predBestOutput: Array[Int]) = {
    
    val goldInitState = initState.getSelfCopy();
    
    val goldMusk = Array.fill[Int](initState.size)(1);
    
    // correct-variable not need to change
    for (i <- 0 until goldInitState.size) {
      val vari = ex.getVariableGivenIndex(i);
      if (vari.values(predBestOutput(i)).isCorrect) { // correct value
        goldMusk(i) = 0;
        goldInitState.output(i) = predBestOutput(i);
      }
    }
    
    // no-correct-value variable no need to change
    for (i <- 0 until goldInitState.size) {
    	val vari = ex.getVariableGivenIndex(i);
    	if (vari.getCorrectNonPruningValueIndices().length == 0) { // no correct value
    		goldMusk(i) = 0;
    		goldInitState.output(i) = predBestOutput(i);
    	}
    }
    
    // set musk
    goldInitState.slotMusk = goldMusk;
    goldInitState;
  }
  
  def constructGoldMuskNoPredict(ex: AceJointTaskExample, initState: SearchState) = {
    
    val goldInitState = initState.getSelfCopy();
    
    val goldMusk = Array.fill[Int](initState.size)(1);
    
    // no-correct-value variable no need to change
    for (i <- 0 until goldInitState.size) {
    	val vari = ex.getVariableGivenIndex(i);
    	if (vari.getCorrectNonPruningValueIndices().length == 0) { // no correct value
    		goldMusk(i) = 0;
    		//goldInitState.output(i) = 0; // just set to be 0 (first index)
    	}
    }
    
    // set musk
    goldInitState.slotMusk = goldMusk;
    goldInitState;
  }
  
  
  // for synthetic experiment only !
  // Don't use this in actual testing
  def prunedActionSpaceMusk(ex: AceJointTaskExample, initState: SearchState) = {
    /*
     val copyState = initState.getSelfCopy();
    
     val musk = Array.fill[Int](initState.size)(1);
    
     val correctIdx = new ArrayBuffer[Int]();
     val otherIdx = new ArrayBuffer[Int]();
     
     // correct-variable not need to change
    for (i <- 0 until copyState.size) {
      val vari = ex.getVariableGivenIndex(i);
      if (vari.values(copyState.output(i)).isCorrect) { // correct value
        correctIdx += i;
      } else {
        otherIdx += i;
      }
    }
    
    // random pick a half correct slots
    for (idx <- correctIdx) {
      musk(idx) = 0;
    }
    
    copyState.slotMusk = musk;
    copyState;*/
    initState;
  }
  
  // run learner
  def runLearningDelayUpdate(allTrains: ArrayBuffer[AceJointTaskExample], 
                             featIndexer: Indexer[String],
                             testExs: ArrayBuffer[AceJointTaskExample],
                             beamSize: Int,
                             timeRestart: Int,
                             unaryPruner: StaticDomainPruner,
                             numIter: Int): Array[Double] = {
      
      val trainTimer = new MyTimeCounter("Training time");
      trainTimer.start();

      var weight = Array.fill[Double](featIndexer.size)(0);
      var weightSum = Array.fill[Double](featIndexer.size)(0);
      var lastWeight = Array.fill[Double](featIndexer.size)(0);

      //val  = 40;
      
      val Iteration = numIter;//10;
      val learnRate = 0.1;
      val lambda = 1e-8;

      var updateCnt = 0;
      var lastUpdtCnt = 0;

      for (iter <- 0 until Iteration) {
    	  val timeStart = Instant.now();
        lastUpdtCnt = updateCnt;
        Array.copy(weight, 0, lastWeight, 0, weight.length);

        println("Iteration " + iter);
        var exId = 0;
        for (example <- allTrains) {

          exId += 1;

          if (exId % 100 == 0) println("docCnt " + exId);
          //println("doc name " + example.docGraph.corefDoc.rawDoc.docID);
          
          val initState = SearchBasedLearner.getInitStateWithUnaryScore(example, unaryPruner, ignorePruningAtInit);//.getRandomInitState(example);
          //val initState = SearchBasedLearner.getZeroInitState(example);//.getRandomInitState(example);
          val initWithMusk = prunedActionSpaceMusk(example, initState);
          //initWithMusk.printState();
          //val predBestOutput = hillClimbing(example, initState, weight, false).output;//example.infereceIndepBest(weight); // my prediction
          //val predBestOutput = hillClimbing(example, initWithMusk, weight, false, false).output;
          val predBestOutput = beamSearch(example, initWithMusk, beamSize, weight, false, false).output;
          //val predBestOutput = searchWithRandomRestart(example, beamSize, weight, false, false, timeRestart).output;
          
          val gdinit = SearchBasedLearner.getGoldInitState(example);
          val goldInitMask = constructGoldMusk(example, gdinit, predBestOutput);
          //val goldBestOutput = hillClimbing(example, goldInit, weight, true, false).output;//example.infereceIndepGoldBest(weight);  // gold best
          val goldBestOutput = beamSearch(example, goldInitMask, beamSize, weight, true, false).output;
          
          //println("Pred = " + predBestOutput);
          //println("Gold = " + goldBestOutput);
          
          // update?
          if (!example.isCorrectOutput(predBestOutput)) {
            updateCnt += 1;
            if (updateCnt % 1000 == 0) println("Update " + updateCnt);
            
            val featGold = example.featurize(goldBestOutput);
            val featPred = example.featurize(predBestOutput);
            
            SearchBasedLearner.updateWeight(weight, 
                         featGold,
                         featPred,
                         learnRate,
                         lambda);
            SingleTaskStructTesting.sumWeight(weightSum, weight);
          }
        }

        ///////////////////////////////////////////////////
        // have a test after each iteration (for learning curve)
        val tmpAvg = new Array[Double](weightSum.size)
        Array.copy(weightSum, 0, tmpAvg, 0, weightSum.size);
        SingleTaskStructTesting.divdeNumber(tmpAvg, updateCnt.toDouble);

        //greedySearchQuickTest(allTrains, tmpAvg, unaryPruner);
        //greedySearchQuickTest(testExs, tmpAvg, unaryPruner);
        //beamSearchQuickTest(allTrains, beamSize,  tmpAvg, unaryPruner, timeRestart);
        beamSearchQuickTest(testExs, beamSize,  tmpAvg, unaryPruner, timeRestart);
        
        println("Iter Update Cnt = " + (updateCnt - lastUpdtCnt));
        checkSparsity(tmpAvg);
        
        val timeEnd = Instant.now();
        val duraTime = Duration.between(timeStart, timeEnd);
        val mints = duraTime.toMinutes()
        val secnds = duraTime.getSeconds();
        println("Iteration " + iter + " time consuming: " + mints + " minutes.");
        println("Iteration " + iter + " time consuming: " + secnds + " seconds.");
        

      }

      SingleTaskStructTesting.divdeNumber(weightSum, updateCnt.toDouble);

      // training time count~
      trainTimer.end();
      trainTimer.printSecond("Training time");
      
      println("BeamSize = " + beamSize);
      beamSearchQuickTest(allTrains, beamSize, weightSum, unaryPruner, timeRestart);
      
      weightSum;
  }

/*
  def updateWeight(currentWeight: Array[Double], 
                  featGold: HashMap[Int,Double],
                  featPred: HashMap[Int,Double],
                  eta: Double,
                  lambda: Double) {
    var gradient = Array.fill[Double](currentWeight.length)(0);//new Array[Double](currentWeight.length);
    for ((i, vgold) <- featGold) {
      gradient(i) += (vgold);
    }
    for ((j, vpred) <- featPred) {
       gradient(j) -= (vpred);
    }

    // do L2 Regularization
    //var l1norm = getL1Norm(currentWeight);
    for (i2 <- 0 until currentWeight.length) {
      //var regularizerNum: Double = Math.max(0, b);
      //var regularizerDen: Double = Math.max(0, b);
      var reg: Double = 1.0 - (eta * lambda)
          var curWeightVal = currentWeight(i2) * reg;
    currentWeight(i2) = curWeightVal + (gradient(i2) * eta);
    //currentWeight(i2) += (gradient(i2) * eta);
    }
  }
*/
  
  def greedySearchQuickTest(testExs: Seq[AceJointTaskExample], w: Array[Double], pruner: StaticDomainPruner) {
    var sumTotal : Double = 0;
    var sumErr: Double = 0;
    var sumErr1: Double = 0;
    var sumErr2: Double = 0;
    var sumErr3: Double = 0;
    
    for (ex <- testExs) {
      //val predBestOutput = ex.infereceIndepBest(w); // my prediction
      //val initState = SearchBasedLearner.getRandomInitState(ex);
      val initState = SearchBasedLearner.getInitStateWithUnaryScore(ex, pruner, ignorePruningAtInit);
      val predBestOutput = hillClimbing(ex, initState, w, false, false).output; // my search prediction
      val err = ex.getZeroOneError(predBestOutput);
      val (err1, err2, err3) = ex.getZeroOneErrorEachTask(predBestOutput, 0);
      val total = ex.totalSize;
      sumErr += err;
      sumTotal += total;
      
      sumErr1 += err1;
      sumErr2 += err2;
      sumErr3 += err3;
    }
    
    //val errRate = sumErr / sumTotal;
    val eachSum = sumTotal / 3;
    val crct = sumTotal - sumErr;
    val acc = crct / sumTotal;
    println("Error each task = [" + sumErr1 + "," + sumErr2 + "," + sumErr3 +  "] / " + eachSum );
    println("quick test: 01-Acc = " + crct + "/" + sumTotal + " = " + acc);
    //println("quick test: 01-Err0r = " + sumErr + "/" + sumTotal + " = " + errRate);
  }
  
  def beamSearchQuickTest(testExs: Seq[AceJointTaskExample], beamSize: Int, w: Array[Double], pruner: StaticDomainPruner, rndRestart: Int) {
    var sumTotal : Double = 0;
    var sumErr: Double = 0;
    var sumErr1: Double = 0;
    var sumErr2: Double = 0;
    var sumErr3: Double = 0;
    
    for (ex <- testExs) {
      //val predBestOutput = ex.infereceIndepBest(w); // my prediction
      //val initState = SearchBasedLearner.getRandomInitState(ex);
      //val initState = SearchBasedLearner.getZeroInitState(ex);
      val initState = SearchBasedLearner.getInitStateWithUnaryScore(ex, pruner, ignorePruningAtInit);
      //val predBestOutput = hillClimbing(ex, initState, w, false, false).output; // my search prediction
      val predBestOutput = beamSearch(ex, initState, beamSize, w,  false, false).output; // my search prediction
      //val predBestOutput = searchWithRandomRestart(ex, beamSize, w, false, false, rndRestart).output;
      val err = ex.getZeroOneError(predBestOutput);
      val (err1, err2, err3) = ex.getZeroOneErrorEachTask(predBestOutput, 0);
      val total = ex.totalSize;
      sumErr += err;
      sumTotal += total;
      
      sumErr1 += err1;
      sumErr2 += err2;
      sumErr3 += err3;
    }
    
    //val errRate = sumErr / sumTotal;
    val eachSum = sumTotal / 3;
    val crct = sumTotal - sumErr;
    val acc = crct / sumTotal;
    println("Error each task = [" + sumErr1 + "," + sumErr2 + "," + sumErr3 +  "] / " + eachSum );
    println("quick test: 01-Acc = " + crct + "/" + sumTotal + " = " + acc);
    //println("quick test: 01-Err0r = " + sumErr + "/" + sumTotal + " = " + errRate);
  }
  
  
  def searchWithRandomRestart(jointExample: AceJointTaskExample, 
                              initState: SearchState, 
                              bmSize: Int, // no useful here
                              weight: Array[Double], 
                              useGold: Boolean, 
                              isLossAugment: Boolean,
                              restart: Int) = {
    

    val initHashTb = new SearchHashTable(zrbkeys);
    val initList = new ArrayBuffer[SearchState]();
    
    // States generating
    breakable {
    	var icnt = 0;
    	for (i2 <- 0 to 1000) { // max 1000 initial outputs
    		val initState = SearchBasedLearner.getRandomInitState(jointExample);
    		//val hskey = initHashTb.computeIndex(initState);
    		if (!initHashTb.probeExistence(initState)) {
    			initHashTb.insertNewInstance(initState);
    			initList += initState;
    			icnt += 1;
    			if (icnt >= restart) {
    				break;
    			}
    		}
    	}
    }
    
    var bestCost: Double = Double.NegativeInfinity;
    var bestOutState: SearchState = null;
    var bestRank: Int = -1;
    // inference
    for (i <- 0 until initList.size) {
      val predState = beamSearch(jointExample, initState, bmSize, weight, false, false);
      val predSc = predState.cachedPredScore;
      val trueLoss = predState.cachedTrueLoss;
      if (predSc > bestCost) {
        bestCost = predSc;
        bestOutState = predState;
        bestRank = i;
      }
      
      //println(i + ":" + );
    }
    
    (bestOutState, bestRank);
  }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

object SearchBasedLearner {
  
  // get uniform random initial state
  def getRandomInitState(example: AceJointTaskExample) = {
    val r = scala.util.Random;
    
    val zeroOutput = new Array[Int](example.totalSize);
    for (i <- 0 until example.totalSize) {
      val domainValueIndices = example.getVariableDomainSizeGivenIndex(i);
      if (domainValueIndices > 0) {
        zeroOutput(i) = r.nextInt(domainValueIndices);
      } else {
        zeroOutput(i) = -1;
      }
    }
    val state = new SearchState(zeroOutput);
    state;
  }
  
  def getZeroInitState(example: AceJointTaskExample) = {
    val zeroOutput = new Array[Int](example.totalSize);
    for (i <- 0 until example.totalSize) {
      val domainValueIndices = example.getVariableDomainSizeGivenIndex(i);
      zeroOutput(i) = 0;
      /* if (domainValueIndices > 0) {
        zeroOutput(i) = 0;
      } else {
        zeroOutput(i) = -1;
      }*/
    }
    val state = new SearchState(zeroOutput);
    state;
  }
/* 
  def getInitStateWithUnaryScore(ex: AceJointTaskExample, prunerScorer: SearchDomainPruner) = {
    getZeroInitState(ex);
  }
 
*/

/*
  def getInitStateWithUnaryScore(ex: AceJointTaskExample, prunerScorer: StaticDomainPruner) = {//, overRidePruning: Boolean) = {
    val initOutput = new ArrayBuffer[Int]();
    for (i <- 0 until ex.corefVars.size) {
      initOutput += prunerScorer.getCorefVarBestValue(ex.corefVars(i));//(ex.corefVars(i).getBestValue(unaryModelWeight));
    }
    for (i <- 0 until ex.nerVars.size) {
      initOutput += prunerScorer.getNerVarBestValue(ex.nerVars(i));//(ex.nerVars(i).getBestValue(unaryModelWeight));
    }
    for (i <- 0 until ex.wikiVars.size) {
      initOutput += prunerScorer.getWikiVarBestValue(ex.wikiVars(i));//(ex.wikiVars(i).getBestValue(unaryModelWeight));
    }
    val state = new SearchState(initOutput.toArray);
    state;
  }
*/
  def getInitStateWithUnaryScore(ex: AceJointTaskExample, prunerScorer: StaticDomainPruner, overRidePruning: Boolean) = {
    val initOutput = new ArrayBuffer[Int]();
    for (i <- 0 until ex.corefVars.size) {
      //initOutput += prunerScorer.getCorefVarBestValue(ex.corefVars(i));//(ex.corefVars(i).getBestValue(unaryModelWeight));
      val vidx1 = prunerScorer.getCorefVarBestValue(ex.corefVars(i), overRidePruning);//(ex.corefVars(i).getBestValue(unaryModelWeight));
      if (overRidePruning) ex.corefVars(i).values(vidx1).isPruned = false; // override
      initOutput += vidx1;
    }
    for (i <- 0 until ex.nerVars.size) {
      val vidx2 = prunerScorer.getNerVarBestValue(ex.nerVars(i), overRidePruning);//(ex.nerVars(i).getBestValue(unaryModelWeight));
      if (overRidePruning) ex.nerVars(i).values(vidx2).isPruned = false; // override
      initOutput += vidx2;
    }
    for (i <- 0 until ex.wikiVars.size) {
      val vidx3 = prunerScorer.getWikiVarBestValue(ex.wikiVars(i), overRidePruning);//(ex.wikiVars(i).getBestValue(unaryModelWeight));
      if (overRidePruning) ex.wikiVars(i).values(vidx3).isPruned = false; // override
      initOutput += vidx3;
    }
    val state = new SearchState(initOutput.toArray);
    state;
  }
  ///////////////////////////////////
  // construct gold init state
  ///////////////////////////////////
  
  def getGoldInitState(ex: AceJointTaskExample) = {
    val initOutput = new Array[Int](ex.totalSize);
    for (i <- 0 until ex.totalSize) {
      val vrbl = ex.getVariableGivenIndex(i);
      val crrIdxs = vrbl.getCorrectValueIndices();
      initOutput(i) = 0;
      if (crrIdxs.length > 0) { // if there is a correct value
        initOutput(i) = crrIdxs(0);
      }
    }
    val state = new SearchState(initOutput);
    state;
  }
  
  def getGoldInitStateWithUnaryScore(ex: AceJointTaskExample, prunerScorer: StaticDomainPruner) = {
    val initOutput = new ArrayBuffer[Int]();
    for (i <- 0 until ex.corefVars.size) {
      initOutput += prunerScorer.getCorefVarCorrectBestValue(ex.corefVars(i));//(ex.corefVars(i).getBestValue(unaryModelWeight));
    }
    for (i <- 0 until ex.nerVars.size) {
      initOutput += prunerScorer.getNerVarCorrectBestValue(ex.nerVars(i));//(ex.nerVars(i).getBestValue(unaryModelWeight));
    }
    for (i <- 0 until ex.wikiVars.size) {
      initOutput += prunerScorer.getWikiVarCorrectBestValue(ex.wikiVars(i));//(ex.wikiVars(i).getBestValue(unaryModelWeight));
    }
    val state = new SearchState(initOutput.toArray);
    state;
  }

  def outputFromSearchState(state: SearchState) = {
    state.output;
  }
  
  // for speed test only
  def getRandomFeatureVec() = {
    val feat = new HashMap[Int, Double]();
    for (i <- 0 until 100) {
      feat += (i -> 1.0);
    }
    feat;
  }
  
  def isSameFeature(feat1: HashMap[Int, Double], feat2: HashMap[Int, Double]): Boolean = {
    for ((k1, v1) <- feat1) {
      val v2 = feat2(k1);
      if (v2 != v1) {
        throw new RuntimeException("Feature2 value at index " + k1 + " diffs: feat1 = " + v1 + ", feat2 = " + v2);
      }
    }
    for ((k2, v2) <- feat2) {
      val v1 = feat1(k2);
      if (v2 != v1) {
        throw new RuntimeException("Feature1 value at index " + k2 + " diffs: feat1 = " + v1 + ", feat2 = " + v2);
      }
    }
    true;
  }
  

  def updateWeight(currentWeight: Array[Double], 
                  featGold: HashMap[Int,Double],
                  featPred: HashMap[Int,Double],
                  eta: Double,
                  lambda: Double) {
    var gradient = Array.fill[Double](currentWeight.length)(0);//new Array[Double](currentWeight.length);
    for ((i, vgold) <- featGold) {
      gradient(i) += (vgold);
    }
    for ((j, vpred) <- featPred) {
       gradient(j) -= (vpred);
    }

    // do L2 Regularization
    //var l1norm = getL1Norm(currentWeight);
    for (i2 <- 0 until currentWeight.length) {
      //var regularizerNum: Double = Math.max(0, b);
      //var regularizerDen: Double = Math.max(0, b);
      var reg: Double = 1.0 - (eta * lambda)
          var curWeightVal = currentWeight(i2) * reg;
    currentWeight(i2) = curWeightVal + (gradient(i2) * eta);
    //currentWeight(i2) += (gradient(i2) * eta);
    }
  }
  
}

