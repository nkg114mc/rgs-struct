package berkeleyentity.oregonstate

import java.io.PrintWriter
import berkeleyentity.oregonstate.pruner.StaticDomainPruner
import scala.util.Random
import scala.collection.mutable.ArrayBuffer
import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.futile.util.Counter
import scala.util.control.Breaks.break
import scala.util.control.Breaks.breakable
import scala.collection.mutable.HashMap

object CSearchLearner {
    
  //////////////////////////////////////////////
  //////////////////////////////////////////////
  //////////////////////////////////////////////
  
  def CSearchLearning(allTrains: ArrayBuffer[AceJointTaskExample], 
                      featIndexer: Indexer[String],
                      testExs: ArrayBuffer[AceJointTaskExample],
                      unaryPruner: StaticDomainPruner,
                      numIter: Int,
                      zrbkeys: ZobristKeys): Array[Double] = {
      
      var weight = Array.fill[Double](featIndexer.size)(0);
      var weightSum = Array.fill[Double](featIndexer.size)(0);
      var lastWeight = Array.fill[Double](featIndexer.size)(0);

      val logger = new PrintWriter("C-Search-Heuristic-Err.log");

      val Iteration = numIter;//10;
      var updateCnt = 0;
      var lastUpdtCnt = 0;
      val runOnTrajTraining = true;
      val trainErrCounter = new Counter[String]();

      for (iter <- 0 until Iteration) {
        lastUpdtCnt = updateCnt;
        Array.copy(weight, 0, lastWeight, 0, weight.length);

        println("Iteration " + iter);
        for (example <- allTrains) {
          val initState = SearchBasedLearner.getInitStateWithUnaryScore(example, unaryPruner,true);//.getRandomInitState(example);
          val updateCntInc = CSearchGreedy(example, initState, weight, weightSum, runOnTrajTraining, true, trainErrCounter, zrbkeys);
          updateCnt += updateCntInc; 
        }

        ///////////////////////////////////////////////////
        // have a test after each iteration (for learning curve)
        val tmpAvg = new Array[Double](weightSum.size)
        Array.copy(weightSum, 0, tmpAvg, 0, weightSum.size);
        SingleTaskStructTesting.divdeNumber(tmpAvg, updateCnt.toDouble);
        println("Iter Update Cnt = " + (updateCnt - lastUpdtCnt));
        
        ////
        logger.print("Train: ");
        CSearchHeuristicQuickTest(allTrains, tmpAvg, unaryPruner, logger, zrbkeys);
        logger.print("Test: ");
        CSearchHeuristicQuickTest(testExs, tmpAvg, unaryPruner, logger, zrbkeys);
      }

      SingleTaskStructTesting.divdeNumber(weightSum, updateCnt.toDouble);
      logger.close();
      weightSum;
  }
  
  def CSearchHeuristicQuickTest(testExs: ArrayBuffer[AceJointTaskExample], w: Array[Double], pruner: StaticDomainPruner, logger: PrintWriter, zrbkeys: ZobristKeys) {
    var sumTotal : Double = 0;
    val errCounter = new Counter[String]();
    val wsum = new Array[Double](w.length);
    for (ex <- testExs) {
      val initState = SearchBasedLearner.getInitStateWithUnaryScore(ex, pruner,true);
      CSearchGreedy(ex, initState, w, wsum, true, false, errCounter, zrbkeys); // my search prediction
    }
    val sumErr1 = errCounter.getCount("good").toInt;
    val sumErr2 = errCounter.getCount("netral").toInt;
    val sumErr3 = errCounter.getCount("bad").toInt;
    val eachSum = sumErr1 + sumErr2 + sumErr3;
    println("Trajectory action types[good,netral,bad] = [" + sumErr1 + "," + sumErr2 + "," + sumErr3 +  "] / " + eachSum );
    logger.println("Trajectory action types[good,netral,bad] = [" + sumErr1 + "," + sumErr2 + "," + sumErr3 +  "] / " + eachSum );
    logger.flush();
  }
  
  def getBestChildState(actions: ArrayBuffer[SearchState]) = {
    var bestAct: SearchState = null; 
    var bestScore: Double = -Double.MaxValue;
    for (act <- actions) {
      if (act.cachedPredScore > bestScore) {
        bestScore = act.cachedPredScore;
        bestAct = act;
      }
    }
    (bestAct, bestScore);
  }
  
  def getBestCorrectChildState(actions: ArrayBuffer[SearchState]) = {
	  var bestGoodAct: SearchState = null; 
  var bestScore: Double = -Double.MaxValue;
  for (act <- actions) {
	  if (act.actClass.equals("good")) {
		  if (act.cachedPredScore > bestScore) {
			  bestScore = act.cachedPredScore;
			  bestGoodAct = act;
		  }
	  }
  }

  // no good, try netral!
  if (bestGoodAct == null) {
	  for (act <- actions) {
		  if (act.actClass.equals("good") || act.actClass.equals("netral")) {
			  if (act.cachedPredScore > bestScore) {
				  bestScore = act.cachedPredScore;
				  bestGoodAct = act;
			  }
		  }
	  }
  }
  
  (bestGoodAct, bestScore);
  }
  
  def getBestAction(actions: ArrayBuffer[SearchAction]) = {
    var bestAct: SearchAction = null; 
    var bestScore: Double = -Double.MaxValue;
    for (act <- actions) {
      if (act.score > bestScore) {
        bestScore = act.score;
        bestAct = act;
      }
    }
    (bestAct, bestScore);
  }

  def getBestCorrectAction(actions: ArrayBuffer[SearchAction]) = {
	  var bestGoodAct: SearchAction = null; 
    var bestScore: Double = -Double.MaxValue;
    for (act <- actions) {
	    if (act.actClass.equals("good")) {
		    if (act.score > bestScore) {
			    bestScore = act.score;
			    bestGoodAct = act;
		    }
	    }
    }
    (bestGoodAct, bestScore);
  }
  
  def isGoodAction(act: SearchAction) = {
	  if (act.actClass != null) {
		  if (act.actClass.equals("good")) {
			  true;
		  } else {
			  false;
		  }
	  } else {
		  false;
	  }
  }
  
  def isBadAction(act: SearchAction) = {
    if (act.actClass != null) {
      if (act.actClass.equals("bad")) {
        true;
      } else {
        false;
      }
    } else {
      false;
    }
  }
  
  def isGoodState(act: SearchState) = {
    if (act.actClass != null) {
      if (act.actClass.equals("good")) {
        true;
      } else {
        false;
      }
    } else {
      false;
    }
  }
  
  
  // mainly compute features incrementally
  def stateFeatureIncreamentCompute(oldFeat: HashMap[Int, Double], jointExample: AceJointTaskExample, oldState: SearchState, action: SearchAction) = {//, newState: SearchState) {
    
    // about features
    //val newFeatMap = oldState.cachedFeatureVector.clone();
    val newFeatMap = oldFeat.clone();
    
    val involvedTask = jointExample.getVariableTaskFromIndex(action.idx);
    val globalIndex = action.idx;
    val involvedIndex = jointExample.getSingleTaskIndex(action.idx);
    val oldvIdx = action.undoOldValue;
    val newvIdx = action.newValIdx;

    /*
    println("nMents = " + jointExample.numMentions);
    println(action.idx +", "+involvedIndex + ", " + involvedTask);
    
    println(jointExample.corefOutputArrStart +" - "+ jointExample.corefOutputArrEnd);
    println(jointExample.nerOutputArrStart +" - "+ jointExample.nerOutputArrEnd);
    println(jointExample.wikiOutputArrStart+" - "+ jointExample.wikiOutputArrEnd);
    */
    
    if (involvedTask == 1) {
      
      // coref
      val cvari = jointExample.getVariableGivenIndex(globalIndex);
      val nivIdx = oldState.output(jointExample.getNerGlobalIndex(involvedIndex));
      val wivIdx = oldState.output(jointExample.getWikiGlobalIndex(involvedIndex));
      
      // minus old
      
      // affected variables
      val factorNerOldj = jointExample.corefNerFactors(involvedIndex)(oldvIdx);
      val factorWikiOldj = jointExample.corefWikiFactors(involvedIndex)(oldvIdx);
      val njvIdxOld = oldState.output(jointExample.getNerGlobalIndex(oldvIdx));
      val wjvIdxOld = oldState.output(jointExample.getWikiGlobalIndex(oldvIdx));

      val feat1 = cvari.values(oldvIdx).feature;
      val feat2 = factorNerOldj.feature(nivIdx)(njvIdxOld);
      val feat3 = factorWikiOldj.feature(wivIdx)(wjvIdxOld);
      
      jointExample.subtractIndexedVector(newFeatMap, feat1);
      jointExample.subtractIndexedVector(newFeatMap, feat2);
      jointExample.subtractIndexedVector(newFeatMap, feat3);
      
      // plus new!
      val factorNerNewj = jointExample.corefNerFactors(involvedIndex)(newvIdx);
      val factorWikiNewj = jointExample.corefWikiFactors(involvedIndex)(newvIdx);
      val njvIdxNew = oldState.output(jointExample.getNerGlobalIndex(newvIdx));
      val wjvIdxNew = oldState.output(jointExample.getWikiGlobalIndex(newvIdx));

      val feat4 = cvari.values(newvIdx).feature;
      val feat5 = factorNerNewj.feature(nivIdx)(njvIdxNew);
      val feat6 = factorWikiNewj.feature(wivIdx)(wjvIdxNew);
      
      jointExample.addIndexedVector(newFeatMap, feat4);
      jointExample.addIndexedVector(newFeatMap, feat5);
      jointExample.addIndexedVector(newFeatMap, feat6);
      
    } else if (involvedTask == 2) {
      
      // ner
      val nvari = jointExample.getVariableGivenIndex(globalIndex);
      val wivIdx = oldState.output(jointExample.getWikiGlobalIndex(involvedIndex));

      val factorNerWiki = jointExample.nerWikiFactors(involvedIndex);
      
      val feat1 = nvari.values(oldvIdx).feature;
      val feat2 = factorNerWiki.feature(oldvIdx)(wivIdx);
      
      val corefOut = jointExample.getCorefSubStruct(oldState.output);
      val nerOut = jointExample.getNerSubStruct(oldState.output);
      val j = corefOut(involvedIndex);
      val njvIdx = nerOut(j);
      
      val changedCorefNerFactorsFromi = jointExample.corefNerFactors(involvedIndex)(j);
      val feat3 = changedCorefNerFactorsFromi.feature(oldvIdx)(njvIdx);
      
      
      jointExample.subtractIndexedVector(newFeatMap, feat1);
      jointExample.subtractIndexedVector(newFeatMap, feat2);
      jointExample.subtractIndexedVector(newFeatMap, feat3);
      
      val feat4 = nvari.values(newvIdx).feature;
      val feat5 = factorNerWiki.feature(newvIdx)(wivIdx);
      val feat6 = changedCorefNerFactorsFromi.feature(newvIdx)(njvIdx);
      
      jointExample.addIndexedVector(newFeatMap, feat4);
      jointExample.addIndexedVector(newFeatMap, feat5);
      jointExample.addIndexedVector(newFeatMap, feat6);
      
      //val changedCorefNerFactorsToi = new ArrayBuffer[]();
      for (i <- (involvedIndex + 1) until corefOut.length) {
    	  if (corefOut(i) == involvedIndex) {
    		  // i --> involvedIndex
    		  val nivIdx = nerOut(i);
    		  val changedCorefNerFactorsToi = jointExample.corefNerFactors(i)(involvedIndex);
    		  val ftOld = changedCorefNerFactorsToi.feature(nivIdx)(oldvIdx);
    		  val ftNew = changedCorefNerFactorsToi.feature(nivIdx)(newvIdx);
    		  jointExample.subtractIndexedVector(newFeatMap, ftOld);
    		  jointExample.addIndexedVector(newFeatMap, ftNew);
    	  }
      }
      
    } else if (involvedTask == 3) {
      
      // wiki
      val wvari = jointExample.getVariableGivenIndex(globalIndex);
      val nivIdx = oldState.output(jointExample.getNerGlobalIndex(involvedIndex));

      val factorNerWiki = jointExample.nerWikiFactors(involvedIndex);
      
      val feat1 = wvari.values(oldvIdx).feature;
      val feat2 = factorNerWiki.feature(nivIdx)(oldvIdx);
      
      val corefOut = jointExample.getCorefSubStruct(oldState.output);
      val wikiOut = jointExample.getWikiSubStruct(oldState.output);
      val j = corefOut(involvedIndex);
      val wjvIdx = wikiOut(j);
      
      val changedCorefWikiFactorsFromi = jointExample.corefWikiFactors(involvedIndex)(j);
      val feat3 = changedCorefWikiFactorsFromi.feature(oldvIdx)(wjvIdx);
      
      
      jointExample.subtractIndexedVector(newFeatMap, feat1);
      jointExample.subtractIndexedVector(newFeatMap, feat2);
      jointExample.subtractIndexedVector(newFeatMap, feat3);
      
      val feat4 = wvari.values(newvIdx).feature;
      val feat5 = factorNerWiki.feature(nivIdx)(newvIdx);
      val feat6 = changedCorefWikiFactorsFromi.feature(newvIdx)(wjvIdx);
      
      jointExample.addIndexedVector(newFeatMap, feat4);
      jointExample.addIndexedVector(newFeatMap, feat5);
      jointExample.addIndexedVector(newFeatMap, feat6);
      
      //val changedCorefNerFactorsToi = new ArrayBuffer[]();
      for (i <- (involvedIndex + 1) until corefOut.length) {
        if (corefOut(i) == involvedIndex) {
          // i --> involvedIndex
          val wivIdx = wikiOut(i);
          val changedCorefWikiFactorsToi = jointExample.corefWikiFactors(i)(involvedIndex);
          val ftOld = changedCorefWikiFactorsToi.feature(wivIdx)(oldvIdx);
          val ftNew = changedCorefWikiFactorsToi.feature(wivIdx)(newvIdx);
          jointExample.subtractIndexedVector(newFeatMap, ftOld);
          jointExample.addIndexedVector(newFeatMap, ftNew);
        }
      }
      
    } else {
      throw new RuntimeException("Unknown task number: " + involvedTask);
    }

    newFeatMap;
  }
  

  
  //// Conventional C-Search implementation
  def CSearchGreedy(jointExample: AceJointTaskExample,
                    initState: SearchState, 
                    weight: Array[Double], 
                    weightSum: Array[Double],
                    useOnTraj: Boolean, 
                    doUpdate: Boolean, 
                    errCounter: Counter[String],
                    zrbkeys: ZobristKeys) = {
    

    val learnRate = 0.1;
    val lambda = 1e-8;
    var updateCnt: Int = 0;

    val searcher = new SearchBasedLearner(zrbkeys);
    val hashTb = new SearchHashTable(zrbkeys);
    val trajectory = new Array[SearchState](2000);

    val maxDepth = 10000;
    var depth = 0;
    var flatStep = 0;
    
    // initialization!
    var currentState = initState.getSelfCopy();
    currentState.cachedPredScore = SingleTaskStructTesting.computeScoreSparse(weight, jointExample.featurize(currentState.output));
    currentState.cachedTrueLoss = searcher.getTrueAcc(jointExample, currentState);
    currentState.cachedFeatureVector = jointExample.featurize(currentState.output);
    trajectory(0) = currentState;
    hashTb.insertNewInstance(currentState);
    
    
    // start greedy search
    breakable {
      for (step <- 1 until maxDepth) {
        
        println("Greedy search depth: " + step);

        var siblingsCnt: Int = 0;
        var actionCnter = new Counter[String]();

        val actions = searcher.actionGenerationNormal(jointExample, currentState, weight, false, true);
        val nonRepeatActions = new ArrayBuffer[SearchAction]();
        val siblingStates = new ArrayBuffer[SearchState]();
        for (act <- actions) {
          // do this action
          val newChildState = searcher.performActionGetNewState(jointExample, currentState, act);

          val stateIsOk = (!hashTb.probeExistence(newChildState));
          if (stateIsOk) {
            siblingsCnt += 1;
            
            //println(printActionVerbose(jointExample, act));
            val incrementalFeature = stateFeatureIncreamentCompute(currentState.cachedFeatureVector, jointExample, currentState, act);
            //val stateFeature = jointExample.featurize(newChildState.output);
            //SearchBasedLearner.isSameFeature(stateFeature, incrementalFeature); // check!
            val stateFeature = incrementalFeature;
            
            //val stateFeature = SearchBasedLearner.getRandomFeatureVec();
            //println(stateFeature.size);
            val predScore = SingleTaskStructTesting.computeScoreSparse(weight, stateFeature);
            var trueAcc = searcher.getTrueAcc(jointExample, newChildState);
            var actClass = if (trueAcc > trajectory(step - 1).cachedTrueLoss) {
              "good"
            } else if (trueAcc == trajectory(step - 1).cachedTrueLoss) {
              "netral"
            } else {
              "bad";
            }
            
            //act.sparseFeature = stateFeature;
            act.actClass = actClass;
            act.score = predScore;
            act.trueAcc = trueAcc;
            actionCnter.incrementCount(actClass, 1.0);
            nonRepeatActions += act;
            
            // about the state
            newChildState.cachedFeatureVector = stateFeature;
            newChildState.cachedPredScore = predScore;
            newChildState.cachedTrueLoss = trueAcc;
            newChildState.actClass = actClass;
            
            siblingStates += newChildState;
          }

          // undo action
          //cancelAction(jointExample, currentState, act);
        }
        
        val goodCnt = actionCnter.getCount("good").toInt;
        val netralCnt = actionCnter.getCount("netral").toInt;
        val badCnt = actionCnter.getCount("bad").toInt;
        println("branch factor = " + siblingsCnt + ", output size = " + jointExample.totalSize);
        println("(good:" + goodCnt + ", netral:" + netralCnt + ", bad:" + badCnt + ")");
        println("HashTb size = " + hashTb.totalSize);
        

        val (bestAct, bestActScore) = getBestAction(nonRepeatActions);
        val (bestGoodAct, bestGoodActScore) = getBestCorrectAction(nonRepeatActions);
        
        val (bestState, bestScore) = getBestChildState(siblingStates);
        val (bestGoodState, bestGoodScore) = getBestCorrectChildState(siblingStates);
        

        // shall we continue?
        if (useOnTraj) {
        	if (goodCnt == 0) {
        		break; // no good action anymore!
        	}
        } else {
        	if (bestScore < trajectory(step - 1).cachedPredScore) {
        		break; // reach the hill peak!
        	} else if (bestScore == trajectory(step - 1).cachedPredScore){
        		// reach a flat range, if this continues for 5 more step, break (local optimal)
        		flatStep += 1;
        		if (flatStep >= 5) {
        			break;
        		}
        	}
        }
        
        //println("bestAction = " + printActionVerbose(jointExample, bestAct) + ", action_typ: " + bestAct.actClass);
        //println("bestGoodAction = " + printActionVerbose(jointExample, bestGoodAct)+ ", action_typ: " + bestGoodAct.actClass);
        
        //println("bestAction = " + printActionVerbose(jointExample, bestAct) + ", action_typ: " + bestAct.actClass);
        //println("bestGoodAction = " + printActionVerbose(jointExample, bestGoodAct)+ ", action_typ: " + bestGoodAct.actClass);
        errCounter.incrementCount(bestAct.actClass, 1.0);
        
        
        // Update ////////////////////////////////////////////
        if (doUpdate) {
          // check Error and update
          if (needUpdate(siblingStates, bestState)) {
            //val updtCnt = fullRankingUpdate(siblingStates, weight, weightSum, learnRate, lambda);
            //println("Step update " + updtCnt);
            //updateCnt += updtCnt;
          }
        }
        // End-of-update /////////////////////////////////////
        
        currentState = if (useOnTraj) { // on trajectory
          bestGoodState;
        } else { // off trajectory
          bestState;
        }
        trajectory(step) = currentState;
        hashTb.insertNewInstance(currentState); // remember the extended state
      }

    } // breakable

    updateCnt;
  }
  
  def needUpdate(childrenStates: ArrayBuffer[SearchState], bestState: SearchState) = { // ?
    (!isGoodState(bestState));
  }

/*
  // return the number of update!
  def fullRankingUpdate(childrenStates: ArrayBuffer[SearchState], weight: Array[Double], weightSum: Array[Double], learnRate: Double, lambda: Double) : Int = {
    var updateCnt = 0;

    // sort by the pred score
    val sortedChildStates = (childrenStates.toSeq.sortWith(_.cachedPredScore > _.cachedPredScore)).toArray;
    
    // rank learning!
    for (fst <- 0 until (sortedChildStates.size - 1)) {
    	if (sortedChildStates(fst).actClass.equals("good")) {
    		for (snd <- 0 until fst) {
    			if (!(sortedChildStates(snd).actClass.equals("good"))) { // update!
    				updateCnt += 1;
    				val featGold = sortedChildStates(fst).cachedFeatureVector; // good
    				val featPred = sortedChildStates(snd).cachedFeatureVector; // non=good
    				updateWeight(weight, 
    						featGold,
    						featPred,
    						learnRate,
    						lambda);
    				SingleTaskStructTesting.sumWeight(weightSum, weight);
    			}
    		}

    	}
    }
    
    updateCnt;
  }
*/
  
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  
  def randomSampleGeneratingLearning(allTrains: ArrayBuffer[AceJointTaskExample], 
                                     featIndexer: Indexer[String],
                                     testExs: ArrayBuffer[AceJointTaskExample],
                                     unaryPruner: StaticDomainPruner): Array[Double] = {
      
      var weight = Array.fill[Double](featIndexer.size)(0);
      var weightSum = Array.fill[Double](featIndexer.size)(0);
      var lastWeight = Array.fill[Double](featIndexer.size)(0);

      //val logger = new PrintWriter("Sampling-Err.log");

      val Iteration = 10;
      var updateCnt = 0;
      var lastUpdtCnt = 0;
      val runOnTrajTraining = true;
      val trainErrCounter = new Counter[String]();

      for (iter <- 0 until Iteration) {
        lastUpdtCnt = updateCnt;
        Array.copy(weight, 0, lastWeight, 0, weight.length);

        println("Iteration " + iter);
        var docCnt = 0;
        for (example <- allTrains) {
          docCnt += 1;
          println("Training doc " + docCnt);
          val initState = SearchBasedLearner.getInitStateWithUnaryScore(example, unaryPruner,true);
          //val updateCntInc = 
          randomSampling(example, initState, weight, weightSum, runOnTrajTraining, true, trainErrCounter, 1.0);
          //updateCnt += updateCntInc; 
        }

        ///////////////////////////////////////////////////
        // have a test after each iteration (for learning curve)
        val tmpAvg = new Array[Double](weightSum.size)
        Array.copy(weightSum, 0, tmpAvg, 0, weightSum.size);
        SingleTaskStructTesting.divdeNumber(tmpAvg, updateCnt.toDouble);
        println("Iter Update Cnt = " + (updateCnt - lastUpdtCnt));
        
        ////
        //logger.print("Train: ");
        //CSearchHeuristicQuickTest(allTrains, tmpAvg, unaryPruner, logger);
        //logger.print("Test: ");
        //CSearchHeuristicQuickTest(testExs, tmpAvg, unaryPruner, logger);
      }

      SingleTaskStructTesting.divdeNumber(weightSum, updateCnt.toDouble);
      //logger.close();
      weightSum;
  }
  
  def randomSampling(ex: AceJointTaskExample, initState: SearchState, weight: Array[Double], weightSum: Array[Double],
                     useOnTraj: Boolean, doUpdate: Boolean, errCounter: Counter[String], epsilon: Double) = {
    
    val asGold = false;
    val applyPrune = true;
    
    

      // get value indices
      for (i <- 0 until ex.totalSize) {
        val ivariable = ex.getVariableGivenIndex(i);
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

        var scaledSum: Double = 0;
        val scaledScores = new ArrayBuffer[Double]();

        for (vIdx <- indices) {
          val actualScore = ivariable.values(vIdx).unaryScore;
          //val scaledWeight = exp(actualScore / epsilon);

          //scaledSum += scaledWeight;

        }
      }

  }

  
  def getRandomInitState(ex: AceJointTaskExample): SearchState = {
    val random = new Random();
    val output = new Array[Int](ex.totalSize);
    // get value indices
    for (i <- 0 until ex.totalSize) {
    	val ivariable = ex.getVariableGivenIndex(i);
    	val varDomainSize = ivariable.values.size;
    	val idx = (varDomainSize.toDouble * random.nextDouble()).toInt % varDomainSize;
    	output(i) = idx;
    }
    val initState = new SearchState(output);
    initState;
  }
  
/*  
  def searchWithRandomRestart(jointExample: AceJointTaskExample, beamSize: Int, weight: Array[Double], useGold: Boolean, isLossAugment: Boolean, timeRestart: Int) = {

    if (timeRestart < 1) {
      throw new RuntimeException("Wrong restart times = " + timeRestart);
    }
    
    if (useGold) {
      throw new RuntimeException("Should not run RandomeRestart for gold!");
    }
    
    val allBest = new ArrayBuffer[SearchState]();
    for (i <- 0 until timeRestart) {
    	val rndInit = getRandomInitState(jointExample);
    	val thisOutput = beamSearch(jointExample, rndInit, beamSize, weight, useGold, isLossAugment);
    	allBest += thisOutput;
    	println("Restart(" + i + "): " + thisOutput.cachedPredScore + ", " + thisOutput.cachedTrueLoss);
    }

    var bestPred: Double = -Double.MaxValue;
    var bestState: SearchState = null;
    for (j <- 0 until allBest.size) {
      if (allBest(j).cachedPredScore > bestPred) {
        bestPred = allBest(j).cachedPredScore;
        bestState = allBest(j)
      }
    }
    println("Best: " + bestState.cachedPredScore + ", " + bestState.cachedTrueLoss);
    
    bestState;
	}
*/
  
}