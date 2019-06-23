package berkeleyentity.oregonstate.pruner

import java.io.File
import java.io.PrintWriter

import scala.util.Random
import scala.collection.mutable.ArrayBuffer
import scala.util.control.Breaks.break
import scala.util.control.Breaks.breakable
import scala.collection.mutable.HashMap

import berkeleyentity.oregonstate.SearchAction
import berkeleyentity.oregonstate.SearchState
import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.futile.util.Counter
import berkeleyentity.oregonstate.SearchHashTable
import berkeleyentity.oregonstate.SingleTaskStructTesting
import berkeleyentity.oregonstate.ZobristKeys
import berkeleyentity.oregonstate.AceJointTaskExample
import berkeleyentity.oregonstate.SearchBasedLearner
import berkeleyentity.oregonstate.CSearchLearner
import berkeleyentity.oregonstate.StatePredComparator
import berkeleyentity.oregonstate.SearchBeam
import berkeleyentity.coref.DocumentGraph
import berkeleyentity.oregonstate.VarValue
import berkeleyentity.oregonstate.AceMultiTaskExample
import berkeleyentity.oregonstate.CorefTesting
import berkeleyentity.oregonstate.CorefDecisionExample
import berkeleyentity.ilp.HistgramRecord
import berkeleyentity.xgb.XgbMatrixBuilder
import berkeleyentity.oregonstate.QueryWikiValue
import berkeleyentity.coref.PairwiseIndexingFeaturizer


import ml.dmlc.xgboost4j.scala.Booster
import ml.dmlc.xgboost4j.java.{DMatrix => JDMatrix}
import ml.dmlc.xgboost4j.java.example.util.DataLoader
import ml.dmlc.xgboost4j.scala.{XGBoost, DMatrix}
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.InputStream
import java.nio.charset.Charset
import java.io.FileInputStream

/*
trait DynamicPruner {
  
  // input is the children states, and beam states
  // children states is the new states expanded from the last best state
  // Note: in greedy search, beam states are just states in current beam except the 
  //       expanded best state 
  
  // return a subset of states that pruner believe to be good       
  def pruneChildStates(example: AceJointTaskExample, childStates: Seq[SearchState]): Seq[SearchState];
  
  // return a subset of actions that 
  def pruneStateAndChileActions(example: AceJointTaskExample, expandState: SearchState, childActions: Seq[SearchAction]): Seq[SearchAction];
   
}
*/
/*
// the pruner 
class TopAlphaDynamicPruner(val topAlpha: Double) extends DynamicPruner {
  
  // TODO: Not implemented yet...
  override def pruneChildStates(example: AceJointTaskExample, childStates: Seq[SearchState]): Seq[SearchState] = {
    childStates;
  }
  
  override def pruneStateAndChileActions(example: AceJointTaskExample, expandState: SearchState, childActions: Seq[SearchAction]): Seq[SearchAction] = {
    if (childActions.size < 10) { // no need tp prune at all...
      childActions;
    } else {
      doPruning(example, expandState, childActions);
    }
  }
  

  def doPruning(example: AceJointTaskExample, expandState: SearchState, childActions: Seq[SearchAction]): Seq[SearchAction] = {
    
    val remainActs = new ArrayBuffer[SearchAction]();
    for (act <- childActions) {
        
       // val incrementalScore = jointExample.getScoreFastIncrementalByAction(currentState.output, act, corefRightLink);
    	  //val newLoss = getTrueLossIncreamentalByAction(jointExample, oldTrueLoss, act);
    	  
    		// do this action
    		DynamicPruner.doAction(example, expandState, act);
    		
    		//val incrementalFeature = CSearchLearner.stateFeatureIncreamentCompute(example, expandState, act);
    		//val stateFeature = example.featurize(expandState.output);
  

    		// undo action
    		DynamicPruner.undoAction(example, expandState, act);
    }
    remainActs;
  }

}
*/

class ActionRankElement(val action: SearchAction, val label: Int, val feature: HashMap[Int,Double]) {
  
  var shouldBePruned: Boolean = false; // true = pruned, false = kept
  var rankScore: Double = 0; // ranking score
  
  def clearFeature() {
    feature.clear();
  }
  
}

class InitPruner(val booster: Booster,
                 val topAlpha: Double) extends StaticDomainPruner {

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


object InitPruner {
  
  val overRidePruningAtInit = true;
/*  
  def main(args: Array[String]) {
    val trainMax = loadMatrixFromSvmFile((new File("./model", "initprn_train.dat")).toString());
    loadGroups((new File("./model", "initprn_train.grp")).toString(), trainMax);
    val testMax = loadMatrixFromSvmFile((new File("./model", "initprn_test.dat")).toString());
    loadGroups((new File("./model", "initprn_test.grp")).toString(), testMax);
    
    val booster = performLearningGvienTrainTestDMatrix(trainMax, testMax, "xgb-initpruner-100.model"); 
  }
 */ 
  def main(args: Array[String]) {
    val trainMax = loadMatrixFromSvmFile((new File("./model", "initprn_dev_train.dat")).toString());
    loadGroups((new File("./model", "initprn_dev_train.grp")).toString(), trainMax);
    val testMax = loadMatrixFromSvmFile((new File("./model", "initprn_dev_test.dat")).toString());
    loadGroups((new File("./model", "initprn_dev_test.grp")).toString(), testMax);
    
    val booster = performLearningGvienTrainTestDMatrix(trainMax, testMax, "xgb-initpruner-dev.model"); 
  }
  
  def loadMatrixFromSvmFile(fn: String): DMatrix = {
    val spData = DataLoader.loadSVMFile(fn);
    val trainMax2 = new DMatrix(spData.rowHeaders, spData.colIndex, spData.data, JDMatrix.SparseType.CSR);
    trainMax2.setLabel(spData.labels);
    return trainMax2;
  }
  
  def loadGroups(fn: String, mtrx: DMatrix) {
    var lncnt = 0;
    val grps = new ArrayBuffer[Int]();
	  
    val fis = new FileInputStream(fn);
    val isr = new InputStreamReader(fis);//, Charset.forName("UTF-8"));
    val br = new BufferedReader(isr);
    var line: String = br.readLine();
    while (line != null) {
      val grpsz = Integer.parseInt(line);
    	grps += (grpsz);
    	lncnt += grpsz;
    	line = br.readLine();
    }
    if (mtrx.rowNum != lncnt) throw new RuntimeException("Row inconsistent:"+mtrx.rowNum +" != "+ lncnt);
    mtrx.setGroup(grps.toArray);
  }
  
  def dumpInitActionPrunerTrainingFiles(trains: Seq[AceJointTaskExample],
                                        tests: Seq[AceJointTaskExample],
                                        featIndexer: Indexer[String],
                                        unaryPruner: StaticDomainPruner,
                                        bsearcher: SearchBasedLearner,
                                        initPrunAlpha: Double) = {
    
    val file = new File("./model");
    //val trnDataf = new PrintWriter(new File(file, "initprn_train.dat"));
    //val tstDataf = new PrintWriter(new File(file, "initprn_test.dat"));
    //val trnGroupf = new PrintWriter(new File(file, "initprn_train.grp"));
   // val tstGroupf = new PrintWriter(new File(file, "initprn_test.grp"));
    val trnDataf = new PrintWriter(new File(file, "initprn_dev_train.dat"));
    val tstDataf = new PrintWriter(new File(file, "initprn_dev_test.dat"));
    val trnGroupf = new PrintWriter(new File(file, "initprn_dev_train.grp"));
    val tstGroupf = new PrintWriter(new File(file, "initprn_dev_test.grp"));
    
    
    println("Start init state domain pruner training!");
    val dumpyWeight = new Array[Double](featIndexer.size());
    
    var trnCnt = 0;
    var tstCnt = 0;
    
    var cnt = 0;
    println("Dumping training data...");
    for (example <- trains) {
      cnt += 1;
      if (cnt % 10 == 0) {
        println("dump " + cnt + " rank lists.");
      }
      val initState = SearchBasedLearner.getInitStateWithUnaryScore(example, unaryPruner, overRidePruningAtInit);
      val initWithMusk = bsearcher.prunedActionSpaceMusk(example, initState);
      
      val corefRightLink = example.computeCorefRightLink(initState.output);
        
      val actions = bsearcher.actionGenerationWithMusk(example, initState, dumpyWeight, false, true);
      println(actions.size);
      
      for (act <- actions) {
    	  //doActionGetNewState(example, initState, act);
    	  val feat = example.getIncrementalFeatureByAction(initState.output, act, corefRightLink);
    	  //val feat = example.featurize(initState.output);
    	  //println(feat.size);
    	  // undoAction(example, initState, act);
    	  val isRelav = isRelaventAction(example, act);
    	  val label = if (isRelav) {  1; } else { 0; }
    	  rankElemToStr(trnDataf, feat, label);
      }

      println("RankListLen = " + actions.size);
      trnGroupf.println(actions.size);
      trnCnt += actions.size;
      //val alist = contructInitActionRankList(example, initState, actions);
    }
    
    cnt = 0;
    println("Dumping testing data...");
    for (exmp <- tests) {
    	cnt += 1;
    	if (cnt % 10 == 0) {
    		println("dump " + cnt + " rank lists.");
    	}
      val initState = SearchBasedLearner.getInitStateWithUnaryScore(exmp, unaryPruner, overRidePruningAtInit);
      val initWithMusk = bsearcher.prunedActionSpaceMusk(exmp, initState);
      val corefRightLink = exmp.computeCorefRightLink(initState.output);
      val actions = bsearcher.actionGenerationWithMusk(exmp, initState, dumpyWeight, false, true);

      for (act <- actions) {
    	  //doActionGetNewState(exmp, initState, act);
    	  //val feat = exmp.featurize(initState.output);
    	  val feat = exmp.getIncrementalFeatureByAction(initState.output, act, corefRightLink);
    	  //undoAction(exmp, initState, act);

    	  val isRelav = isRelaventAction(exmp, act);
    	  val label = if (isRelav) {  1; } else { 0; }
    	  rankElemToStr(tstDataf, feat, label);
      }

      println("RankListLen = " + actions.size);
      tstGroupf.println(actions.size);
      tstCnt += actions.size;
    }
    
    println("Total training points: " + trnCnt);
    println("Total testing points: " + tstCnt);

    println("Done data dumping ...");
    
    trnDataf.close();
    tstDataf.close();
    trnGroupf.close();
    tstGroupf.close();
  }
  
  def rankElemToStr(prner: PrintWriter, feat: HashMap[Int, Double], lb: Int) {
    val sb = new StringBuilder("");
    sb.append(lb);
    /*val featIdxs = feat.toSeq.sortWith{_ < _};
    // dump features
    for (i <- featIdxs) {
    	val fidx = i + 1;
    	val value = featIdxs(i);
    	sb.append(" " + fidx + ":" + value);
    }*/
    val featIdxs = feat.keySet.toSeq.sortWith{_ < _};
    // dump features
    for ((i,v) <- feat) {
    	val fidx = i + 1;
    	val value = v;//featIdxs(i);
    	sb.append(" " + fidx + ":" + value);
    }
    prner.println(sb.toString());
  }
  
/*
  def runInitActionPrunerTraining(trains: Seq[AceJointTaskExample],
                                  tests: Seq[AceJointTaskExample],
                                  featIndexer: Indexer[String],
                                  unaryPruner: StaticDomainPruner,
                                  bsearcher: SearchBasedLearner,
                                  initPrunAlpha: Double) = {
    
    println("Start init state domain pruner training!");
    val dumpyWeight = new Array[Double](featIndexer.size());
    
    var cnt = 0;
    val allTrainRankLists = new ArrayBuffer[ArrayBuffer[ActionRankElement]]();
    for (example <- trains) {
      cnt += 1;
      if (cnt % 100 == 0) {
        println("dump " + cnt + " rank lists.");
      }
      val initState = SearchBasedLearner.getInitStateWithUnaryScore(example, unaryPruner);
      val initWithMusk = bsearcher.prunedActionSpaceMusk(example, initState);
        
      val actions = bsearcher.actionGenerationWithMusk(example, initState, dumpyWeight, false, true);

      //val predBestOutput = beamSearch(example, initWithMusk, beamSize, weight, false, false).output;
      //val gdinit = SearchBasedLearner.getGoldInitState(example);
      //val goldInitMask = constructGoldMusk(example, gdinit, predBestOutput);
      //val goldBestOutput = beamSearch(example, goldInitMask, beamSize, weight, true, false).output;
      
      
      val alist = contructInitActionRankList(example, initState, actions);
      allTrainRankLists += alist;
    }
    
    cnt = 0;
    val allTestRankLists = new ArrayBuffer[ArrayBuffer[ActionRankElement]]();
    for (exmp <- tests) {
    	cnt += 1;
    	if (cnt % 100 == 0) {
    		println("dump " + cnt + " rank lists.");
    	}
      val initState = SearchBasedLearner.getInitStateWithUnaryScore(exmp, unaryPruner);
      val initWithMusk = bsearcher.prunedActionSpaceMusk(exmp, initState);
      val actions = bsearcher.actionGenerationWithMusk(exmp, initState, dumpyWeight, false, true);

      val alist2 = contructInitActionRankList(exmp, initState, actions);
      allTestRankLists += alist2;
    }
    
/*    
    val (trainMtrx, trWidth) = actionsRankListToDMatrix(allTrainRankLists, true);
    val (testMtrx, tstWidth) = actionsRankListToDMatrix(allTestRankLists, true);
	  
    
    val file = new File("./model");
    trainMtrx.saveBinary(file.getAbsolutePath + "/ininpruner_train.buffer")
    testMtrx.saveBinary(file.getAbsolutePath + "/ininpruner_test.buffer")
*/
    //// train
    println("Start pruner training ...");
	  //val bstr = performLearningGvienTrainTestDMatrix(trainMtrx, testMtrx);

    //testPrunerPrecisionRecall(allTestRankLists,bstr, 0.1);
  }
*/
  def runInitActionPrunerTesting(//trains: Seq[AceJointTaskExample],
                                  tests: Seq[AceJointTaskExample],
                                  featIndexer: Indexer[String],
                                  unaryPruner: StaticDomainPruner,
                                  bsearcher: SearchBasedLearner,
                                  initPrunAlpha: Double,
                                  bstr: Booster) = {
    
    println("Test the init state domain pruner! ==> Alpha = " + initPrunAlpha);
    val dumpyWeight = new Array[Double](featIndexer.size());

    var cnt = 0;
    val allTestRankLists = new ArrayBuffer[ArrayBuffer[ActionRankElement]]();
    for (exmp <- tests) {
    	cnt += 1;
    	if (cnt % 100 == 0) {
    		println("Get " + cnt + " rank lists.");
    	}
      val initState = SearchBasedLearner.getInitStateWithUnaryScore(exmp, unaryPruner, overRidePruningAtInit);
      val initWithMusk = bsearcher.prunedActionSpaceMusk(exmp, initState);
      val actions = bsearcher.actionGenerationWithMusk(exmp, initState, dumpyWeight, false, true);
      val corefRightLink = exmp.computeCorefRightLink(initState.output);
      
      val alist2 = contructInitActionRankList(exmp, initState, corefRightLink, actions);
      predictOneRankList(alist2, bstr);
      alist2.map { x => x.clearFeature(); }
      
      allTestRankLists += alist2;
    }

    //// train
    println("Start pruner testing ...");
    /*
    testPrunerPrecisionRecallWithScores(tests, allTestRankLists, 1.0);
    testPrunerPrecisionRecallWithScores(tests, allTestRankLists, 0.1);
    testPrunerPrecisionRecallWithScores(tests, allTestRankLists, 0.2);
    testPrunerPrecisionRecallWithScores(tests, allTestRankLists, 0.3);
    testPrunerPrecisionRecallWithScores(tests, allTestRankLists, 0.4);
    testPrunerPrecisionRecallWithScores(tests, allTestRankLists, 0.5);
    testPrunerPrecisionRecallWithScores(tests, allTestRankLists, 0.6);
    testPrunerPrecisionRecallWithScores(tests, allTestRankLists, 0.7);
    testPrunerPrecisionRecallWithScores(tests, allTestRankLists, 0.8);
    testPrunerPrecisionRecallWithScores(tests, allTestRankLists, 0.9);
		*/
    testPrunerPrecisionRecallWithScores(tests, allTestRankLists, initPrunAlpha);
  }
  
  def clearExamplePruneFlag(ex: AceJointTaskExample) {
    val corefVars = ex.corefVars;
    for (i <- 0 until corefVars.size) {
      corefVars(i).values.map { v => v.isPruned = false }
    }
    val nerVars = ex.nerVars
    for (i <- 0 until nerVars.size) {
      nerVars(i).values.map { v => v.isPruned = false }
    }
    val wikiVars = ex.wikiVars;
    for (i <- 0 until wikiVars.size) {
      wikiVars(i).values.map { v => v.isPruned = false }
    }
  }
  
  def testPrunerPrecisionRecallWithScores(exs: Seq[AceJointTaskExample],
                                          allTestRankLists: ArrayBuffer[ArrayBuffer[ActionRankElement]],
                                          //booster: Booster, 
                                          alpha: Double) {
    var truPos = 0;
    var truNeg = 0;
    var flsPos = 0;
    var flsNeg = 0;
    var crr = 0;
    var wrg = 0;
    
    //for (alist <- allTestRankLists) {
    for (i <- 0 until allTestRankLists.size) {
      val alist = allTestRankLists(i);
      val exmp = exs(i);
      clearExamplePruneFlag(exmp);
      //predictOneRankList(alist, booster);
      
      ////
      for (i <- 0 until alist.size) {
        alist(i).shouldBePruned = true;
      }
    
      // sort!
      val sortv = (alist.toSeq.sortWith(_.rankScore > _.rankScore)).toArray;
      //for (decs <- sortc) {
      //  println("c: " + decs.score + " " + decs.isCorrect);
      //}

      //var topk = 100;//(alist.size.toDouble * alpha).toInt;
      var topk = (alist.size.toDouble * alpha).toInt;
      if (topk < 1) topk = 1; 
      val nonPrunedValuesNumber = if (sortv.size > topk) topk else sortv.size
      for (j <- 0 until nonPrunedValuesNumber) {
        sortv(j).shouldBePruned = false;
      }
      
      //// apply pruning here!
      for (actElem <- alist) {
        val action = actElem.action;
        val varible = exmp.getVariableGivenIndex(action.idx);
        val values = varible.values;
        values(action.undoOldValue).isPruned = false;
        values(action.newValIdx).isPruned = actElem.shouldBePruned;
      }
      
      
      
      /////// compute accuracy
      for (aelmt <- sortv) {
        if ((aelmt.shouldBePruned == false) && (aelmt.label == 1)) {
          truPos += 1; crr += 1;
        } else if ((aelmt.shouldBePruned == true) && (aelmt.label == 0)) {
          truNeg += 1; crr += 1;
        } else if ((aelmt.shouldBePruned == false) && (aelmt.label == 0)) {
          flsPos += 1; wrg += 1;
        } else if ((aelmt.shouldBePruned == true) && (aelmt.label == 1)) {
          flsNeg += 1; wrg += 1;
        }
      }
    }
    
    val pre = (truPos.toDouble) / (truPos.toDouble + flsPos.toDouble);
    val rec = (truPos.toDouble) / (truPos.toDouble + flsNeg.toDouble);
    val f1 = 2.0 / ((1.0 / rec).toDouble + (1.0 / pre).toDouble);
    
    println("Alpha = " + alpha + ": " + "Precision = " + pre + " Recall = " + rec + " F1 = " + f1);
    StaticDomainPruner.noCorrectCountJoint(exs);
  }
  
    // return best value 
  def predictOneRankList(rankList: ArrayBuffer[ActionRankElement], booster: Booster) {
    val oneInstanMtrx = singleListToDMatrix(rankList, false);
    val predicts2 = booster.predict(oneInstanMtrx);
		for (j <- 0 until rankList.size) {
		  rankList(j).rankScore = predicts2(j)(0);
		}
  }
  
  def contructInitActionRankList(example: AceJointTaskExample, initState: SearchState, corefRightLink:Array[ArrayBuffer[Int]], actions: ArrayBuffer[SearchAction]): ArrayBuffer[ActionRankElement] = {
    val rankElms = new ArrayBuffer[ActionRankElement]();
    for (act <- actions) {
      val feat = example.getIncrementalFeatureByAction(initState.output, act, corefRightLink);
      val isRelav = isRelaventAction(example, act);
      val label = if (isRelav) {
        1; // correct
      } else {
        0; // wrong
      }
      
      val aElem = new ActionRankElement(act, label, feat);
      rankElms += aElem;
    }
    rankElms;
  }
  
  def doAction(example: AceJointTaskExample, state: SearchState, action: SearchAction) {
    action.undoIndex = action.idx;
    action.undoOldValue = state.output(action.idx);
    state.output(action.idx) = action.newValIdx;
  }
  
  def undoAction(example: AceJointTaskExample, state: SearchState, action: SearchAction) {
    state.output(action.idx) = action.undoOldValue;
  }
  
  def doActionGetNewState(example: AceJointTaskExample, state: SearchState, action: SearchAction) = {
    val newState = state.getSelfCopy();
    action.undoIndex = action.idx;
    action.undoOldValue = state.output(action.idx);
    newState.output(action.idx) = action.newValIdx;
    newState;
  }
  
  
  def trainPruner() {
    
    
    
  }
  
  // for training only
  def isRelavantState() = {
        
  }
  
  def isRelaventAction(jointExample: AceJointTaskExample, action: SearchAction): Boolean = {
    val ivariable = jointExample.getVariableGivenIndex(action.idx);
    if (ivariable.values(action.newValIdx).isCorrect) {
      return true;
    } else {
      return false;
    }
  }
  
  
/*
  def predictOneDocBooster(featurizer: PairwiseIndexingFeaturizer, docGraph: DocumentGraph, booster: Booster): Array[Int] = {
    val featsChart = docGraph.featurizeIndexNonPrunedUseCache(featurizer);
    val decisions = CorefTesting.extractDeciExmlOneDoc(docGraph);
    
    val result = new Array[Int](decisions.length);
    for (i <- 0 until decisions.length) {
      result(i) = predictDecision(decisions(i), booster);
      val crr = decisions(i).isCorrect(result(i));
    }
    result;
  }
  
  // return best value 
  def predictDecision(decision: CorefDecisionExample, booster: Booster): Int = {
    
    val (oneInstanMtrx, wdt) = XgbMatrixBuilder.corefExmpToDMatrix(Seq(decision), -1, false);
    val predicts2 = booster.predict(oneInstanMtrx);
    
    var bestLbl = -1;
		var bestScore = -Double.MaxValue;
		for (j <- 0 until decision.values.length) {
			val l = decision.values(j);
			var score = predicts2(j)(0);
			if (score > bestScore) {
				bestScore = score;
				bestLbl = j;
			}
		}
		decision.values(bestLbl);
  }
*/
  
  def performLearningGvienTrainTestDMatrix(trainMax: DMatrix, testMax: DMatrix, mdName: String): Booster = {
    
    //// train
	  
    println("Trainset size: " + trainMax.rowNum);
	  println("Testset size: " + testMax.rowNum);
    
    val params = new HashMap[String, Any]()
    val round = 10;//370
    //params += "distribution" -> "bernoulli"
    params += "eta" -> 0.1
    params += "max_depth" -> 100
    params += "silent" -> 0
    //params += "colsample_bytree" -> 0.9
    //params += "min_child_weight" -> 10
    params += "objective" -> "rank:pairwise"
    params += "eval_metric" -> "pre@100"
    params += "nthread" -> 4
      
    val watches = new HashMap[String, DMatrix]
    watches += "train" -> trainMax
    watches += "test" -> testMax


    // train a model
    val booster = XGBoost.train(trainMax, params.toMap, round, watches.toMap)
    
    //val bestScore = booster.
    //val bestIteration = model.best_iteration + 1   // note that xgboost start building tree with index 0
    //print("best_score: %s" % best_score)
    //print("opmital # of trees: %s" % best_iteration)
    
    // predict
    val predicts = booster.predict(testMax)
    // save model to model path
    val file = new File("./model")
    if (!file.exists()) {
      file.mkdirs()
    }
    booster.saveModel(file.getAbsolutePath + "/" + mdName);//"/xgb-initpruner-100.model")
    // dump model
    //booster.getModelDump(file.getAbsolutePath + "/dump.raw.txt", true)
    // dump model with feature map
    //booster.getModelDump(file.getAbsolutePath + "/featmap.txt", true)
    // save dmatrix into binary buffer
    trainMax.saveBinary(file.getAbsolutePath + "/dtrain.buffer")
    testMax.saveBinary(file.getAbsolutePath + "/dtest.buffer")
    
    booster;
  }
  
  
    
  def actionsRankListToDMatrix(allRankLists: ArrayBuffer[ArrayBuffer[ActionRankElement]], verbose: Boolean): (DMatrix, Int) = {

    var totalCnt = 0;
    var posCnt = 0;
    var negCnt = 0;
    var maxFeatIdx = 0;
    val tlabels = new ArrayBuffer[Float]();
    val tdata   = new ArrayBuffer[Float]();
    val theaders = new ArrayBuffer[Long]();
    val tindex = new ArrayBuffer[Int]();
    val tgroup = new ArrayBuffer[Int]();

    var gpCnt: Int = 0;
    var rowCnt: Int = 0;
    var rowheader:Long = 0L;
    theaders += (rowheader);
    
    for (alist <- allRankLists) {
      gpCnt += 1;
      
      if (gpCnt % 10000 == 0) {
        println("Feature " + gpCnt + " coref ments.");
      }
      
      for (j <- 0 until alist.size) {
        rowCnt += 1;
        // for jth value
        val aelem = alist(j);
        val featIdxs = aelem.feature.keySet.toSeq.sortWith{_ < _};
        val lbl = aelem.label;//if (ment.isCorrect(ante)) 1.0f else 0.0f
        // dump features
        for (i <- featIdxs) {
          val fidx = i + 1;
          val value = featIdxs(i);
    			tdata += (value);
    			tindex += (fidx);
    			if (i > maxFeatIdx) {
    			  maxFeatIdx = i;
    			}
    		}
        
        var totalFeats = featIdxs.length.toLong;

    		rowheader += totalFeats;
    		theaders += (rowheader);
    		tlabels += (lbl);
    	}
      tgroup += (alist.size);
    }
    
    val splabels: Array[Float] = tlabels.toArray;
    val spgroups: Array[Int] = tgroup.toArray;
    val spdata: Array[Float] = tdata.toArray;
    val spcolIndex: Array[Int] = tindex.toArray;
    val sprowHeaders: Array[Long] = theaders.toArray;
    
    if (verbose) {
    	println("splabels = " + splabels.length);
    	println("spgroups = " + spgroups.length);
    	println("spdata = " + spdata.length);
    	println("spcolIndex = " + spcolIndex.length);
    	println("sprowHeaders = " + sprowHeaders.length);
    }
    
    
    val mx = new DMatrix(sprowHeaders, spcolIndex, spdata, JDMatrix.SparseType.CSR);
    mx.setLabel(splabels);
    mx.setGroup(spgroups);
    
    // print some statistics
    if (verbose) {
    	println("Groups: " + gpCnt);
    	println("Rows: " + rowCnt);
    	println("Max feature index: " + maxFeatIdx);
    }

    (mx, maxFeatIdx);
  }
  
  def singleListToDMatrix(alist: ArrayBuffer[ActionRankElement], verbose: Boolean): DMatrix = {

    var totalCnt = 0;
    var posCnt = 0;
    var negCnt = 0;
    var maxFeatIdx = 0;
    val tlabels = new ArrayBuffer[Float]();
    val tdata   = new ArrayBuffer[Float]();
    val theaders = new ArrayBuffer[Long]();
    val tindex = new ArrayBuffer[Int]();
    val tgroup = new ArrayBuffer[Int]();

    var gpCnt: Int = 0;
    var rowCnt: Int = 0;
    var rowheader:Long = 0L;
    theaders += (rowheader);
    
    for (j <- 0 until alist.size) {
    	rowCnt += 1;
    	// for jth value
    	val aelem = alist(j);
    	//val featIdxs = aelem.feature.keySet.toSeq.sortWith{_ < _};
    	val lbl = aelem.label;//if (ment.isCorrect(ante)) 1.0f else 0.0f
    	// dump features
    	//for (i <- featIdxs) {
    	for ((i,v) <- aelem.feature) {
    		val fidx = i + 1;
    		val value = v.toFloat;//featIdxs(i);
    		tdata += (value);
    		tindex += (fidx);
    		if (i > maxFeatIdx) {
    			maxFeatIdx = i;
    		}
    	}

    	var totalFeats = aelem.feature.size.toLong;

    	rowheader += totalFeats;
    	theaders += (rowheader);
    	tlabels += (lbl);
    }
    tgroup += (alist.size);
    
    val splabels: Array[Float] = tlabels.toArray;
    val spgroups: Array[Int] = tgroup.toArray;
    val spdata: Array[Float] = tdata.toArray;
    val spcolIndex: Array[Int] = tindex.toArray;
    val sprowHeaders: Array[Long] = theaders.toArray;
    
    if (verbose) {
    	println("splabels = " + splabels.length);
    	println("spgroups = " + spgroups.length);
    	println("spdata = " + spdata.length);
    	println("spcolIndex = " + spcolIndex.length);
    	println("sprowHeaders = " + sprowHeaders.length);
    }
    
    
    val mx = new DMatrix(sprowHeaders, spcolIndex, spdata, JDMatrix.SparseType.CSR);
    mx.setLabel(splabels);
    mx.setGroup(spgroups);
    
    // print some statistics
    if (verbose) {
    	println("Groups: " + gpCnt);
    	println("Rows: " + rowCnt);
    	println("Max feature index: " + maxFeatIdx);
    }

    
    (mx);
  }
  
  
  
  
  
  
  
  
  
  


  // do not perform loss-augmented inference
  // for pruner learning only
  def beamPrunerSearch(jointExample: AceJointTaskExample, 
                       initState: SearchState, 
                       beamSize: Int, 
                       costWeight: Array[Double], 
                       useGold: Boolean,
                       zrbkeys: ZobristKeys) = {
    
    val dumpySearcher = new SearchBasedLearner(zrbkeys);
    val verbose = false;

    jointExample.updateValueScoreGivenWeight(costWeight);
    
    val learnRate = 0.1;
    val lambda = 1e-8;
    var updateCnt: Int = 0;

    val hashTb = new SearchHashTable(zrbkeys);
    val trajectory = new Array[SearchState](2000);
    val beam = new SearchBeam(beamSize, new StatePredComparator);
    val maxDepth = jointExample.totalSize * 500;
    var depth = 0;
    var flatStep = 0;

    // initialization!
    var currentState = initState.getSelfCopy();
    val computedScore = SingleTaskStructTesting.computeScoreSparse(costWeight, jointExample.featurize(currentState.output));
    currentState.cachedPredScore = computedScore;
    currentState.cachedTrueLoss = dumpySearcher.getTrueAcc(jointExample, currentState);
    
    trajectory(0) = currentState;
    hashTb.insertNewInstance(currentState);

    var lastDepth: Int = 1;
    breakable {
      
      val actCnt = 0;
      
    	for (step <- 1 until maxDepth) {

    	  lastDepth = step;
    		//if (step % 1 == 0) {
    			//println("Greedy search depth: " + step);
    		  //println(step + ": " + currentState.cachedPredScore + ", " + currentState.cachedTrueLoss);
    		//}

    		//val actions = actionGenerationNormal(jointExample, currentState, weight, useGold, true);
        //val actions = actionGenerationWithMusk(jointExample, currentState, weight, useGold, true);
    		//val (bestAct, siblingsCnt, bestActScore) = fastFindGreedyBestAction(jointExample, currentState, hashTb, null, useGold, isLossAugment, weight); // for speedup!
    		val (topkActs, minRankedAct, topTrueActs, minTrueAct, siblingsCnt) = dumpySearcher.fastFindTopKAction(jointExample, currentState, hashTb, null, beam.beamSize, useGold, false, costWeight, step);

    		// insert new states
    		val betterActionsThanBeam = topkActs;
    		
    		for (tkAct <- betterActionsThanBeam) {
    			var newState = doActionGetNewState(jointExample, currentState, tkAct);
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
        
        val bestScore = bestState.cachedPredScore;
        val lastStepScore = trajectory(step - 1).cachedPredScore;

    		// shall we continue?
    		if (siblingsCnt == 0) {
    		  if (verbose) println("Stop search: no non-repreat actions!" + " depth = " + step + " useGold = " + useGold);
    			break; // no legal action!
    		}
    		if (bestScore < lastStepScore){
    		  if (verbose) println("Stop search: reached peak!" + " depth = " + step + " useGold = " + useGold);
    			break; // reach the hill peak!
    		} else if (bestScore == lastStepScore){
    			// reach a flat range, if this continues for 5 more step, break (local optimal)
    			flatStep += 1;
    			if (flatStep >= 5) {//jointExample.totalSize) {
    			  if (verbose) println("Stop search: No increasing for 5 steps!" + " depth = " + step + " useGold = " + useGold);
    				break;
    			}
    		}

    		// drop the state out of beam size
    		beam.keepTopKOnly();
    			
    		// pop the best
    		currentState = beam.popBest();

    		// store to the trajectory
    		trajectory(step) = currentState;
    		hashTb.insertNewInstance(currentState); // remember the extended state

    	}
		
    } // breakable

    currentState;
  }

  
  def dumpRankingList(dumpySearcher: SearchBasedLearner,
                      jointExample: AceJointTaskExample, 
                      currentState: SearchState, 
                      hashTb: SearchHashTable,
                      genActions: ArrayBuffer[SearchAction], 
                      topk: Int,
                      useGold: Boolean,
                      isLossAugment: Boolean,
                      weight: Array[Double]) {
    
    val actions = dumpySearcher.actionGenerationWithMusk(jointExample, currentState, weight, useGold, true);
    val oldStateHash = hashTb.computeIndex(currentState);
    
    for (act <- actions) {
    		dumpySearcher.performAction(jointExample, currentState, act); // do this action
    		val stateIsOk = (!hashTb.probeExistenceWithAction(currentState, oldStateHash, act));
    		if (stateIsOk) {
    		  
    		  val feat = jointExample.featurize(currentState.output);
    		  val isGood = isRelaventAction(jointExample, act);
    		  
    		  
    			  
    		}
    		dumpySearcher.cancelAction(jointExample, currentState, act); // undo action
    }
  }
  
  def dumpFeatToFile() {
    
  }
  
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
          val initState = SearchBasedLearner.getInitStateWithUnaryScore(example, unaryPruner,false);//.getRandomInitState(example);
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
      val initState = SearchBasedLearner.getInitStateWithUnaryScore(ex, pruner,false);
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
  def stateFeatureIncreamentCompute(jointExample: AceJointTaskExample, oldState: SearchState, action: SearchAction) = {//, newState: SearchState) {
    
    // about features
    val newFeatMap = oldState.cachedFeatureVector.clone();
    
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
            val incrementalFeature = stateFeatureIncreamentCompute(jointExample, currentState, act);
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
          val initState = SearchBasedLearner.getInitStateWithUnaryScore(example, unaryPruner,false);
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
  
  
}