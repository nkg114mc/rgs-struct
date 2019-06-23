package berkeleyentity.oregonstate

import java.io.File
import java.io.PrintWriter

import scala.annotation.migration
import scala.collection.JavaConverters.asScalaSetConverter
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap

import berkeleyentity.ConllDocReader
import berkeleyentity.Driver
import berkeleyentity.GUtil
import berkeleyentity.coref.CorefDoc
import berkeleyentity.coref.CorefDocAssembler
import berkeleyentity.coref.MentionPropertyComputer
import berkeleyentity.coref.NumberGenderComputer
import berkeleyentity.lang.Language
import berkeleyentity.ner.MCNerExample
import berkeleyentity.ner.MCNerFeaturizer
import berkeleyentity.sem.BrownClusterInterface
import berkeleyentity.wiki.WikipediaInterface
import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.futile.util.Counter
import edu.berkeley.nlp.futile.util.Logger
import edu.illinois.cs.cogcomp.sl.applications.cs_multiclass.LabeledMultiClassData
import edu.illinois.cs.cogcomp.sl.applications.cs_multiclass.MultiClassLabel
import edu.illinois.cs.cogcomp.sl.applications.cs_multiclass.MultiClassModel
import edu.illinois.cs.cogcomp.sl.core.AbstractFeatureGenerator
import edu.illinois.cs.cogcomp.sl.core.AbstractInferenceSolver
import edu.illinois.cs.cogcomp.sl.core.IInstance
import edu.illinois.cs.cogcomp.sl.core.IStructure
import edu.illinois.cs.cogcomp.sl.core.SLParameters
import edu.illinois.cs.cogcomp.sl.latentsvm.AbstractLatentInferenceSolver
import edu.illinois.cs.cogcomp.sl.latentsvm.LatentLearner
import edu.illinois.cs.cogcomp.sl.learner.LearnerFactory
import edu.illinois.cs.cogcomp.sl.util.FeatureVectorBuffer
import edu.illinois.cs.cogcomp.sl.util.IFeatureVector
import edu.illinois.cs.cogcomp.sl.util.WeightVector
import ml.dmlc.xgboost4j.java.{ DMatrix => JDMatrix }
import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.scala.XGBoost

class tmpExample(val exmp: MCNerExample, 
		             val feats: Array[Array[Int]]) extends IInstance {

  val labelSize = MCNerFeaturizer.StdLabelIndexer.size();
  var isCorrect = false;
  var needRelearn = false;
	def getGoldIdx() = {
		val goldTag = NerTesting.getGoldNerTag(exmp.goldLabel);
		val goldLabelIdx = MCNerFeaturizer.StdLabelIndexer.indexOf(goldTag);
		goldLabelIdx;
	}

}

/**
 * @author machao
 */
object NerTesting {


 
  def extractExamples(corefDocs: Seq[CorefDoc]) = {
    val exs = new ArrayBuffer[MCNerExample];
    var dCnt = 0;
    for (corefDoc <- corefDocs) {
      dCnt += 1;
      val rawDoc = corefDoc.rawDoc;
      val docName = rawDoc.docID
      for (i <- 0 until corefDoc.predMentions.size) {
        val pm = corefDoc.predMentions(i);
        val nerExmp = new MCNerExample(rawDoc.words(pm.sentIdx), rawDoc.pos(pm.sentIdx), rawDoc.trees(pm.sentIdx), pm.startIdx, pm.headIdx, pm.endIdx, pm.nerString);
        nerExmp.ment = pm;
        nerExmp.docID = dCnt;
        exs += nerExmp;
      }
    }
    Logger.logss(exs.size + " ner chunks");
    exs;
  }

 
  def NerTestingInterfaceACE() {
    
    val trainDataPath = "data/ace05/train";
    val devDataPath = "data/ace05/dev";
    val testDataPath = "data/ace05/test";
    val wikiPath = "data/ace05/ace05-all-conll-wiki"
    val wikiDBPath = "models/wiki-db-ace.ser.gz"
    
    Driver.numberGenderDataPath = "data/gender.data";
    Driver.brownPath = "data/bllip-clusters";
 
    val lambda = 1e-8F
    val batchSize = 1
    val numItrs = 20
    
    val dict = constructDictionary(trainDataPath, devDataPath, testDataPath);
    
    // Read in CoNLL documents
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    val assembler = CorefDocAssembler(Language.ENGLISH, true); //use gold mentions
    val trainDocs = ConllDocReader.loadRawConllDocsWithSuffix(trainDataPath, -1, "", Language.ENGLISH);
    val trainCorefDocs = trainDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));

    val wikiDB = GUtil.load(wikiDBPath).asInstanceOf[WikipediaInterface];
    
    // Make training examples, filtering out those with solutions that are unreachable because
    // they're not good for training
    val trainExs = extractExamples(trainCorefDocs);
    //val testExs = extractExamples(testCorefDocs, goldWikification, wikiDB, filterImpossible = true);
    
    println("ACE NER chunks: " + trainExs.size);
    
    // Extract features
    val featIndexer = new Indexer[String]
    val maybeBrownClusters = if (Driver.brownPath != "") Some(BrownClusterInterface.loadBrownClusters(Driver.brownPath, 0)) else None;
    val nerFeaturizer = MCNerFeaturizer(Driver.nerFeatureSet.split("\\+").toSet, featIndexer, MCNerFeaturizer.StdLabelIndexer, trainDocs.flatMap(_.words), Some(wikiDB), maybeBrownClusters);
    val nerAddiFeatr = new NerAdditionalFeature(featIndexer, MCNerFeaturizer.StdLabelIndexer, dict);
    
    
    var sumLen: Int = 0;
    var maxLen: Int = 0;
    var minLen: Int = 300000;
    var qid = 0;
    
    var allTrainFeats = new Array[Array[Array[Int]]](trainExs.size);
    var allTrains = new ArrayBuffer[tmpExample]();
    for (trainEx <- trainExs) {
      qid += 1;
      val featEachLabel = nerFeaturizer.featurizeAdd(trainEx, true, nerAddiFeatr);//nerFeaturizer.featurize(trainEx, true);
      allTrainFeats(qid - 1) = featEachLabel;
      allTrains += (new tmpExample(trainEx, featEachLabel));
      for (idx <- 0 until 7) {
        val len = featEachLabel(idx).length;
        sumLen = sumLen + len;
        if (maxLen < len) {
          maxLen = len;
        }
        if (minLen > len) {
          minLen = len
        }
      }
    }

    /////////////// for testing ///////////////
    val testDocs = ConllDocReader.loadRawConllDocsWithSuffix(testDataPath, -1, "", Language.ENGLISH);
    val testCorefDocs = testDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    val testExs = extractExamples(testCorefDocs);
    val testEmpExs = new ArrayBuffer[tmpExample]();
      for (testEx <- testExs) {
      val featEachLabel = nerFeaturizer.featurizeAdd(testEx, false, nerAddiFeatr);//nerFeaturizer.featurize(testEx, false);
      val thisTmp = new tmpExample(testEx, featEachLabel);
      testEmpExs += (thisTmp); // add to list
    }
      
    /////////////// for validating ////////////
    val devDocs = ConllDocReader.loadRawConllDocsWithSuffix(devDataPath, -1, "", Language.ENGLISH);
    val devCorefDocs = devDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    val devExs = extractExamples(devCorefDocs);
    val devEmpExs = new ArrayBuffer[tmpExample]();
      for (devEx <- devExs) {
      val featEachLabel = nerFeaturizer.featurizeAdd(devEx, false, nerAddiFeatr);//nerFeaturizer.featurize(devEx, false);
      val thisTmp = new tmpExample(devEx, featEachLabel);
      devEmpExs += (thisTmp); // add to list
    }
    
    println("Ner feature size = " + nerFeaturizer.featureIndexer.size());
    
    val allInst = allTrains ++ devEmpExs ++ testEmpExs;
    //NerAdditionalFeature.collectNerDictionary(allInst);
    //NerAdditionalFeature.collectNerDictionary(allTrains,testEmpExs);
    
    //NerAdditionalFeature.lengthCount(allInst);
    //NerAdditionalFeature.tokenTagCnt(allInst);
    //NerAdditionalFeature.wordTagCnt(allInst);

    //val wght = structurePerceptrion(allTrains ++ devEmpExs, featIndexer, testEmpExs, 100);
    // learn!
    val wght = structurePerceptrion(allTrains, featIndexer, testEmpExs, 50);
    //val wght = multiClassSVM(allTrains, featIndexer, testEmpExs);
    //val wght = trainXgb(allTrains, featIndexer, testEmpExs);
    
    //val wght = structurePerceptrion(allTrains ++ testEmpExs, featIndexer, testEmpExs, 50);
    //println("Weight length = " + wght.length);
    
    //val logger = new PrintWriter("ner_ace05_learning_curve_devset.csv");
    //computePrecisionRecallCurver(testEmpExs, wght, Some(logger));
    //logger.close();
    
    var avglen: Double = sumLen;
    avglen = avglen / ((trainExs.size).toDouble);
    Logger.logss(featIndexer.size + " features");
    Logger.logss("min = " + minLen + ", maxLen = " + maxLen + ", avgLen = " + avglen);
    // Train
    //val gt = new GeneralTrainer[JointQueryDenotationExample]();
    //val weights = gt.trainAdagrad(trainExs, computer, featIndexer.size, 1.0F, lambda, batchSize, numItrs);
    //val chooser = new JointQueryDenotationChooser(featIndexer, weights)

    //testAceNerSystem(allTrains, wght, None);
    //testAceNerSystem(devEmpExs, wght, None);
    testAceNerSystem(testEmpExs, wght, None, true);
/*
    var trnErrWriter = new PrintWriter("trainNerErrFeat.txt");
    var tstErrWriter = new PrintWriter("testNerErrFeat.txt");
    var devErrWriter = new PrintWriter("devNerErrFeat.txt");
    
    NerErrorFinder.dumpDecisionsAceNer(allTrains, wght, Some(trnErrWriter));
    NerErrorFinder.dumpDecisionsAceNer(testEmpExs, wght, Some(tstErrWriter));
    NerErrorFinder.dumpDecisionsAceNer(devEmpExs, wght, Some(devErrWriter));
    
    //NerErrorFinder.dumpDecisionsAceNerAll(allTrains, Some(trnErrWriter));
    //NerErrorFinder.dumpDecisionsAceNerAll(testEmpExs, Some(tstErrWriter));
    //NerErrorFinder.dumpDecisionsAceNerAll(devEmpExs, Some(devErrWriter));
    
    trnErrWriter.close();
    tstErrWriter.close();
    devErrWriter.close();
    
    
    // test error ranking
    //val errRankModel = "errfinder/ner_error_ace05_lambdamart.txt";
    //val errRankModel = "errfinder/ner_error_ace05_lambdamart_dev.txt";
    //val errRankModel = "errfinder/ner_error_ace05_lambdamart_2.txt";
    val errRankModel = "errfinder/ner_error_ace05_lambdamart_w0_global.txt";
    val ranker = new UMassRankLib();
    ranker.loadModelFile(errRankModel);
    //val trainRelearnExs = NerErrorFinder.computeUpperBoundWithErrFinder(allTrains, wght, ranker, false);
    //val testRelearnExs = NerErrorFinder.computeUpperBoundWithErrFinder(testEmpExs, wght, ranker, false);
    NerErrorFinder.computeUpperBoundGlobalFinder(allTrains, wght, ranker, false);
    NerErrorFinder.computeUpperBoundGlobalFinder(testEmpExs, wght, ranker, false);

    //val wght2 = structurePerceptrion(trainRelearnExs, featIndexer, testRelearnExs, 100);
    //testAceNerSystem(testRelearnExs, wght2, None);
*/
    Logger.logss("All Done!");
  }
 
  def testAceNerSystem(testExs: ArrayBuffer[tmpExample], 
                       weight: Array[Double],
                       mayLogger: Option[PrintWriter],
                       verbose: Boolean) {
    
    val confuseCnt = new Counter[String]();
    val typCnt = new Counter[String]();
    
    //val featSparsityCounter = new FeatureSparsity(weight.length);
    //featSparsityCounter.init();
    var total: Double = 0;
    var correct: Double = 0;
    for (testEx <- testExs) {
      val featEachLabel = testEx.feats;//nerFeaturizer.featurize(testEx, false);
      //outputRankingExamples(featWriter2, testEx, qid, featEachLabel);
      val goldLabelIdx = MCNerFeaturizer.StdLabelIndexer.indexOf(getGoldNerTag(testEx.exmp.goldLabel));
      var bestLbl = -1;
      var bestScore = -Double.MaxValue;
      for (l <- 0 until 7) {
        //featSparsityCounter.addOneSample(featEachLabel(l));
        var score = computeScore(weight, featEachLabel(l));
        if (score > bestScore) {
          bestScore = score;
          bestLbl = l;
        }
      }
      //println("bestscore = " + bestScore + ", bestLbl = " + bestLbl + ", gold = " + goldLabelIdx);
      total += 1;
      testEx.isCorrect = false;
      if (goldLabelIdx == bestLbl) {
          correct += 1;
          testEx.isCorrect = true;
      } else {
        if (verbose) {
          println("["+getSpanStr(testEx.exmp.words, testEx.exmp.startIdx, testEx.exmp.endIdx) + "] " + testEx.exmp.ment.mentionType.toString() + " " + getGoldMentTyp(testEx.exmp.ment.nerString) + "  " + MCNerFeaturizer.StdLabelIndexer.getObject(bestLbl) + " shouldbe " + MCNerFeaturizer.StdLabelIndexer.getObject(goldLabelIdx));
          val my = MCNerFeaturizer.StdLabelIndexer.getObject(bestLbl)
          var cr = MCNerFeaturizer.StdLabelIndexer.getObject(goldLabelIdx)
          confuseCnt.incrementCount(my + "-" + cr, 1.0);
          typCnt.incrementCount(testEx.exmp.ment.mentionType.toString(), 1.0);
        }
      }
      
      
      
      //NerErrorFinder.outputErrorExample(writer: PrintWriter, exmp: tmpExample, globalQid: Int, featBestLabel: Array[Int], testEx.isCorrect);
    }
    //featWriter.close();
    
    if (verbose) {
      println("========================");
      val allPairs = confuseCnt.getEntrySet.asScala.toList.sortWith(_.getValue > _.getValue);
      for (pr <- allPairs) {
    	//println(pr + ": " + confuseCnt.getCount(pr.getKey));
    	  println(pr);
      }
      println("========================");
      val allTyps = typCnt.getEntrySet.asScala.toList.sortWith(_.getValue > _.getValue);
      for (tp <- allTyps) {
        println(tp);
      }
    }
    
    //featSparsityCounter.printHistogram();
    val accuracy = correct / total;
    println("Total = " + total + ", correct = " + correct + ", acc = " + accuracy);
    //exampleByDocs(testExs); // coount by doc
    
    //if (mayLogger != None) {
    //  val writer = mayLogger.get;
    //  writer.println(total + ", " + correct + ", " + accuracy);
    //  writer.flush();
   // }
  }
  
  def getSpanStr(strArr: Seq[String], startIdx: Int, endIdx: Int) = {
    val sb = new StringBuilder("");
    for (i <- startIdx until endIdx) {
      sb.append(strArr(i));
      sb.append(" ");
    }
    sb.toString();
  }
  
  def constructDictionary(trainDataPath: String, devDataPath: String, testDataPath: String) = {

    // Read in CoNLL documents
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer), Some(new GoldMentionTypePredictor()));
    val assembler = CorefDocAssembler(Language.ENGLISH, true); //use gold mentions
    val trainDocs = ConllDocReader.loadRawConllDocsWithSuffix(trainDataPath, -1, "", Language.ENGLISH);
    val trainCorefDocs = trainDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));

    /////////////// for testing ///////////////
    val devDocs = ConllDocReader.loadRawConllDocsWithSuffix(devDataPath, -1, "", Language.ENGLISH);
    val devCorefDocs = devDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    /////
    val testDocs = ConllDocReader.loadRawConllDocsWithSuffix(testDataPath, -1, "", Language.ENGLISH);
    val testCorefDocs = testDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));

    
    val trainExs = extractExamples(trainCorefDocs);
    val devExs = extractExamples(devCorefDocs);
    val testExs = extractExamples(testCorefDocs);

    
    val allExs = trainExs ++ devExs ++ testExs;
    val tinsts = nerExmpToTmpExmpNoFeats(allExs);
    val dict = NerAdditionalFeature.wordTagCnt(tinsts);
    
    dict;
  }
  
  def nerExmpToTmpExmpNoFeats(exs: Seq[MCNerExample]) = {
    val emptyFeats = Array(Array[Int]());
    val texs = exs.map{ ex => new tmpExample(ex, emptyFeats) };
    texs;
  }
  
/*
  def computePrecisionRecallCurver(testExs: ArrayBuffer[tmpExample], 
                                   weight: Array[Double],
                                   mayLogger: Option[PrintWriter]) {
 
    var decisions = new ArrayBuffer[NerDecision]();
    
    var mid = 0;
    for (ex <- testExs) {
      mid += 1;
      val featEachLabel = ex.feats;
      val goldLabelIdx = MCNerFeaturizer.StdLabelIndexer.indexOf(NerTesting.getGoldNerTag(ex.exmp.goldLabel));
      var bestLbl = -1;
      var bestScore = -Double.MaxValue;
      
      val exmpDecis = new ArrayBuffer[NerDecision]();
      for (l <- 0 until 7) {
        val score = NerTesting.computeScore(weight, featEachLabel(l));
        val dc = (new NerDecision(ex, 1, mid, l, score, (goldLabelIdx == l)));
        dc.features = featEachLabel(l);
        exmpDecis += dc;
        
        if (score > bestScore) {
          bestScore = score;
          bestLbl = l;
        }
      }
      decisions = decisions ++ exmpDecis;//sortedExamDecis;
    }
    
    println("Decision sorting...");
    val sorted = (decisions.toSeq.sortWith(_.score > _.score)).toArray;
    println("Done sorting...");
    
    val allOne = testExs.length.toDouble;
    var corrOneCnt: Double = 0;
    for (j <- 0 until sorted.size) {
      val curCnt = (j + 1).toDouble;
      if (sorted(j).isCorrect) {
        corrOneCnt += 1.0;
      }
      
      val thres = sorted(j).score;
      val pre = corrOneCnt / curCnt;
      val rec = corrOneCnt / allOne;
      //println(j + "," + thres + "," + rec + "," + pre);
      
      if (mayLogger != None) {
        val writer = mayLogger.get;
        writer.println(j + "," + thres + "," + rec + "," + pre);
        writer.flush();
      }
    }
  }
*/
  def exampleByDocs(testExs: ArrayBuffer[tmpExample]) {
    // split by document
    val docExMap = new HashMap[Int, ArrayBuffer[tmpExample]]();
    for (testEx <- testExs) {
      val dname = testEx.exmp.docID;
      if (docExMap.contains(dname)) {
        docExMap(dname) += testEx;
      } else {
        val newArr = new ArrayBuffer[tmpExample]();
        newArr += testEx;
        docExMap += (dname -> newArr);
      }
    }
    
    var topkCnt = 0;
    val topk = 30;
    // count by document
    for ((k, vlist) <- docExMap) {
      val crrCnt = vlist.filter{ x => x.isCorrect }.size;
      val wngCnt = vlist.filter{ x => (!x.isCorrect) }.size;
      val tltCnt = vlist.size;
      //println("cw: " + crrCnt + "/" + wngCnt);
      if (vlist.size > topk) {
        topkCnt += 30;
      } else {
        topkCnt += vlist.size;
      }
    }
    
    println("allTopk count = " + topkCnt);
  }
  
  def getGoldNerTag(nerSymbol: String) = {
    if (nerSymbol.contains("-")) {
      nerSymbol.substring(0, nerSymbol.indexOf("-"));
    } else {
      nerSymbol
    }
  }
  
  def getGoldMentTyp(nerSymbol: String) = {
    if (nerSymbol == null) {
      "-"; // nothing
    }
    if (nerSymbol.contains("-")) {
      nerSymbol.substring(nerSymbol.indexOf("-") + 1, nerSymbol.length());
    } else {
      "-" // nothing
    }
  }
 
  def structurePerceptrion(allTrains: ArrayBuffer[tmpExample], 
                           featIndexer: Indexer[String],
                           testExs: ArrayBuffer[tmpExample],
                           Iteration: Int) : Array[Double] = {
   
    val logger = new PrintWriter("ner_ace05_learning_curve_devset.csv");
    
    
    var weight = Array.fill[Double](featIndexer.size)(0);
    var weightSum = Array.fill[Double](featIndexer.size)(0);
    val learnRate = 0.1;
    val lambda = 1e-8;
    
    var updateCnt = 0;
    
    for (iter <- 0 until Iteration) {
      println("Iter " + iter);
      for (example <- allTrains) {
        val goldTag = getGoldNerTag(example.exmp.goldLabel);
        val goldLabelIdx = MCNerFeaturizer.StdLabelIndexer.indexOf(goldTag);
        var bestLbl = -1;
        var bestScore = -Double.MaxValue;
        for (l <- 0 until 7) {
          var score = computeScore(weight, example.feats(l));
          if (score > bestScore) {
            bestScore = score;
            bestLbl = l;
          }
        }
        
        // update?
        if (bestLbl != goldLabelIdx) {
          updateCnt += 1;
          if (updateCnt % 1000 == 0) println("Update " + updateCnt);
          updateWeight(weight: Array[Double], 
                       example.feats(goldLabelIdx),
                       example.feats(bestLbl),
                       learnRate,
                       lambda);
          sumWeight(weightSum, weight);
        }
      }
    
      ///////////////////////////////////////////////////
      // have a test after each iteration (for learning curve)
      val tmpAvg = new Array[Double](weightSum.size)
      Array.copy(weightSum, 0, tmpAvg, 0, weightSum.size);
      divdeNumber(tmpAvg, updateCnt.toDouble);
      //testAceNerSystem(allTrains, tmpAvg, Some(logger));
      testAceNerSystem(testExs, tmpAvg, Some(logger), false);
    }
    
    divdeNumber(weightSum, updateCnt.toDouble);
   
    /*
    for (i <- 0 until weightSum.length) {
      if (weightSum(i) != 0) {
        println("weight(" + i + ") = " + weightSum(i));
      }
    }
    */
    //computePrecisionRecallCurver(testExs, weightSum, Some(logger));
    
    
    logger.close();
    
    weightSum;
    //weight;
  }
  
  def updateWeight(currentWeight: Array[Double], 
                   featGold: Array[Int],
                   featPred: Array[Int],
                   eta: Double,
                   lambda: Double) {
    var gradient = Array.fill[Double](currentWeight.length)(0);//new Array[Double](currentWeight.length);
    for (i <- featGold) {
      if (i >= 0) {
        gradient(i) += (1.0);
      }
    }
    for (j <- featPred) {
      if (j >= 0) {
        gradient(j) -= (1.0);
      }
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
  
  def computeScore(wght: Array[Double], feat: Array[Int]) : Double = {
    var result : Double = 0;
    for (idx <- feat) {
      if (idx >= 0) {
        result += (wght(idx));
      }
    }
    result;
  }
  
  def sumWeight(sum: Array[Double], w: Array[Double]) {
    for (i <- 0 until w.length) {
      sum(i) += w(i);
    }
  }
  
  def divdeNumber(w: Array[Double], deno: Double) {
    for (i <- 0 until w.length) {
      w(i) = (w(i) / deno);
    }
  }
  
  def getL1Norm(w: Array[Double]): Double = {
    var norm: Double = 0;
    for (v <- w) {
      norm += Math.abs(v);
    }
    return norm;
  }
  

  
  def outputRankingExamples(writer: PrintWriter, exmp: MCNerExample, globalQid: Int, featEachLabel: Array[Array[Int]]) {
    val goldTag = getGoldNerTag(exmp.goldLabel);
    val goldLabelIdx = MCNerFeaturizer.StdLabelIndexer.indexOf(goldTag);
    println(goldTag+" = "+goldLabelIdx);
    var includeTrue = false;
    for (l <- 0 until 7) {
        val feat = featEachLabel(l);
        val rank = if (goldLabelIdx == l) 1 else 0;
        
        // print!
        writer.print(rank);
        writer.print(" qid:" + globalQid);        
        for (i <- 0 until feat.length) {
          val idx = feat(i) + 1;
          writer.print(" " + idx + ":1.0");
        }
        writer.println();
        
        if (goldLabelIdx == l) {
          includeTrue = true;
        }
    }
    if (!includeTrue) {
      throw new RuntimeException("No ground truth!");
    }
  }
    
  def main(args: Array[String]) {
    NerTestingInterfaceACE();
  }
  
  //////////////////////////////////////
  //////////////////////////////////////
  //////////////////////////////////////
  
  // test with SVM
   def multiClassSVM(trainExs: ArrayBuffer[tmpExample], 
                     featIndexer: Indexer[String],
                     testExs: ArrayBuffer[tmpExample]) : Array[Double] = {
     
      val model = new MultiClassModel();
      
      val spTrain = extractInstances(trainExs, featIndexer, true);// MultiClassIOManager.readTrainingData(trainingDataPath);
      model.labelMapping = spTrain.labelMapping;
      model.numFeatures = spTrain.numFeatures;
      model.cost_matrix = getCostMatrix();//MultiClassIOManager.getCostMatrix(sp.labelMapping,costMatrixPath);
      
      val featGener = new UiucSLFeatureGenerator();//MultiClassFeatureGenerator();


      val configFilePath = "config/uiuc-sl-config/myDCD-ner.config";
      val para = new SLParameters();
      para.loadConfigFile(configFilePath);
      model.featureGenerator = featGener;
      
            // initialize the inference solver
      //model.infSolver = new NerClassInferenceSolver(model.cost_matrix, featGener);
      val latentInfslvr = new NerLatentInferenceSolver(model.cost_matrix, featGener);
      model.infSolver = latentInfslvr;
      val learner = LearnerFactory.getLearner(model.infSolver, model.featureGenerator, para);
      
      val latentPara = new SLParameters();
      latentPara.MAX_NUM_ITER = 10;
      val latentLearner = new LatentLearner(learner, featGener, latentPara, latentInfslvr);
      //model.wv = learner.train(spTrain);
      model.wv = latentLearner.train(spTrain);
      model.config = new java.util.HashMap[String, String]();
      
      // save the model
      val modelPath = "indep_ner_ace05.txt";
      model.saveModel(modelPath);
      
      ///////////////////////////////////////////////////

      val spTest = extractInstances(testExs, featIndexer, false);//MultiClassIOManager.readTestingData(testDataPath, model.labelMapping, model.numFeatures);

      var pred_loss : Double = 0.0;
      for (i <- 0 until spTest.size()) {
        val ri = spTest.instanceList.get(i).asInstanceOf[NerClassInstance];
        val pred = model.infSolver.getBestStructure(model.wv, ri).asInstanceOf[MultiClassLabel];
        val gner = spTest.goldStructureList.get(i).asInstanceOf[MultiClassLabel];
        pred_loss += model.cost_matrix(gner.output)(pred.output);
        //println(pred.output);
      }
      println("Loss = " + (pred_loss / spTest.size()));
      
      getDoubleWeight(model);
    }


   def getCostMatrix() = {
	   val nerIdxer = MCNerFeaturizer.StdLabelIndexer;
	   val costMat = Array.ofDim[Float](nerIdxer.size, nerIdxer.size);
	   for (i <- 0 until nerIdxer.size) {
		   for (j <- 0 until nerIdxer.size) {
			   if (i == j) {
				   costMat(i)(j) = 0.0F;
			   } else {
				   costMat(i)(j) = 1.0F;
			   }
		   }
	   }
	   costMat;
   }

   def extractInstances(exs: ArrayBuffer[tmpExample], 
		                    featIndexer: Indexer[String],
		                    isTrain: Boolean) = {

	   val nerIdxer = MCNerFeaturizer.StdLabelIndexer;
	   val labelsMapping = new java.util.HashMap[String, Integer]();
	   for (idx <- 0 until nerIdxer.size) {
		   labelsMapping.put(nerIdxer.getObject(idx), new Integer(idx));
	   }
	   val res = new LabeledMultiClassData(labelsMapping, featIndexer.size);

	   for (ex <- exs) {
		   val mi = new NerClassInstance(ex.feats, featIndexer.size, nerIdxer.size, ex.getGoldIdx);
		   res.instanceList.add(mi);
		   res.goldStructureList.add(new MultiClassLabel(mi.goldLabelIdx));
	   }

	   res;
   }
   
   def getDoubleWeight(modl: MultiClassModel): Array[Double] = {
		   val wv = modl.wv;
		   val farr = wv.getWeightArray; // (0) is bias
		   var darr = new Array[Double](farr.length - 1);
		   for (i <- 0 until darr.length) {
			   darr(i) = farr(i + 1).toDouble;
		   }
		   darr;
   }

   class UiucSLFeatureGenerator() extends AbstractFeatureGenerator {

	   override def getFeatureVector(x: IInstance, y: IStructure): IFeatureVector = {
			   val mx = x.asInstanceOf[NerClassInstance];
			   val my = y.asInstanceOf[MultiClassLabel];

			   val fb = new FeatureVectorBuffer();
			   val valMap = mx.featurize(my.output);
			   for (idx <- valMap.keySet) {
				   //val normVal = if (normalizedDem == 0) { 0.0d } else {  / normalizedDem }
				   fb.addFeature(idx + 1, valMap(idx));
			   }
			   fb.toFeatureVector();
	   }
	   
	   override def getFeatureVectorDiff(x: IInstance, y1: IStructure, y2: IStructure): IFeatureVector = {
			   val f1 = getFeatureVector(x, y1);
			   val f2 = getFeatureVector(x, y2);		
			   return f1.difference(f2);
	   }
     
   }
   
   class NerClassInstance(val fvMatrix: Array[Array[Int]],
                          val baseNfeature: Int,
                          val numberOfClasses: Int, 
                          val goldLabelIdx: Int) extends IInstance {
  
	   def featurize(labelIdx: Int) = {
        getSparseFeatMap(fvMatrix(labelIdx));
	   }

	   def getSparseFeatMap(feat: Array[Int]) = {
        val vMap = new HashMap[Int,Double]();
        for (i <- feat) {
          addValueToVector(vMap, i, 1.0);
        }
        vMap;
	   }
     
     def addValueToVector(myMap: HashMap[Int,Double], index: Int, value: Double) {
      val maybeVal = myMap.get(index);
      if (maybeVal == None) {
        myMap += (index -> 1.0);
      } else {
        val newV = maybeVal.get + 1.0;
        myMap += (index -> newV);
      }
    }

  }
   
   
  
   class NerClassInferenceSolver(lossMatrix: Array[Array[Float]],
                                 featureGener: UiucSLFeatureGenerator) extends AbstractInferenceSolver {


	   override def getLossAugmentedBestStructure(weight: WeightVector, 
                                                ins: IInstance, 
                                                goldStructure: IStructure) : IStructure = {

			   val mi = ins.asInstanceOf[NerClassInstance];;
			   val lmi = goldStructure.asInstanceOf[MultiClassLabel];

			   var bestOutput: Int = -1;
			   var bestScore: Float = -Float.MaxValue;

			   for (i <- 0 until mi.numberOfClasses) {
           val currentl = new MultiClassLabel(i);
				   var score = weight.dotProduct(featureGener.getFeatureVector(mi, currentl));

				   if ((lmi != null) && (i != lmi.output)){
					   if (lossMatrix == null)
						   score += 1.0F;
					   else
						   score += lossMatrix(lmi.output)(i);
				   }               

				   if (score > bestScore){
					   bestOutput = i;
					   bestScore = score;
				   }
			   }

			   (new MultiClassLabel(bestOutput));   
	   }

	   override def getBestStructure(weight: WeightVector, ins: IInstance) = {
		   getLossAugmentedBestStructure(weight, ins, null);
	   }

	   override def getLoss(ins: IInstance, gold: IStructure, pred: IStructure) = {
		   var loss = 0.0F;
		   val lmi = gold.asInstanceOf[MultiClassLabel];
		   val pmi = pred.asInstanceOf[MultiClassLabel];
		   if (pmi.output != lmi.output){
			   if (lossMatrix == null)
				   loss = 1.0f;
			   else
				   loss = lossMatrix(lmi.output)(pmi.output);        
		   }
		   loss;
	   }

   }
   
   
   
   class NerLatentInferenceSolver(lossMatrix: Array[Array[Float]],
                                  featureGener: UiucSLFeatureGenerator) extends AbstractLatentInferenceSolver {


	   override def getLossAugmentedBestStructure(weight: WeightVector, 
                                                ins: IInstance, 
                                                goldStructure: IStructure) : IStructure = {

			   val mi = ins.asInstanceOf[NerClassInstance];;
			   val lmi = goldStructure.asInstanceOf[MultiClassLabel];

			   var bestOutput: Int = -1;
			   var bestScore: Float = -Float.MaxValue;

			   for (i <- 0 until mi.numberOfClasses) {
           val currentl = new MultiClassLabel(i);
				   var score = weight.dotProduct(featureGener.getFeatureVector(mi, currentl));

				   if ((lmi != null) && (i != lmi.output)){
					   if (lossMatrix == null)
						   score += 1.0F;
					   else
						   score += lossMatrix(lmi.output)(i);
				   }               

				   if (score > bestScore){
					   bestOutput = i;
					   bestScore = score;
				   }
			   }

			   (new MultiClassLabel(bestOutput));   
	   }

	   override def getBestStructure(weight: WeightVector, ins: IInstance) = {
		   getLossAugmentedBestStructure(weight, ins, null);
	   }

	   override def getLoss(ins: IInstance, gold: IStructure, pred: IStructure) = {
		   var loss = 0.0F;
		   val lmi = gold.asInstanceOf[MultiClassLabel];
		   val pmi = pred.asInstanceOf[MultiClassLabel];
		   if (pmi.output != lmi.output){
			   if (lossMatrix == null)
				   loss = 1.0f;
			   else
				   loss = lossMatrix(lmi.output)(pmi.output);        
		   }
		   loss;
	   }
	   
	   override def getBestLatentStructure(weight: WeightVector, ins: IInstance, gold: IStructure) = {
	     gold;
	   }

   }

   
   /////////////////////////////////////////////////////////////////////////////////////////
   ///// Xgb Learning //////////////////////////////////////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////
   
   def trainXgb(trainExs: ArrayBuffer[tmpExample], 
                featIndexer: Indexer[String],
                testExs: ArrayBuffer[tmpExample]) : Array[Double] = {
   
     val trainMax = examplesToMatrix(trainExs, true);
     val testMax = examplesToMatrix(testExs, true);
     
   
    //// train
    println("Trainset size: " + trainMax.rowNum);
	  println("Testset size: " + testMax.rowNum);
    
    val params = new HashMap[String, Any]()
    val round = 500
    //params += "distribution" -> "bernoulli"
    params += "eta" -> 0.1
    params += "max_depth" -> 20
    params += "silent" -> 0
    //params += "colsample_bytree" -> 0.9
    //params += "min_child_weight" -> 10
    params += "objective" -> "rank:pairwise"
    params += "eval_metric" -> "pre@1"
    params += "nthread" -> 4
      
    val watches = new HashMap[String, DMatrix]()
    watches += "train" -> trainMax
    watches += "test" -> testMax


    // train a model
    val booster = XGBoost.train(trainMax, params.toMap, round, watches.toMap)

    // predict
    val predicts = booster.predict(testMax)
    // save model to model path
    val file = new File("./model")
    if (!file.exists()) {
      file.mkdirs()
    }
    booster.saveModel(file.getAbsolutePath + "/xgb-ner.model")
    // dump model
    //booster.getModelDump(file.getAbsolutePath + "/dump.raw.txt", true)
    // dump model with feature map
    //booster.getModelDump(file.getAbsolutePath + "/featmap.txt", true)
    // save dmatrix into binary buffer
    trainMax.saveBinary(file.getAbsolutePath + "/dtrain.buffer")
    testMax.saveBinary(file.getAbsolutePath + "/dtest.buffer")
    
    ///////// test
    
    // reload model and data
    val booster2 = XGBoost.loadModel(file.getAbsolutePath + "/xgb-ner.model")
    val testMax2 = new DMatrix(file.getAbsolutePath + "/dtest.buffer")
    val predicts2 = booster2.predict(testMax2);
     
     val emptyWgt = new Array[Double](featIndexer.size());
     emptyWgt;
   }
   
   def examplesToMatrix(exs: ArrayBuffer[tmpExample], verbose: Boolean): DMatrix = {
     
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
    
    for (ex <- exs) {
      gpCnt += 1;
      
      if (gpCnt % 10000 == 0) {
        println("Feature " + gpCnt + " instances.");
      }
      
      val goldTag = getGoldNerTag(ex.exmp.goldLabel);
      val goldLabelIdx = MCNerFeaturizer.StdLabelIndexer.indexOf(goldTag);

      for (j <- 0 until MCNerFeaturizer.StdLabelIndexer.size) {
        rowCnt += 1;
        // for jth value
        val featIdxs = ex.feats(j).sortWith{_ < _};
        val lbl = if (goldLabelIdx == j) 1.0f else 0.0f
        // dump features
        var thisMaxIdx = -1;
        for (i <- 0 until featIdxs.length) {
    			tdata += (1.0f);
    			val fidx = featIdxs(i) + 1
    			tindex += (fidx);
    			if (fidx > maxFeatIdx) {
    			  maxFeatIdx = fidx;
    			}
    			if (fidx > thisMaxIdx) {
    			  thisMaxIdx = fidx;
    			}
    		}
        
        var totalFeats = featIdxs.length.toLong;
    		rowheader += totalFeats;
    		theaders += (rowheader);
    		tlabels += (lbl);
    	}
      tgroup += (MCNerFeaturizer.StdLabelIndexer.size);
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

    (mx);
   }
   
   def testXgb() {
     
   }
   
}
