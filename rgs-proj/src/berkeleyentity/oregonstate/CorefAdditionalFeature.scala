package berkeleyentity.oregonstate

import berkeleyentity.coref.DocumentInferencerBasic
import berkeleyentity.sem.QueryCountsBundle
import berkeleyentity.coref.NumberGenderComputer
import berkeleyentity.sem.BasicWordNetSemClasser
import berkeleyentity.coref.FeatureSetSpecification
import berkeleyentity.coref.PairwiseScorer
import berkeleyentity.coref.CorefSystem
import berkeleyentity.coref.LexicalCountsBundle
import berkeleyentity.coref.CorefFeaturizerTrainer
import berkeleyentity.coref.PairwiseLossFunctions
import berkeleyentity.coref.PairwiseIndexingFeaturizerJoint
import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.futile.util.Logger
import berkeleyentity.Driver
import berkeleyentity.coref.CorefDoc
import berkeleyentity.coref.DocumentGraph
import berkeleyentity.coref.MentionPropertyComputer
import java.io.PrintWriter
import berkeleyentity.coref.CorefEvaluator
import berkeleyentity.coref.OrderedClustering
import berkeleyentity.ConllDocReader
import berkeleyentity.coref.CorefPruner
import berkeleyentity.coref.CorefDocAssembler
import scala.collection.mutable.ArrayBuffer
import berkeleyentity.coref.PairwiseIndexingFeaturizer
import berkeleyentity.lang.Language
import ml.dmlc.xgboost4j.scala.Booster
import berkeleyentity.ilp.SingleDecision
import berkeleyentity.xgb.XgbMatrixBuilder
import scala.util.control.Breaks._

class CorefAdditionalFeature(val featurizer: PairwiseIndexingFeaturizer, val weights: Array[Double]) {

  def runPruningDocs(docGraphs: Seq[DocumentGraph]) {
    for (dg <- docGraphs) {
      runPruningSingle(dg);
    }
  }
  
  def runPruningSingle(docGraph: DocumentGraph) {
    val featsChart = docGraph.featurizeIndexNonPrunedUseCache(featurizer);
    val pedgs = docGraph.prunedEdges;
    for (i <- 0 until pedgs.size) {
      val okDomain = new ArrayBuffer[CorefDecision]();
      for (j <- 0 to i) {
        if (pedgs(i)(j)) {
          // has been pruned
        } else {
          // not been pruned
          //pedgs(i)(j) = false;
          //cachedFeats(i)(j) = emptyIntArray;
          val feat = featsChart(i)(j);
          val score = CorefAdditionalFeature.computeScore(weights, feat);
          val crr = docGraph.isGoldNoPruning(i, j);
          okDomain += new CorefDecision(i, j, score, crr);
        }
      }
    }
  }


}

object CorefAdditionalFeature {

  def main(args: Array[String]) {
    
    // set some configs
	  Driver.numberGenderDataPath = "../coref/berkfiles/data/gender.data";
	  Driver.brownPath = "../coref/berkfiles/data/bllip-clusters";
	  Driver.useGoldMentions = true;
	  Driver.doConllPostprocessing = false;
	  //Driver.pruningStrategy = "build:../coref/berkfiles/corefpruner-ace.ser.gz:-5:5";
    Driver.lossFcn = "customLoss-1-1-1";

	  Driver.corefNerFeatures = "indicators+currlex+antlex";
	  Driver.wikiNerFeatures = "categories+infoboxes+appositives";
	  Driver.corefWikiFeatures = "basic+lastnames";
    
    val trainDataPath = "../coref/berkfiles/data/ace05/train";
    val devDataPath = "../coref/berkfiles/data/ace05/dev";
    val testDataPath = "../coref/berkfiles/data/ace05/test";
    
    //val trainDataPath = "data/ace05/train_1";
    //val testDataPath = "data/ace05/test_1";

    //val wikiPath = "../coref/berkfiles/data/ace05/ace05-all-conll-wiki"
    //val wikiDBPath = "../coref/berkfiles/models/wiki-db-ace.ser.gz"
    
    val berkeleyCorefPrunerDumpPath = "models:../coref/berkfiles/corefpruner-ace.ser.gz:-5";

    val featIndexer = new Indexer[String]();
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    val assembler = CorefDocAssembler(Language.ENGLISH, true); //use gold mentions
    val trainDocs = ConllDocReader.loadRawConllDocsWithSuffix(trainDataPath, -1, "", Language.ENGLISH);
    val trainCorefDocs = trainDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));

    // load pruner
    val berkPruner = CorefPruner.buildPrunerArguments(berkeleyCorefPrunerDumpPath, trainDataPath, -1);
  
    val trainDocGraphs = trainCorefDocs.map(new DocumentGraph(_, true));

    // run berkeley pruning
    CorefStructUtils.preprocessDocsCacheResources(trainDocGraphs);
    //berkPruner.pruneAll(trainDocGraphs);

    
    // test examples
    val testCorefDocs = CorefStructUtils.loadCorefDocs(testDataPath, -1, Driver.docSuffix, mentionPropertyComputer);
    val testDocGraphs = SingleTaskStructTesting.testGetdocgraphs(testDataPath, -1, mentionPropertyComputer);
    //berkPruner.pruneAll(testDocGraphs);
    
    CorefAdditionalFeature.testPruningCoverage(trainDocGraphs);
    CorefAdditionalFeature.testPruningCoverage(testDocGraphs);
    
    
    trainAddPruner(trainCorefDocs, testCorefDocs, berkPruner);
    //trainAddPruner(trainCorefDocs, testCorefDocs, null);
  }
  
  def testAddPruner() {
    
  }
  
  def testPruningCoverage(docGraphs: Seq[DocumentGraph]) {
    
    var total = 0;
    var cntntCrr = 0;
    
	  for (docGraph <- docGraphs) {
		  val pedgs = docGraph.prunedEdges;
		  for (i <- 0 until pedgs.size) {
			  val okDomain = new ArrayBuffer[CorefDecision]();
			  val hasCrr = new ArrayBuffer[CorefDecision]();
			  for (j <- 0 to i) {
				  if (pedgs(i)(j)) {
					  // has been pruned
				  } else {
					  // not been pruned
					  val crr = docGraph.isGoldNoPruning(i, j);
					  val cdec = new CorefDecision(i, j, 0, crr);
					  okDomain += cdec;
					  if (crr) {
					    hasCrr += cdec;
					  }
				  }
			  }

			  total += 1;
			  if (hasCrr.size > 0) cntntCrr += 1;

		  }
	  }
	  
	  val rate = cntntCrr.toDouble / total.toDouble;
	  println("Coref Pruner Coverage: " + cntntCrr + " / " + total + " = " + rate);
  }

  def testMentionTypeAcc(docGraphs: Seq[DocumentGraph], mtpredictor: MentionPropertyComputer) {
    
    var total = 0;
    var cntntCrr = 0;
    
	  for (docGraph <- docGraphs) {
	    val corefDoc = docGraph.corefDoc;
	    for (m <- corefDoc.predMentions) {
	      val ptp = mtpredictor.predictAceMentionType(m);
	      if (ptp == m.cachedMentionTypeGold) {
	        cntntCrr += 1;
	      }
	      total += 1;
	    }
	  }
	  
	  val rate = cntntCrr.toDouble / total.toDouble;
	  println("Mention Type Accuracy: " + cntntCrr + " / " + total + " = " + rate);
  }
  
  def testCorefOnDocs(docGraphs: Seq[DocumentGraph],
                      myModel: CorefTesting) {
    
    Logger.startTrack("My Coref Testing:");
    
    val corefFeatizer = new CorefFeaturizerTrainer();
    corefFeatizer.featurizeBasic(docGraphs, myModel.featurizer);

    val allPredBackptrs = new Array[Array[Int]](docGraphs.size);
    for (i <- 0 until docGraphs.size) {
      val docGraph = docGraphs(i);
      Logger.logs("Decoding " + i);
      val predBackptrs = myModel.inferenceDecisionByDecision(docGraph);
      //val predBackptrs = myModel.inferenceDecisionByDecisionAndDumpErrFeatrues(docGraph, dumper, (i + 1));
      allPredBackptrs(i) = predBackptrs;
    }

    ////
    val allPredClusteringsSeq = (0 until docGraphs.size).map(i => OrderedClustering.createFromBackpointers(allPredBackptrs(i)));
    val allPredClusterings = allPredClusteringsSeq.toArray;
    
    
    //val (allPredBackptrs, allPredClusterings) = testDocGraphs(docGraphs, myModel, dumper);  
    val scoreOutput = CorefEvaluator.evaluateAndRender(docGraphs, allPredBackptrs, allPredClusterings, Driver.conllEvalScriptPath, "DEV: ", Driver.analysesToPrint);
    Logger.logss(scoreOutput);
    Logger.endTrack();
  }
  
  def testCorefBooster(docGraphs: Seq[DocumentGraph], featurizer: PairwiseIndexingFeaturizer, booster: Booster) {
    
    Logger.startTrack("My Coref Testing:");
    
    val allPredBackptrs = new Array[Array[Int]](docGraphs.size);
    for (i <- 0 until docGraphs.size) {
      val docGraph = docGraphs(i);
      //Logger.logs("Decoding " + i);
      val predBackptrs = XgbInterface.predictOneDocBooster(featurizer, docGraph, booster);// myModel.inferenceDecisionByDecision(docGraph);
      allPredBackptrs(i) = predBackptrs;
    }

    ////
    val allPredClusteringsSeq = (0 until docGraphs.size).map(i => OrderedClustering.createFromBackpointers(allPredBackptrs(i)));
    val allPredClusterings = allPredClusteringsSeq.toArray;
    
    //val (allPredBackptrs, allPredClusterings) = testDocGraphs(docGraphs, myModel, dumper);  
    val scoreOutput = CorefEvaluator.evaluateAndRender(docGraphs, allPredBackptrs, allPredClusterings, Driver.conllEvalScriptPath, "DEV: ", Driver.analysesToPrint);
    Logger.logss(scoreOutput);
    Logger.endTrack();
  }

  ///////////////////////////////////////////////////////////////////////////////////
  
   
  def trainAddPruner(foldTrainingDocs: Seq[CorefDoc], foldTestDocs: Seq[CorefDoc], corefPruner: CorefPruner) {
    
    val ifPrune = (corefPruner != null);
    
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val queryCounts: Option[QueryCountsBundle] = None;

    val foldTrainDocGraphs = foldTrainingDocs.map(new DocumentGraph(_, true));
    CorefSystem.preprocessDocsCacheResources(foldTrainDocGraphs);
    if (ifPrune) {
     corefPruner.pruneAll(foldTrainDocGraphs);
    }
    
      
    val lexicalCounts = LexicalCountsBundle.countLexicalItems(foldTrainingDocs, Driver.lexicalFeatCutoff);
    val featureIndexer = new Indexer[String]();
    featureIndexer.getIndex(PairwiseIndexingFeaturizerJoint.UnkFeatName);
    val featureSetSpec = FeatureSetSpecification(Driver.pairwiseFeats, Driver.conjScheme, Driver.conjFeats, Driver.conjMentionTypes, Driver.conjTemplates);
    val basicFeaturizer = new PairwiseIndexingFeaturizerJoint(featureIndexer, featureSetSpec, lexicalCounts, queryCounts, Some(new BasicWordNetSemClasser));
    val featurizerTrainer = new CorefFeaturizerTrainer();
    featurizerTrainer.featurizeBasic(foldTrainDocGraphs, basicFeaturizer);
    
    //////////////////////////////////////////
    
    val decisionTrainExmps = extractDecisionExamples(foldTrainDocGraphs);
    val devDocGraphs = CorefTesting.prepareTestDocsGivenCorefDocs(foldTestDocs, corefPruner, ifPrune);//CorefTesting.prepareTestDocuments(testPath, -1, corefpruner, false);
    featurizerTrainer.featurizeBasic(devDocGraphs, basicFeaturizer);
    val decisionTestExmps = extractDecisionExamples(devDocGraphs);
    
    CorefAdditionalFeature.testPruningCoverage(foldTrainDocGraphs);
    CorefAdditionalFeature.testPruningCoverage(devDocGraphs);
    
    val gbxm = new XgbInterface();
    val bstr = gbxm.runQuickLearning(decisionTrainExmps, decisionTestExmps);
    
    testCorefBooster(devDocGraphs, basicFeaturizer, bstr);
    printCorefHistgram(devDocGraphs, basicFeaturizer, bstr); // print histgram
    
    println("Done.");
  }

  
  def extractDecisionExamples(docGraphs: Seq[DocumentGraph]): ArrayBuffer[CorefDecisionExample] = {
    val result = new ArrayBuffer[CorefDecisionExample]();
    for (i <- 0 until docGraphs.size) {
      result ++= extractDeciExmlOneDoc(docGraphs(i));
    }
    result;
  }
  
  def extractDeciExmlOneDoc(docGraph: DocumentGraph): ArrayBuffer[CorefDecisionExample] = {
   
    val exmpleArr = new ArrayBuffer[CorefDecisionExample]();
    
    for (i <- 0 until docGraph.size) {
      val valArr = new ArrayBuffer[Int]();
      val featArr = new ArrayBuffer[Array[Int]]();
      val goldArr = new ArrayBuffer[Int]();

      for (j <- 0 to i) {
        if (!docGraph.prunedEdges(i)(j)) {
          //require(featsChart(i)(j).size > 0);
          //featurizer.featurizeIndex(docGraph, i, j, false);
          //scoreChart(i)(j) = GUtil.scoreIndexedFeats(featsChart(i)(j), scorer.weights);
          valArr += j;
          featArr += (docGraph.cachedFeats(i)(j));
          if (docGraph.isGoldNoPruning(i, j)) {
            goldArr += j
          }
        } else {
          // was pruned
          //  scoreChart(i)(j) = Float.NegativeInfinity;
        }
      }
      
      exmpleArr += (new CorefDecisionExample(i, valArr.toArray, goldArr.toArray, featArr.toArray));
    }
    
    exmpleArr;
  }

  
  
  def printCorefHistgram(docGraphs: Seq[DocumentGraph], featurizer: PairwiseIndexingFeaturizer, booster: Booster) {
    
    val corefCrrctRank = Array.fill[Int](10000)(0);//new Array[Int](10000);
    
    
    ///////////////////////////////////////////
    for (i <- 0 until docGraphs.size) {
      val docGraph = docGraphs(i);
      
      val featsChart = docGraph.featurizeIndexNonPrunedUseCache(featurizer);
      val decisions = CorefTesting.extractDeciExmlOneDoc(docGraph);
      val result = new Array[Int](decisions.length);
      for (j <- 0 until decisions.length) {
    	  val singleDecs = scoreDecisionValues(decisions(j), booster);
    	  val crrRank = getFirstCorrectRank(singleDecs, 9999);
    	  corefCrrctRank(crrRank) += 1;
      }
      
    }
    
    for (i <- 0 until 50) {
    	println("Coref["+i+"]: " + corefCrrctRank(i));
    }
  }
  
  
  
  def scoreDecisionValues(decision: CorefDecisionExample, booster: Booster): Array[SingleDecision] = {
    
    val (oneInstanMtrx, wdt) = XgbMatrixBuilder.corefExmpToDMatrix(Seq(decision), -1, false);
    val predicts2 = booster.predict(oneInstanMtrx);
    
    val singleDecs = new ArrayBuffer[SingleDecision](); 
		for (j <- 0 until decision.values.length) {
			val l = decision.values(j);
			val crr = decision.isCorrect(l);
			val score = predicts2(j)(0);
			singleDecs += (new SingleDecision(score, crr, j));
		}
		
		val sortc = (singleDecs.toSeq.sortWith(_.score > _.score)).toArray;
		sortc;
  }
  
  def getFirstCorrectRank(decisions: Array[SingleDecision], defaultNoCrrctRank: Int) = {
	  var result = defaultNoCrrctRank;
	  breakable {
		  for (rank <- 0 until decisions.length) {
			  if (decisions(rank).isCorrect) {
				  result = rank;
				  break;
			  }
		  }
	  }
	  result;
  }
  
  
  
  
  
  
  
  
  
  
  //// About Learning ...
  
  def structurePerceptrion(allTrains: ArrayBuffer[CorefDecisionExample],
		                       featIndexer: Indexer[String],
		                       testExs: ArrayBuffer[CorefDecisionExample]) : Array[Double] = {

		  //val logger = new PrintWriter("wiki_ace05_train.txt");
		  //val logger2 = new PrintWriter("wiki_ace05_test.txt");

		  var weight = Array.fill[Double](featIndexer.size)(0);//new Array[Double](featIndexer.size());
		  var weightSum = Array.fill[Double](featIndexer.size)(0);
		  var lastWeight = Array.fill[Double](featIndexer.size)(0);

		  val Iteration = 10;
		  val learnRate = 0.1;
		  val lambda = 1e-8;

		  var updateCnt = 0;
		  var lastUpdtCnt = 0;

		  /*
      var exId2 = 0;
      for (extst <- testExs) {
        exId2 += 1;
        val domains = constructWikiExampleDomains(extst, wikiDB);
        for (l <- 0 until domains.size) {
          logger2.println(domains(l).getFeatStr(exId2));
        }
      }*/

		  for (iter <- 0 until Iteration) {
			  lastUpdtCnt = updateCnt;
			  Array.copy(weight, 0, lastWeight, 0, weight.length);

			  println("Iter " + iter);
			  var exId = 0;
			  for (example <- allTrains) {

				  exId += 1;
				  val domains = example.values;
				  val prunOk = (example.goldValues.length != 0);
				  if (prunOk) {

					  var bestLbl = -1;
					  var bestScore = -Double.MaxValue;
					  var bestCorrectLbl = -1; // latent best
					  var bestCorrectScore = -Double.MaxValue;
					  
					  val allCrrtFeats = new ArrayBuffer[Array[Int]]();
					  for (j <- 0 until domains.size) {
						  val l = domains(j);
						  //println(domains(l).getFeatStr(exId));
						  //logger.println(domains(l).getFeatStr(exId));
						  var score = computeScore(weight, example.features(j));
						  if (score > bestScore) {
							  bestScore = score;
							  bestLbl = j;
						  }
						  if (example.isCorrect(l)) {
							  if (score > bestCorrectScore) {
								  bestCorrectScore = score;
								  bestCorrectLbl = j;
							  }
							  allCrrtFeats += example.features(j);
						  }
					  }

					  //println("size = " + domains.size + " pred = " + bestLbl + " correct = " + bestCorrectLbl)

					  // update?
					  if (!example.isCorrect(domains(bestLbl))) {
						  updateCnt += 1;
						  if (updateCnt % 1000 == 0) println("Update " + updateCnt);
						  updateWeight(weight, 
							         	   example.features(bestCorrectLbl),
								           example.features(bestLbl),
								           learnRate,
								           lambda);
						  /*updateWeightAverageLatent(weight, 
							         	   allCrrtFeats.toArray,//example.features(bestCorrectLbl),
								           example.features(bestLbl),
								           learnRate,
								           lambda);*/
						  sumWeight(weightSum, weight);
					  }
				  }
			  }

			  ///////////////////////////////////////////////////
			  // have a test after each iteration (for learning curve)
			  val tmpAvg = new Array[Double](weightSum.size)
					  Array.copy(weightSum, 0, tmpAvg, 0, weightSum.size);
			  divdeNumber(tmpAvg, updateCnt.toDouble);

			  //quickTest(allTrains, tmpAvg, wikiDB);
			  quickTest(testExs, tmpAvg);
			  println("Iter Update Cnt = " + (updateCnt - lastUpdtCnt));

			  val wdiff = checkWeight(weight, lastWeight);
			  println("Weight diff = " + wdiff);
		  }

		  divdeNumber(weightSum, updateCnt.toDouble);

      /*
		  for (i <- 0 until weightSum.length) {
			  if (weightSum(i) != 0) {
				  println("weight(" + i + ") = " + weightSum(i));
			  }
		  }
      */

		  //logger.close();
		  //logger2.close();
		  weightSum;
  }
  
  def checkWeight(curWeight: Array[Double],
                  oldWeight: Array[Double]): Int = {
    var different: Int = 0;
    for (i <- 0 until curWeight.length) {
      if (curWeight(i) != oldWeight(i)) different += 1;
    }
    different;
  }
  

  def quickTest(testExs: ArrayBuffer[CorefDecisionExample],
                weight: Array[Double]) {
    
    var correct: Double = 0.0;
    
    for (ex  <- testExs) {
      val pred = ex.getBestValue(weight);
      if (ex.isCorrect(pred)) {
        correct += 1.0;
      }
    }
    
    var total = testExs.size.toDouble;
    var acc = correct / total;
    println("Acc: " + correct + "/" + total + " = " + acc);
  }

  

  def updateWeight(currentWeight: Array[Double], 
                   featGold: Array[Int],
                   featPred: Array[Int],
                   eta: Double,
                   lambda: Double) {
    var gradient = Array.fill[Double](currentWeight.length)(0);//new Array[Double](currentWeight.length);
    
    
    val possibleNonZeroIdx = new ArrayBuffer[Int]();
    
    for (i <- featGold) {
      if (i >= 0) {
        gradient(i) += (1.0);
        possibleNonZeroIdx += (i);
      }
    }
    for (j <- featPred) {
      if (j >= 0) {
        gradient(j) -= (1.0);
        possibleNonZeroIdx += (j);
      }
    }
    /*
    var cnt = 0;
    for (nzidx <- possibleNonZeroIdx) {
      if (gradient(nzidx) != 0) cnt += 1;
    }*/
    
    //println("gradient non-zero element cnt: " + cnt);
    
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
  
  def updateWeightAverageLatent(currentWeight: Array[Double], 
                   featGolds: Array[Array[Int]],
                   featPred: Array[Int],
                   eta: Double,
                   lambda: Double) {
    var gradient = Array.fill[Double](currentWeight.length)(0);//new Array[Double](currentWeight.length);
    
    
    val possibleNonZeroIdx = new ArrayBuffer[Int]();
    for (featGold <- featGolds) {
    	for (i <- featGold) {
    		if (i >= 0) {
    			gradient(i) += (1.0);
    			possibleNonZeroIdx += (i);
    		}
    	}
    }
    // do average
    val totalCnt = featGolds.length.toDouble;
    if (totalCnt <= 0) {
    	throw new RuntimeException("no gold feature!");
    }
    for (k <- possibleNonZeroIdx) {
    	//gradient(k) = gradient(k) / totalCnt;
    }

    for (j <- featPred) {
    	if (j >= 0) {
    		gradient(j) -= (1.0);
    		// possibleNonZeroIdx += (j);
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
  
}