package berkeleyentity.oregonstate

import java.util.{HashMap => JHashMap}
import java.io.PrintWriter
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashSet
import scala.util.Sorting._
import berkeleyentity.ConllDocReader
import berkeleyentity.Driver
import berkeleyentity.WordNetInterfacer
import berkeleyentity.coref.CorefDoc
import berkeleyentity.coref.CorefDocAssembler
import berkeleyentity.coref.CorefEvaluator
import berkeleyentity.coref.CorefFeaturizerTrainer
import berkeleyentity.coref.CorefPruner
import berkeleyentity.coref.DocumentGraph
import berkeleyentity.coref.FeatureSetSpecification
import berkeleyentity.coref.LexicalCountsBundle
import berkeleyentity.coref.MentionPropertyComputer
import berkeleyentity.coref.NumberGenderComputer
import berkeleyentity.coref.OrderedClustering
import berkeleyentity.coref.PairwiseIndexingFeaturizer
import berkeleyentity.coref.PairwiseIndexingFeaturizerJoint
import berkeleyentity.sem.BasicWordNetSemClasser
import berkeleyentity.sem.QueryCountsBundle
import berkeleyentity.sem.SemClasser
import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.futile.util.Logger
import ciir.umass.edu.learning.SparseDataPoint
import berkeleyentity.ranking.UMassRankLib
import ciir.umass.edu.learning.DataPoint
import berkeleyentity.ner.MCNerFeaturizer
import edu.illinois.cs.cogcomp.sl.core.SLParameters
import edu.illinois.cs.cogcomp.sl.util.WeightVector
import edu.illinois.cs.cogcomp.sl.core.SLModel
import edu.illinois.cs.cogcomp.sl.core.IStructure
import edu.illinois.cs.cogcomp.sl.util.FeatureVectorBuffer
import edu.illinois.cs.cogcomp.sl.util.IFeatureVector
import edu.illinois.cs.cogcomp.sl.core.AbstractFeatureGenerator
import edu.illinois.cs.cogcomp.sl.core.IInstance
import edu.illinois.cs.cogcomp.sl.learner.LearnerFactory
import edu.illinois.cs.cogcomp.sl.core.AbstractInferenceSolver
import edu.illinois.cs.cogcomp.sl.latentsvm.AbstractLatentInferenceSolver
import edu.illinois.cs.cogcomp.sl.latentsvm.LatentLearner
import edu.illinois.cs.cogcomp.sl.core.SLProblem
import scala.collection.mutable.HashMap
import berkeleyentity.coref.MentionPropertyComputer
import berkeleyentity.sem.QueryCountsBundle
import berkeleyentity.coref.NumberGenderComputer
import berkeleyentity.sem.BasicWordNetSemClasser
import berkeleyentity.coref.FeatureSetSpecification
import berkeleyentity.coref.DocumentGraph
import berkeleyentity.coref.CorefDoc
import berkeleyentity.coref.OrderedClustering
import berkeleyentity.ConllDocReader
import berkeleyentity.coref.LexicalCountsBundle
import berkeleyentity.WordNetInterfacer
import berkeleyentity.coref.CorefFeaturizerTrainer
import berkeleyentity.coref.PairwiseIndexingFeaturizerJoint

class CorefDecision(val index: Int,
                    val antecedent: Int,
                    val rankScore: Double,
                    val isCrr: Boolean) {
  
}

  
class CorefOutput(val output: Int) extends IStructure {

}

class CorefTesting(val featurizer: PairwiseIndexingFeaturizer,
                   val weights: Array[Double]) {
  
  //var errFinderModelPath = "";
  
  def inference(docGraph: DocumentGraph) : Array[Int] = {
    
    val featsChart = docGraph.featurizeIndexNonPrunedUseCache(featurizer);
    for (i <- 0 until docGraph.size) {
      for (j <- 0 to i) {
        //if (!prunedEdges(i)(j)) {
          //require(featsChart(i)(j).size > 0);
          featurizer.featurizeIndex(docGraph, i, j, false);
          
          //scoreChart(i)(j) = GUtil.scoreIndexedFeats(featsChart(i)(j), scorer.weights);
        //} else {
        //  scoreChart(i)(j) = Float.NegativeInfinity;
        //}
      }
    }
   // (featsChart, scoreChart)
    
    
    ???
  }
  
  // treat each mention decision as independent example, and predict
  def inferenceDecisionByDecision(docGraph: DocumentGraph) : Array[Int] = {
    
    val featsChart = docGraph.featurizeIndexNonPrunedUseCache(featurizer);
    val decisions = CorefTesting.extractDeciExmlOneDoc(docGraph);
    
    val result = new Array[Int](decisions.length);
    for (i <- 0 until decisions.length) {
      result(i) = decisions(i).getBestValue(weights);
      val crr = decisions(i).isCorrect(result(i));
      val bestFeat = decisions(i).features(result(i));
    }
    
    result;
  }

  def numWeights(): Int = {
    weights.size
  }
}


class CorefDecisionExample(val index: Int,
                           val values: Array[Int], 
                           val goldValues: Array[Int],
                           val features: Array[Array[Int]]) extends IInstance {
  
  val correctSet = ((new HashSet[Int]()) ++ (goldValues));
  val valueToIndex = new HashMap[Int, Int]();
  for (i <- 0 until values.length) {
    valueToIndex += (values(i) -> i);
  }
  
  // return the first correct index
  def getGold(): Int = {
    if (goldValues.length <= 0) {
      throw new RuntimeException("no gold for this example!");
    }
    goldValues(0);
  }
  
  // return the value that has the highest score
  def getBestValue(wght: Array[Double]): Int = {
		  var bestLbl = -1;
		  var bestScore = -Double.MaxValue;
		  for (j <- 0 until values.size) {
			  val l = values(j);
			  var score = CorefTesting.computeScore(wght, features(j));
			  if (score > bestScore) {
				  bestScore = score;
				  bestLbl = j;
			  }
		  }
		  values(bestLbl);
  }
  
  def getOracleBestValue(wght: Array[Double]): Int = {
		  var bestLbl = -1;
		  var bestScore = -Double.MaxValue;
		  var bestGoldLbl = -1;
		  var bestGoldScore = -Double.MaxValue;
		  for (j <- 0 until values.size) {
			  val l = values(j);
			  var score = CorefTesting.computeScore(wght, features(j));
			  if (score > bestScore) {
				  bestScore = score;
				  bestLbl = l;
			  }
			  if (correctSet.contains(l)) {
				  if (score > bestGoldScore) {
					  bestGoldScore = score;
					  bestGoldLbl = l;
				  }
			  }
		  }
      if (bestGoldLbl < 0) {
        bestLbl;
      }
      bestGoldLbl;
  }
  
  def isCorrect(value: Int): Boolean = {
    correctSet.contains(value);
  }
  
  def valueIndex(value: Int): Int = {
    if (valueToIndex.contains(value)) {
      valueToIndex(value);
    } else {
      throw new RuntimeException("value " + value + " has no index!");
      -1;
    }
  }
  
}


object CorefTesting {
  
  def main(args: Array[String]) {
    val trainDataPath = "data/ace05/train";
    val devDataPath = "data/ace05/dev";
    val testDataPath = "data/ace05/test";
    
    // set some configs
    Driver.useGoldMentions = true;
    Driver.doConllPostprocessing = false;
    
    TrainBerkCorefACE(trainDataPath, testDataPath);
  }
  
  def loadCorefDocs(path: String, size: Int, suffix: String, maybeNumberGenderComputer: Option[NumberGenderComputer]): Seq[CorefDoc] = {
    val docs = ConllDocReader.loadRawConllDocsWithSuffix(path, size, suffix);
    val assembler = CorefDocAssembler(Driver.lang, Driver.useGoldMentions);
    val mentionPropertyComputer = new MentionPropertyComputer(maybeNumberGenderComputer);
    val corefDocs = if (Driver.useCoordination) {
      docs.map(doc => assembler.createCorefDocWithCoordination(doc, mentionPropertyComputer));
    } else {
      docs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    }
    CorefDocAssembler.checkGoldMentionRecall(corefDocs);
    corefDocs;
  }
  
    
  def preprocessDocsCacheResources(allDocGraphs: Seq[DocumentGraph]) {
    if (Driver.wordNetPath != "") {
      val wni = new WordNetInterfacer(Driver.wordNetPath);
      allDocGraphs.foreach(_.cacheWordNetInterfacer(wni));
    }
  }
  

  
  def TrainBerkCorefACE(trainPath: String, testPath: String) {
    
    val trainSize = -1;
    
	  val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
	  val queryCounts: Option[QueryCountsBundle] = None;
	  val trainDocs = loadCorefDocs(trainPath, trainSize, Driver.docSuffix, Some(numberGenderComputer));
	  // Randomize
	  val trainDocsReordered = new scala.util.Random(0).shuffle(trainDocs);
	  val lexicalCounts = LexicalCountsBundle.countLexicalItems(trainDocs, Driver.lexicalFeatCutoff);
	  val semClasser: Option[SemClasser] = Driver.semClasserType match {
	  case "basic" => Some(new BasicWordNetSemClasser);
	  case e => throw new RuntimeException("Other semclassers not implemented");
	  }
	  val trainDocGraphs = trainDocsReordered.map(new DocumentGraph(_, true));
	  preprocessDocsCacheResources(trainDocGraphs);
	  val corefpruner = CorefPruner.buildPruner(Driver.pruningStrategy);
	  //corefpruner.pruneAll(trainDocGraphs);

	  val featureIndexer = new Indexer[String]();
	  featureIndexer.getIndex(PairwiseIndexingFeaturizerJoint.UnkFeatName);
	  val featureSetSpec = FeatureSetSpecification(Driver.pairwiseFeats, Driver.conjScheme, Driver.conjFeats, Driver.conjMentionTypes, Driver.conjTemplates);
	  val basicFeaturizer = new PairwiseIndexingFeaturizerJoint(featureIndexer, featureSetSpec, lexicalCounts, queryCounts, semClasser);
	  val featurizerTrainer = new CorefFeaturizerTrainer();
	  featurizerTrainer.featurizeBasic(trainDocGraphs, basicFeaturizer);
	  PairwiseIndexingFeaturizer.printFeatureTemplateCounts(featureIndexer)


    val decisionTrainExmps = extractDecisionExamples(trainDocGraphs);
    
    ////////////// About testing examples //////////////////////
    val devDocGraphs = prepareTestDocuments(testPath, -1, corefpruner, false);
    featurizerTrainer.featurizeBasic(devDocGraphs, basicFeaturizer);
    val decisionTestExmps = extractDecisionExamples(devDocGraphs);
    /////////////////////////////////////////////////////////
    
    val wght = CorefSVMLearning.corefLatentSVM(decisionTrainExmps, featureIndexer, decisionTestExmps);
    val mmodel = new CorefTesting(basicFeaturizer, wght);

    testCorefOnDocs(devDocGraphs, mmodel);

    
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
  
  
  //////////////////////////////////////////////////////////////////////////////////////////
  
  def prepareTestDocuments(devPath: String, devSize: Int, corefPruner: CorefPruner, doprune: Boolean): Seq[DocumentGraph] = {
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val devDocs = loadCorefDocs(devPath, devSize, Driver.docSuffix, Some(numberGenderComputer));
    val devDocGraphs = devDocs.map(new DocumentGraph(_, false));
    preprocessDocsCacheResources(devDocGraphs);
    //CorefPruner.buildPruner(Driver.pruningStrategy).pruneAll(devDocGraphs);
    if (doprune) {
      corefPruner.pruneAll(devDocGraphs);
    }
    devDocGraphs;
  }
  
  def prepareTestDocsGivenCorefDocs(devDocs: Seq[CorefDoc], corefPruner: CorefPruner, doprune: Boolean): Seq[DocumentGraph] = {
    val devDocGraphs = devDocs.map(new DocumentGraph(_, false));
    preprocessDocsCacheResources(devDocGraphs);
    if (doprune) {
      corefPruner.pruneAll(devDocGraphs);
    }
    devDocGraphs;
  }
  
  def testCorefOnDocs(devDocGraphs: Seq[DocumentGraph],
                      myModel: CorefTesting) {
    Logger.startTrack("My Coref Testing:");
    val (allPredBackptrs, allPredClusterings) = testDocGraphs(devDocGraphs, myModel);  
    val scoreOutput = CorefEvaluator.evaluateAndRender(devDocGraphs, allPredBackptrs, allPredClusterings, Driver.conllEvalScriptPath, "DEV: ", Driver.analysesToPrint);
    Logger.logss(scoreOutput);
    Logger.endTrack();
  }
  
  //def testDocGraphs(docGraphs: Seq[DocumentGraph], myModel: CorefTesting, dumper: PrintWriter): (Array[Array[Int]], Array[OrderedClustering]) = {
  def testDocGraphs(docGraphs: Seq[DocumentGraph], myModel: CorefTesting): (Array[Array[Int]], Array[OrderedClustering]) = {

    val corefFeatizer = new CorefFeaturizerTrainer();
    corefFeatizer.featurizeBasic(docGraphs, myModel.featurizer);
    
    // results
    var wngCnt = 0;
    var total = 0;
    var mTotal = 0;
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
    val allPredClusteringsArr = allPredClusteringsSeq.toArray;
    
    // return
    (allPredBackptrs, allPredClusteringsArr);
  }

}



object CorefSVMLearning {
  
  //////////////////////////////////////
  //////////////////////////////////////
  //////////////////////////////////////
  
  def constructProblem(exs: ArrayBuffer[CorefDecisionExample]): SLProblem = {
		val p = new SLProblem();
		for (i <- 0 until exs.size) {
			val x = exs(i);
			val y = new CorefOutput(exs(i).getGold());
			//IStructure gold = exs.get(i);
			// explaining
			// latent
			// structure
			p.instanceList.add(x);
			p.goldStructureList.add(y);
		}
		return p;
	}


  // test with SVM
   def corefLatentSVM(trainExs: ArrayBuffer[CorefDecisionExample], 
                      featIndexer: Indexer[String],
                      testExs: ArrayBuffer[CorefDecisionExample]) : Array[Double] = {
     
      val trainProbl = constructProblem(trainExs);
     
      val model = new SLModel();
      model.numFeatuerBit = featIndexer.size();
      
      val featGener = new CorefSlFeatureGenerator();


      val configFilePath = "config/uiuc-sl-config/myDCD-coref-struct.config";
      val para = new SLParameters();
      para.loadConfigFile(configFilePath);
      model.featureGenerator = featGener;
      
            // initialize the inference solver
      val latendInferencer = new CorefLatentInferenceSolver(featGener);
      model.infSolver = latendInferencer;

      val learner = LearnerFactory.getLearner(model.infSolver, model.featureGenerator, para);
      
      val latentPara = new SLParameters();
      //latentPara.loadConfigFile(configFilePath);
      latentPara.MAX_NUM_ITER = 15;
      val latentLearner = new LatentLearner(learner,  model.featureGenerator, latentPara, latendInferencer);
      
      model.wv = latentLearner.train(trainProbl, new WeightVector(featIndexer.size));
      //model.wv = new WeightVector(featIndexer.size + 1);
      model.config = new java.util.HashMap[String, String]();
      
      // save the model
      val modelPath = "indep_coref_ace05.txt";
      model.saveModel(modelPath);
      
      ///////////////////////////////////////////////////
      testSvmCoref(model, trainExs);
      testSvmCoref(model, testExs);

      getDoubleWeight(model);
    }
   
   def testSvmCoref(model: SLModel, testExs: ArrayBuffer[CorefDecisionExample]) {
     val latendInferencer = model.infSolver.asInstanceOf[CorefLatentInferenceSolver];
     var pred_loss : Double = 0.0;
      for (i <- 0 until testExs.size) {
        val ri = testExs(i).asInstanceOf[IInstance];
        //val pred = latendInferencer.getBestLatentStructure(model.wv, ri, null).asInstanceOf[CorefOutput];
        val pred = latendInferencer.getBestStructure(model.wv, ri).asInstanceOf[CorefOutput];
        if (testExs(i).isCorrect(pred.output)) {
          pred_loss += 1.0;
        }
      }
      println("Acc = " + (pred_loss / testExs.size.toDouble));
   }

/*
   def getDoubleWeight(modl: SLModel): Array[Double] = {
		   val wv = modl.wv;
		   val farr = wv.getWeightArray; // (0) is bias
		   var darr = new Array[Double](farr.length - 1);
		   for (i <- 0 until darr.length) {
			   darr(i) = farr(i + 1).toDouble;
		   }
		   darr;
   }
*/
  def getDoubleWeight(modl: SLModel): Array[Double] = {
		  val wv = modl.wv;
		  val farr = wv.getWeightArray; // (0) is bias
		  var darr = new Array[Double](farr.length);
		  for (i <- 0 until darr.length) {
			  darr(i) = farr(i).toDouble;
		  }
		  darr;
  }
   
   class CorefSlFeatureGenerator() extends AbstractFeatureGenerator {

	   override def getFeatureVector(x: IInstance, y: IStructure): IFeatureVector = {
			   val mx = x.asInstanceOf[CorefDecisionExample];
			   val my = y.asInstanceOf[CorefOutput];
 			   
			   val vidx = mx.valueIndex(my.output);
			   val nonzeroIdxs = mx.features(vidx);
			   
			   val fb = new FeatureVectorBuffer();
			   for (idx <- nonzeroIdxs) {
				   //fb.addFeature(idx + 1, 1.0f);
			     fb.addFeature(idx, 1.0f);
			   }
			   fb.toFeatureVector();
	   }
	   
	   override def getFeatureVectorDiff(x: IInstance, y1: IStructure, y2: IStructure): IFeatureVector = {
			   val f1 = getFeatureVector(x, y1);
			   val f2 = getFeatureVector(x, y2);		
			   return f1.difference(f2);
	   }
     
   }
  
   class CorefLatentInferenceSolver(val featureGener: CorefSlFeatureGenerator) extends AbstractLatentInferenceSolver {

	   override def getLossAugmentedBestStructure(weight: WeightVector, ins: IInstance, gold: IStructure) : IStructure = {

			   val mi = ins.asInstanceOf[CorefDecisionExample];;
			   val lmi = gold.asInstanceOf[CorefOutput];

			   var bestOutput: Int = -1;
			   var bestScore: Float = -Float.MaxValue;

			   for (idx <- 0 until mi.values.length) {
			     val i = mi.values(idx);
           val currentl = new CorefOutput(i);
				   var score = weight.dotProduct(featureGener.getFeatureVector(mi, currentl));

				   // if this is wrong, add the loss
				   if (gold != null) {
					   if (!mi.isCorrect(i)) {
						   score += 1.0F;
					   }
				   }
				   
				   if (score > bestScore){
					   bestOutput = i;
					   bestScore = score;
				   }
			   }

			   (new CorefOutput(bestOutput));   
	   }

	   override def getBestLatentStructure(weight: WeightVector, ins: IInstance, gold: IStructure): IStructure = {
	     
			   val mi = ins.asInstanceOf[CorefDecisionExample];;
			   val lmi = gold.asInstanceOf[CorefOutput];

			   var bestOutput: Int = -1;
			   var bestScore: Float = -Float.MaxValue;

			   for (idx <- 0 until mi.goldValues.length) {
			     val i = mi.goldValues(idx);
           val currentl = new CorefOutput(i);
				   var score = weight.dotProduct(featureGener.getFeatureVector(mi, currentl));
				   if (score > bestScore){
					   bestOutput = i;
					   bestScore = score;
				   }
			   }
			   
			   if (bestOutput == -1) {
			     throw new RuntimeException("Latent inference error: -1!");
			   }

			   (new CorefOutput(bestOutput));
	   }
	   
	   override def getBestStructure(weight: WeightVector, ins: IInstance) = {
		   getLossAugmentedBestStructure(weight, ins, null);
	   }

	   override def getLoss(ins: IInstance, gold: IStructure, pred: IStructure) = {
		   var loss = 0.0f;
		   val mi = ins.asInstanceOf[CorefDecisionExample];
		   //val lmi = gold.asInstanceOf[CorefOutput];
		   val pmi = pred.asInstanceOf[CorefOutput];
		   //if (pmi.output != lmi.output){
			 //  loss = 1.0f;    
		   //}
		   if (!mi.isCorrect(pmi.output)) {
		     loss = 1.0f;
		   }
		   loss;
	   }
	   
	   override def clone() = {
		   val fg = new CorefSlFeatureGenerator();
		   val cpInferencer = new CorefLatentInferenceSolver(fg);
		   cpInferencer;
	   }

   }
   
}