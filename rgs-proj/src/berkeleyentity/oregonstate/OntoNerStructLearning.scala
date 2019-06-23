package berkeleyentity.oregonstate

import berkeleyentity.sem.BrownClusterInterface
import berkeleyentity.wiki.WikificationEvaluator
import berkeleyentity.coref.MentionPropertyComputer
import berkeleyentity.sem.QueryCountsBundle
import berkeleyentity.joint.JointDoc
import berkeleyentity.joint.GeneralTrainer
import berkeleyentity.joint.JointDocACE
import berkeleyentity.joint.JointComputerShared
import berkeleyentity.coref.NumberGenderComputer
import berkeleyentity.sem.BasicWordNetSemClasser
import berkeleyentity.coref.FeatureSetSpecification
import edu.berkeley.nlp.futile.fig.exec.Execution
import berkeleyentity.coref.CorefEvaluator
import berkeleyentity.ner.NerPrunerFromMarginals
import berkeleyentity.coref.DocumentGraph
import berkeleyentity.joint.FactorGraphFactoryACE
import berkeleyentity.wiki.WikipediaInterface
import berkeleyentity.joint.FactorGraphFactoryOnto
import berkeleyentity.joint.JointPredictor
import berkeleyentity.ner.MCNerFeaturizer
import berkeleyentity.coref.CorefDoc
import berkeleyentity.coref.CorefDocAssemblerACE
import berkeleyentity.lang.Language
import berkeleyentity.joint.JointFeaturizerShared
import berkeleyentity.sem.SemClasser
import berkeleyentity.wiki.ACEMunger
import berkeleyentity.wiki.DocWikiAnnots
import berkeleyentity.ner.NerFeaturizer
import berkeleyentity.coref.CorefPruner
import berkeleyentity.coref.CorefDocAssembler
import berkeleyentity.wiki.CorpusWikiAnnots
import berkeleyentity.coref.LexicalCountsBundle
import berkeleyentity.ner.NEEvaluator
import berkeleyentity.ner.NerDriver
import berkeleyentity.wiki.WikiAnnotReaderWriter
import berkeleyentity.coref.OrderedClusteringBound
import berkeleyentity.coref.PairwiseIndexingFeaturizerJoint
import berkeleyentity.ner.NerSystemLabeled
import berkeleyentity.coref.PairwiseIndexingFeaturizer
import edu.berkeley.nlp.futile.classify.SequenceExample
import edu.berkeley.nlp.futile.classify.GeneralLogisticRegression
import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.futile.fig.basic.IOUtils
import edu.berkeley.nlp.futile.util.Counter
import edu.berkeley.nlp.futile.util.Logger
import berkeleyentity.ner.NerPruner
import berkeleyentity.joint.JointLossFcns
import berkeleyentity.coref.PairwiseScorer
import berkeleyentity.coref.OrderedClustering
import berkeleyentity.coref.PairwiseLossFunctions
import berkeleyentity.coref.UID
import berkeleyentity.wiki._
import berkeleyentity.joint.JointPredictorACE
import berkeleyentity.coref.CorefSystem
import berkeleyentity.ConllDocReader
import berkeleyentity.Chunk
import berkeleyentity.GUtil
import berkeleyentity.ConllDoc

import edu.illinois.cs.cogcomp.sl.core._
import edu.illinois.cs.cogcomp.sl.learner._
import edu.illinois.cs.cogcomp.sl.util._
import edu.illinois.cs.cogcomp.sl.core.IStructure
import edu.illinois.cs.cogcomp.sl.core.IInstance
import edu.illinois.cs.cogcomp.sl.util.IFeatureVector
import edu.illinois.cs.cogcomp.sl.core.SLModel
import edu.illinois.cs.cogcomp.sl.core.SLParameters
import edu.illinois.cs.cogcomp.sl.core.SLProblem
import edu.illinois.cs.cogcomp.sl.core.AbstractFeatureGenerator
import edu.illinois.cs.cogcomp.sl.util.SparseFeatureVector
import edu.illinois.cs.cogcomp.sl.util.FeatureVectorBuffer
import edu.illinois.cs.cogcomp.sl.core.AbstractInferenceSolver

import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._

import java.util.ArrayList;
import java.nio.file.Files;
import java.nio.file.CopyOption._;
import java.io.PrintWriter;
import java.io.File;
import scala.util.control.Breaks._

//import gurobi._;

/**
 * @author machao
 */
object OntoNerStructLearning {
  
  var trainPath: String = "";
  var testPath: String = "";
  var algthm: Int = 1;
  
  def parseArg(args: Array[String]) {
    for (i <- 0 until args.length) {
      val str = args(i);
      if (str.equals("-trainPath")) {
        trainPath = args(i + 1);
      } else if (str.equals("-testPath")) {
        testPath = args(i + 1);
      } else if (str.equals("-algthm")) {
        algthm = Integer.parseInt(args(i + 1));
      }
    }
  }
  
  
  def main(args: Array[String]) {
    
    NerDriver.brownClustersPath = "lib/bllip-clusters";
    //lib/gender.data
    
    parseArg(args);
    NerTestingOntoTrainModel();
    //NerOntoModelTesting();
  }
  
  
  def NerTestingOntoTrainModel() {
    val trainPath: String = "/home/mc/workplace/rand_search/coref/berkfiles/data/ontonotes5/train";
    val testPath: String = "/home/mc/workplace/rand_search/coref/berkfiles/data/ontonotes5/test";
    trainEvaluateNerSystem(trainPath, -1, testPath, -1);
  }
  
  def computeWeight(sparseFeats: Array[Int], wght: Array[Double]): Double = {
    var result: Double = 0;
    for (idx <- sparseFeats) {
      result += wght(idx);
    }
    result;
  }
  
  class OntoNerExample(val transitionFeatures: Array[Array[Array[Int]]],
                       val featuresPerTokenPerState: Array[Array[Array[Int]]], 
                       val goldLabels: Array[Int]) extends IInstance {
    
    var labelAssignment = new Array[Int](goldLabels.length); // current assignment
    val seqExample = new SequenceExample(transitionFeatures, featuresPerTokenPerState, goldLabels);
    val goldFeatures: HashMap[Int,Double] = getSparseFeatureVector(goldLabels);
    
    def isCorrect(output: Array[Int]): Boolean = {
      if (output.length != goldLabels.length) {
        throw new RuntimeException("Wrong length for output: " + output.length);
      }
      output.corresponds(goldLabels){_ == _}; // every element should be the same
    }
    
    def getZeroOneLoss(output: Array[Int]): Float = {
      var loss: Float = 0
      for (i <- 0 until output.length) {
        if (output(i) != goldLabels(i)) {  loss += 1.0f; }
      }
      loss;
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
    
    def getGoldOutputFeatVec(): HashMap[Int,Double] =  {
      goldFeatures;
    }
    
    // construct a feature vector that sum over feature vectors on all edges
    def getSparseFeatureVector(output: Array[Int]): HashMap[Int,Double] = { // get value from labelAssignment
      
      if (output.length != goldLabels.length) {
        throw new RuntimeException("Wrong length for output: " + output.length);
      }
      
      val labelIndexer = NerSystemLabeled.StdLabelIndexer;
      val len = goldLabels.length;
      val valueMap = new HashMap[Int,Double]();
      
      // unary variables
      for (i <- 0 until len) { // token indices
        val lbl = output(i); // label
        val featUnary = featuresPerTokenPerState(i)(lbl);
        for (idx <- featUnary) {
          addValueToVector(valueMap, idx, 1.0);
        }
      }
      // binary variables
      for (i <- 0 until (len - 1)) { // token left indices
        val j = i + 1;// token right indices
        val lbl1 = output(i);
        val lbl2 = output(j);
        val featBin = transitionFeatures(lbl1)(lbl2)
        val prevLabel = labelIndexer.getObject(lbl1);
        val currLabel = labelIndexer.getObject(lbl2);
        if (!NerFeaturizer.isLegalTransition(prevLabel, currLabel)) {
          throw new RuntimeException("Illegal assignment: " + prevLabel +" "+ currLabel);
        }
        for (idx <- featBin) {
          addValueToVector(valueMap, idx, 1.0);
        }
      }
      
      
      //val norValMap = normalizeFeatVec(valueMap, len);
      //norValMap;
      valueMap; // return a sparse feature vector
    
    }
    
    def normalizeFeatVec(vMap: HashMap[Int,Double], len: Int): HashMap[Int,Double] = {
      val normValueMap = new HashMap[Int,Double]();
      val dem: Double = if (len > 0 ) len.toDouble; else 1.0d
      for ((idx, value) <- vMap) {
        normValueMap.put(idx, value / dem);
      }
      normValueMap;
    }
    
    def decode(weights: Array[Double]): Array[Int] = {
      val result = seqExample.decode(weights);
       labelAssignment = result; // store that!
      result;
    }
    
    def decodeByInferencer(wv: WeightVector, solver: BerkNerInferencer): Array[Int] = {
      val yhati = solver.getBestStructure(wv, this);
      val yhat = yhati.asInstanceOf[MyOutput].output;
      yhat;
    }
    
    def decodeByInferencerFromArrWeight(weights: Array[Double], solver: BerkNerInferencer): Array[Int] = {
      val wv: WeightVector = new WeightVector(weights.length);
      for (i <- 0 until weights.length) {
        wv.setElement(i + 1, weights(i).toFloat);
      }
      wv.setElement(0, 0.0f)
      decodeByInferencer(wv, solver);
    }
/*
    // return an array of label-indices
    def ilpInference(weights: Array[Double]) : Array[Int] = {
      
      // about grb model
      val env = new GRBEnv();
      val model = new GRBModel(env);
      
      val labelIndexer = NerSystemLabeled.StdLabelIndexer;
      val len = goldLabels.length;

      
      
      
      
      ////// Variables ////////////
      val allUnaryVars =  new ArrayBuffer[GRBVar]();
      val allBinaryVars = new ArrayBuffer[GRBVar]();
      
      // unary variables
      for (i <- 0 until len) { // token indices
        for (lbl <- 0 until labelIndexer.size) { // labels
          val currProb = computeWeight(featuresPerTokenPerState(i)(lbl), weights);//cNode.cachedBeliefsOrMarginals(jIdx) + nNodei.cachedBeliefsOrMarginals(niIdx);
          val vName: String = "VUnary@" + String.valueOf(i) + "@" + String.valueOf(lbl);// + "@" + nNodej.domain.value(njIdx);
          val newVar = model.addVar(0.0, 1.0, currProb, GRB.BINARY, vName);
          //varCoeffMap.put(newVar, currProb);
          //val ilpvar = (new IlpVariable(newVar, currProb));
        }
      }
      // binary variables
      for (i <- 0 until (len - 1)) { // token left indices
    	  val j = i + 1;// token right indices
    	  for (lbl1 <- 0 until labelIndexer.size) { // label left
    		  for (lbl2 <- 0 until labelIndexer.size) { // label right
    			  val prevLabel = labelIndexer.getObject(lbl1);
    			  val currLabel = labelIndexer.getObject(lbl2);
    			  if (NerFeaturizer.isLegalTransition(prevLabel, currLabel)) {
    				  val currProb = computeWeight(transitionFeatures(lbl1)(lbl2), weights);//cNode.cachedBeliefsOrMarginals(jIdx) + nNodei.cachedBeliefsOrMarginals(niIdx);
    				  val vName: String = "VBinary@" + String.valueOf(i) + "@" + String.valueOf(j) + "@" + String.valueOf(lbl1) + "@" + String.valueOf(lbl2);
    				  val newVar = model.addVar(0.0, 1.0, currProb, GRB.BINARY, vName);
    			  }
    		  }
    	  }

      }
      model.update();
      
      /////// Constraints ////////////
      // 1) unary value constrains
      
      
      // 2) unary binary consistency constrains
      
      
      model.update();
      
      
      ////// Objective /////////
      
      
      
      ///// Solve! ////

      
      
      ///// Parse result ////////
      
      
      ???
    }
    
    // inference with structural feature
    def ilpInferStruct(weights: Array[Double]) : Array[Int] = {
      ???
    }
*/
  }
  
  def trainEvaluateNerSystem(trainPath: String, trainSize: Int, testPath: String, testSize: Int) {

    val maybeBrownClusters = if (NerDriver.brownClustersPath != "") Some(BrownClusterInterface.loadBrownClusters(NerDriver.brownClustersPath, 0)) else None;
    val trainDocs = NerSystemLabeled.loadDocs(trainPath, trainSize, NerDriver.usePredPos)
    
    
    // for testing only
    val tstDocs = NerSystemLabeled.loadDocs(testPath, testSize, NerDriver.usePredPos);
    
    
    val system = trainOntoNerModel(trainDocs, tstDocs,  maybeBrownClusters, NerDriver.featureSet.split("\\+").toSet);
    
    
    // save model?
    if (!NerDriver.modelPath.isEmpty) {
      GUtil.save(system, NerDriver.modelPath);
    }
    
    // evaluate
    NerSystemLabeled.evaluateNerSystem(system, tstDocs);
  }
  
  
  
  def trainOntoNerModel(trainDocs: Seq[ConllDoc],
                        testDocs: Seq[ConllDoc],
                        maybeBrownClusters: Option[Map[String,String]],
                        nerFeatureSet: Set[String]) = {
    
    val labelIndexer = NerSystemLabeled.StdLabelIndexer;
    Logger.logss("Extracting training examples");
    val trainExamples = NerSystemLabeled.extractNerChunksFromConll(trainDocs);
    //showExamples(trainExamples); // by Chao
    val maybeWikipediaDB = None;
    val featureIndexer = new Indexer[String]();
    val nerFeaturizer = NerFeaturizer(nerFeatureSet, featureIndexer, labelIndexer, trainExamples.map(_.words), maybeWikipediaDB, maybeBrownClusters, NerDriver.unigramThreshold, NerDriver.bigramThreshold, NerDriver.prefSuffThreshold);
    
    println("Training example number = " + trainExamples.size);
    
    ////////////////////////////////////////////
    // Featurize transitions and then examples
    val featurizedTransitionMatrix = Array.tabulate(labelIndexer.size, labelIndexer.size)((prev, curr) => {
      nerFeaturizer.featurizeTransition(labelIndexer.getObject(prev), labelIndexer.getObject(curr), true);
    });
    Logger.startTrack("=== Featurizing ===");
    val trainSequenceExs = for (i <- 0 until trainExamples.size) yield {
      if (i % 1000 == 0) {
        Logger.logss("Featurizing train example " + i + "  feature length " + nerFeaturizer.featureIndexer.size());
      }
      val ex = trainExamples(i);
      //new SequenceExample(featurizedTransitionMatrix, nerFeaturizer.featurize(ex, true), ex.goldLabels.map(labelIndexer.getIndex(_)).toArray);
      new OntoNerExample(featurizedTransitionMatrix, nerFeaturizer.featurize(ex, true), ex.goldLabels.map(labelIndexer.getIndex(_)).toArray);
    };
    Logger.endTrack();
    ////////////////////////////////////////////
    //val featsByType = featureIndexer.getObjects().asScala.groupBy(str => str.substring(0, str.indexOf("=")));
    Logger.logss(featureIndexer.size + " features");

    
    
    // about test examples
    val testExamples = NerSystemLabeled.extractNerChunksFromConll(testDocs);
    println("Testing example number = " + testExamples.size);
    val testSequenceExs = for (i <- 0 until testExamples.size) yield {
      if (i % 100 == 0) {   Logger.logss("Featurizing test example " + i); }
      val ex = testExamples(i);
      new OntoNerExample(featurizedTransitionMatrix, nerFeaturizer.featurize(ex, false), ex.goldLabels.map(labelIndexer.getIndex(_)).toArray);
    };

    // Train
    val weights = if (algthm == 0) { // perceptron
      structureTraining(trainSequenceExs,  featureIndexer, testSequenceExs);
    } else if (algthm == 1) { // svm
      uiucStructLearning(trainSequenceExs,  featureIndexer);
    } else if (algthm == 2) { // berk adagradient
      originalBerkeleyTraining(trainSequenceExs,  featureIndexer);
    } else {
      throw new RuntimeException("Unknown learning algorithm! " + algthm);
    }
    
    /*
    val eta = 1.0;
    new GeneralLogisticRegression(true, false).trainWeightsAdagradL1R(trainSequenceExs.asJava, reg, eta, numItrs, batchSize, weights);
    val system = new NerSystemLabeled(labelIndexer, featurizedTransitionMatrix, nerFeaturizer, weights).pack;
    val trainGoldChunks = trainSequenceExs.map(ex => convertToLabeledChunks(ex.goldLabels.map(labelIndexer.getObject(_))));
    val trainPredChunks = trainSequenceExs.map(ex => convertToLabeledChunks(ex.decode(weights).map(labelIndexer.getObject(_))));
    */
    
    val system = new NerSystemLabeled(labelIndexer, featurizedTransitionMatrix, nerFeaturizer, weights);//.pack;
    val trainGoldChunks = trainSequenceExs.map(ex => NerSystemLabeled.convertToLabeledChunks(ex.goldLabels.map(labelIndexer.getObject(_))));
    val trainPredChunks = trainSequenceExs.map(ex => NerSystemLabeled.convertToLabeledChunks(ex.decode(weights).map(labelIndexer.getObject(_))));
    NEEvaluator.evaluateChunksBySent(trainGoldChunks, trainPredChunks);
    system;
    
  }
  
  def decodeAndEvalOnto(testExs: IndexedSeq[OntoNerExample],
                        wght: Array[Double],
                        writer: Option[PrintWriter]) {
    // Decode and check test set accuracy
    val testGoldChunks = testExs.map(ex => NerSystemLabeled.convertToLabeledChunks(ex.goldLabels.map(NerSystemLabeled.StdLabelIndexer.getObject(_))));
    val testPredChunks = testExs.map(ex => NerSystemLabeled.convertToLabeledChunks(ex.decode(wght).map(NerSystemLabeled.StdLabelIndexer.getObject(_))));

    
    val allGoldChunksBySent = testGoldChunks;
    val allPredChunksBySent = testPredChunks;
    val ignoreCase: Boolean = false;
    
    var correct = 0;
    val correctByLabel = new Counter[String];
    var totalPred = 0;
    val totalPredByLabel = new Counter[String];
    var totalGold = 0;
    val totalGoldByLabel = new Counter[String];
    for ((goldChunksRaw, predChunksRaw) <- allGoldChunksBySent.zip(allPredChunksBySent)) {
      val goldChunks = if (ignoreCase) goldChunksRaw.map(chunk => new Chunk(chunk.start, chunk.end, chunk.label.toLowerCase)); else goldChunksRaw;
      val predChunks = if (ignoreCase) predChunksRaw.map(chunk => new Chunk(chunk.start, chunk.end, chunk.label.toLowerCase)); else predChunksRaw;
      totalPred += predChunks.size;
      predChunks.foreach(chunk => totalPredByLabel.incrementCount(chunk.label, 1.0));
      totalGold += goldChunks.size;
      goldChunks.foreach(chunk => totalGoldByLabel.incrementCount(chunk.label, 1.0));
      for (predChunk <- predChunks) {
        if (goldChunks.contains(predChunk)) {
          correct += 1;
          correctByLabel.incrementCount(predChunk.label, 1.0)
        }
      }
    }
    val pre: Double = correct.toDouble / totalPred.toDouble;
    val rec: Double = correct.toDouble / totalGold.toDouble;
    val f1: Double = 2 / (1/pre + 1/rec);
    if (writer != None) {
      writer.get.println(correct + "," + totalPred + "," + totalGold + "," +f1);
    }
    Logger.logss("Results: " + GUtil.renderPRF1(correct, totalPred, totalGold));
    //NEEvaluator.evaluateChunksBySent(testGoldChunks, testPredChunks);
  }
  
  def structureTraining(trainExs: IndexedSeq[OntoNerExample], 
                        featureIndexer: Indexer[String],
                        testExs: IndexedSeq[OntoNerExample]): Array[Double] = {

    val logger = new PrintWriter("ner_ontonotes5_learning_curve_devset.csv");
    val infSolver = new BerkNerInferencer(NerSystemLabeled.StdLabelIndexer.size());
    
    var weight = Array.fill[Double](featureIndexer.size)(0);
    var weightSum = Array.fill[Double](featureIndexer.size)(0);
    val Iteration = 200;
    val learnRate = 0.1;
    val lambda = 1e-8;
    
    var updateCnt = 0;
    
    for (iter <- 0 until Iteration) {
      println("Iter " + iter);
      for (example <- trainExs) {
        val currentOutput = example.decode(weight); // inference with current weight
        
        // update?
        if (!example.isCorrect(currentOutput)) {
          updateCnt += 1;
          if (updateCnt % 1000 == 0) println("Update " + updateCnt);
          updateWeight(weight: Array[Double], 
                       example.getGoldOutputFeatVec(),
                       example.getSparseFeatureVector(currentOutput),
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
      decodeAndEvalOnto(testExs, tmpAvg, Some(logger));
    }
    
    divdeNumber(weightSum, updateCnt.toDouble);

    //for (i <- 0 until weightSum.length) {
    //  if (weightSum(i) != 0) {
    //    println("weight(" + i + ") = " + weightSum(i));
    //  }
    //}
    
    logger.close();
    
    weightSum;
    //weight;
  }
  
  def updateWeight(currentWeight: Array[Double], 
                   featGold: HashMap[Int,Double],
                   featPred: HashMap[Int,Double],
                   eta: Double,
                   lambda: Double) {
    var gradient = Array.fill[Double](currentWeight.length)(0);//new Array[Double](currentWeight.length);
    for ((idx, value) <- featGold) {
      gradient(idx) += (value);
    }
    for ((jdx, walue) <- featPred) {
      gradient(jdx) -= (walue);
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
  
  // training give by berkeley system
  def originalBerkeleyTraining(trainExs: IndexedSeq[OntoNerExample], 
                               featureIndexer: Indexer[String]): Array[Double] = {
    val trainSeqExs =  for (i <- 0 until trainExs.size) yield {
      val ex = trainExs(i);
      new SequenceExample(ex.transitionFeatures, ex.featuresPerTokenPerState, ex.goldLabels);
    };

    val reg: Double = NerDriver.reg;
    val numItrs: Int = NerDriver.numItrs;
    val batchSize: Int = NerDriver.batchSize;
    
    val eta = 1.0;
    val trainer = new GeneralLogisticRegression(true, false)
    var weight = Array.fill[Double](featureIndexer.size)(0);
    trainer.trainWeightsAdagradL1R(trainSeqExs.asJava, reg, eta, numItrs, batchSize, weight);
    weight;
  }
  
  
  
  ////////////////////////////
  // Testing
  ////////////////////////////
  
  // loading model to test performance
  def NerOntoModelTesting() {
    //val testPath: String = "/scratch/entitysystem/OregonStateEntity/data/ontonotes5/test10";
    val testSize: Int = -1;
    val tstDocs = NerSystemLabeled.loadDocs(testPath, testSize, NerDriver.usePredPos);
    
    // load model
    val modelPath = "tmp1-ner-onto.ser.gz";
    val model = GUtil.load(modelPath).asInstanceOf[NerSystemLabeled];
    
    // run testing!
    //NerSystemLabeled.evaluateNerSystem(model, tstDocs);
    myTestNerSystem(model, tstDocs);
  }
  
  def myTestNerSystem(nerSystem: NerSystemLabeled, testDocs: Seq[ConllDoc]) {
    val labelIndexer = nerSystem.labelIndexer;
    Logger.logss("Extracting test examples");
    val testExamples = NerSystemLabeled.extractNerChunksFromConll(testDocs);
    val testSequenceExs = for (i <- 0 until testExamples.size) yield {
      if (i % 100 == 0) {
        Logger.logss("Featurizing test example " + i);
      }
      val ex = testExamples(i);
      new OntoNerExample(nerSystem.featurizedTransitionMatrix, nerSystem.featurizer.featurize(ex, false), ex.goldLabels.map(nerSystem.labelIndexer.getIndex(_)).toArray);
    };
    
    val inferencer = new BerkNerInferencer(NerSystemLabeled.StdLabelIndexer.size());
    // Decode and check test set accuracy
    val testGoldChunks = testSequenceExs.map(ex => NerSystemLabeled.convertToLabeledChunks(ex.goldLabels.map(labelIndexer.getObject(_))));
    //val testPredChunks = testSequenceExs.map(ex => NerSystemLabeled.convertToLabeledChunks(ex.ilpInference(nerSystem.weights).map(labelIndexer.getObject(_))));
    val testPredChunks = testSequenceExs.map(ex => NerSystemLabeled.convertToLabeledChunks(ex.decodeByInferencerFromArrWeight(nerSystem.weights, inferencer).map(labelIndexer.getObject(_))));
    NEEvaluator.evaluateChunksBySent(testGoldChunks, testPredChunks);
  }
  
  
  //////////////////////////////////////
  // for UIUC Structural Learning Lib //
  //////////////////////////////////////
  
  class MyOutput(var output: Array[Int]) extends IStructure {
    def setOutput(newOutput: Array[Int]) {
      if (newOutput.length != output.length) {
        throw new RuntimeException("Wrong length for output: " + newOutput.length);
      }
      output = newOutput;
    }
    
    //@Override
    override def equals(o1: Any): Boolean = {
      val vec = o1.asInstanceOf[MyOutput];
      if (vec.output.length != output.length) {
        throw new RuntimeException("Wrong length for output: " + vec.output.length);
      }
      output.corresponds(vec.output){_ == _};
    }
  }
  
  class BerkNerFeatGener extends AbstractFeatureGenerator {
    def getFeatureVector(xi: IInstance, yhati: IStructure): IFeatureVector = {
      val x = xi.asInstanceOf[OntoNerExample];
      val yhat = yhati.asInstanceOf[MyOutput];
      
      val normalizedDem: Double = yhat.output.length.toDouble;
      
      val fb = new FeatureVectorBuffer();
      val valMap: HashMap[Int,Double] = x.getSparseFeatureVector(yhat.output);
      //val idxArr = new ArrayBuffer[Int]();
      //val valArr = new ArrayBuffer[Double]();
      for (idx <- valMap.keySet) {
        //val normVal = if (normalizedDem == 0) { 0.0d } else {  / normalizedDem }
        fb.addFeature(idx + 1, valMap.get(idx).get);
      }
      fb.toFeatureVector();
    } 
    override def getFeatureVectorDiff(x: IInstance, y1: IStructure, y2: IStructure): IFeatureVector = {
    		val f1 = getFeatureVector(x, y1);
    		val f2 = getFeatureVector(x, y2);		
    		return f1.difference(f2);
    }
  }
  
  class BerkNerInferencer(val nLabels: Int) extends AbstractInferenceSolver {
    def getBestStructure(wi: WeightVector, xi: IInstance): IStructure = {
      getLossAugmentedBestStructure(wi, xi, null);
    }
    def getLoss(xi: IInstance, ystari: IStructure, yhati: IStructure): Float = {
      val x = xi.asInstanceOf[OntoNerExample];
      val yhat = yhati.asInstanceOf[MyOutput];
      x.getZeroOneLoss(yhat.output); // non-normalized loss
    }
    def getLossAugmentedBestStructure(wi: WeightVector, xi: IInstance, ystari: IStructure): IStructure = {
    	val x = xi.asInstanceOf[OntoNerExample];
    	//val ystar = new MyOutput(x.goldLabels);
    	val gold = x.goldLabels;

    	val numOflabels = NerSystemLabeled.StdLabelIndexer.size();
    	val numOfTokens = gold.length;

    	var dpTable = Array.ofDim[Float](numOfTokens, numOflabels);
    	var path = Array.ofDim[Int](numOfTokens, numOflabels);
/*
      for (i <- 0 until numOfTokens) {
        println("(" + i  +") = " + x.featuresPerTokenPerState(i).length);
        for (j <- 0 until numOflabels) {
         println("(" + i + "," + j +") = " + x.featuresPerTokenPerState(i)(j));
        }
      }
 */     
    		// Viterbi algorithm
    	for (j <- 0 until numOflabels) {
    		val priorScore: Float = 0;
        val lossAug = ( if (ystari != null && j != gold(0)) 1.0 else 0.0).toFloat;
    		val zeroOrderScore =  (computeFeatDotWeight(x.featuresPerTokenPerState(0)(j), wi) + lossAug).toFloat;
    		dpTable(0)(j) = priorScore + zeroOrderScore;   
    		path(0)(j) = -1;
    	}

    	for (i <- 1 until numOfTokens) {
    	  for (j <- 0 until numOflabels) {
          val lossAug = ( if (ystari != null && j != gold(i)) 1.0 else 0.0).toFloat;
    			val zeroOrderScore = (computeFeatDotWeight(x.featuresPerTokenPerState(i)(j), wi) + lossAug).toFloat;
    			var bestScore = Float.NegativeInfinity;
          for (k <- 0 until numOflabels) {
    				val candidateScore = if (x.transitionFeatures(k)(j) == null) {
              Float.NegativeInfinity;
            } else {
              dpTable(i-1)(k) + computeFeatDotWeight(x.transitionFeatures(k)(j), wi).toFloat;
            }
            
    				if (candidateScore > bestScore) {
    					bestScore = candidateScore;
    					path(i)(j) = k;
    				}
    			}
    			dpTable(i)(j) = zeroOrderScore + bestScore;
    		}
    	}

    	// find the best sequence   
    	var tags = new Array[Int](numOfTokens);

    	var maxTag = 0;
    	for (i <- 0 until numOflabels) {
    		if (dpTable(numOfTokens - 1)(i) > dpTable(numOfTokens - 1)(maxTag)) 
    			maxTag = i;
    	}
    	tags(numOfTokens - 1) = maxTag;

    	for (i <- (numOfTokens - 1) to 1 by -1) { 
    		tags(i-1) = path(i)(tags(i));
    	}

    	new MyOutput(tags);
    }
  }
  
  def computeFeatDotWeight(fv: Array[Int], wv: WeightVector): Float = {
    var result: Float = 0;
    if (fv == null) {
      return Float.NegativeInfinity;//throw new RuntimeException("null fv!");
    }
    for (idx <- fv) {
      result += wv.get(idx + 1);
    }
    result += wv.get(0); // add bias?
    result;
  }
  
  def uiucStructLearning(trainExs: IndexedSeq[OntoNerExample], 
                         featureIndexer: Indexer[String]): Array[Double] = {
    
    //val slcfgPath = "config/uiuc-sl-config/StructuredPerceptron.config";
    //val slcfgPath = "config/uiuc-sl-config/DCD.config";
    val slcfgPath = "../sl-config/ontonotes-ner-search-DCD.config";
    val model = new SLModel();
    model.lm = new Lexiconer();
    
    val sp: SLProblem = new SLProblem();//readStructuredData(trainingDataPath, model.lm);
    for (ex <- trainExs) {
      val goldOutput = new MyOutput(ex.goldLabels);
      sp.addExample(ex, goldOutput);
    }

    // Disallow the creation of new features
    model.lm.setAllowNewFeatures(false);

    // initialize the inference solver
    model.infSolver = new BerkNerInferencer(NerSystemLabeled.StdLabelIndexer.size());

    val fg = new BerkNerFeatGener();
    val para = new SLParameters();
    para.loadConfigFile(slcfgPath);
    para.TOTAL_NUMBER_FEATURE = featureIndexer.size();
      
    val learner: Learner = LearnerFactory.getLearner(model.infSolver, fg, para);
    model.wv = learner.train(sp);
    WeightVector.printSparsity(model.wv);
    //if(learner instanceof L2LossSSVMLearner)
    //  System.out.println("Primal objective:" + ((L2LossSSVMLearner)learner).getPrimalObjective(sp, model.wv, model.infSolver, para.C_FOR_STRUCTURE));

    // save the model
    //model.saveModel(modelPath);
    getDoubleWeightVector(model.wv);
  }
  
  def getDoubleWeightVector(wv: WeightVector): Array[Double] = {
    val farr = wv.getWeightArray; // (0) is bias
    var darr = new Array[Double](farr.length - 1);
    for (i <- 0 until darr.length) {
      darr(i) = farr(i + 1).toDouble;
    }
    darr;
  }

}