package berkeleyentity.nersearch

import scala.util.Random
import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._
import java.io.FileInputStream
import java.io.ObjectInputStream
import java.io.File
import java.io.FileOutputStream
import java.io.ObjectOutputStream
import java.io.PrintWriter;
import edu.berkeley.nlp.futile.fig.basic.Indexer
import berkeleyentity.ConllDoc
import edu.berkeley.nlp.futile.classify.GeneralLogisticRegression
import berkeleyentity.coref.CorefSystem
import edu.berkeley.nlp.futile.util.Logger
import berkeleyentity.GUtil
import edu.berkeley.nlp.futile.classify.SequenceExample
import edu.berkeley.nlp.futile.fig.basic.IOUtils
import edu.berkeley.nlp.futile.util.Counter
import berkeleyentity.Chunk
import scala.collection.mutable.HashMap
import berkeleyentity.ConllDocReader
import berkeleyentity.lang.Language
import berkeleyentity.ConllDocWriter
import edu.berkeley.nlp.math.SloppyMath
import berkeleyentity.wiki.WikipediaInterface
import berkeleyentity.coref.UID
import berkeleyentity.sem.BrownClusterInterface
import berkeleyentity.ner.NerFeaturizer
import berkeleyentity.ner.NerExample
import berkeleyentity.ner.NEEvaluator
import berkeleyentity.ner.NerDriver

@SerialVersionUID(1L)
class NerSystemConll2003(val labelIndexer: Indexer[String],
                         val featurizedTransitionMatrix: Array[Array[Array[Int]]],
                         val featurizer: NerFeaturizer,
                         val weights: Array[Double]) extends Serializable {
  
  def tagBIO(sentenceWords: Array[String], sentencePos: Array[String]): Array[String] = {
    val example = new NerExample(sentenceWords, sentencePos, null);
    val seqExample = new SequenceExample(featurizedTransitionMatrix, featurizer.featurize(example, false), null);
    seqExample.decode(weights).map(labelIndexer.getObject(_));
  }
  
  def chunk(sentenceWords: Array[String], sentencePos: Array[String]): Seq[Chunk[String]] = {
    val example = new NerExample(sentenceWords, sentencePos, null);
    val seqExample = new SequenceExample(featurizedTransitionMatrix, featurizer.featurize(example, false), null);
    Conll2003NerSystem.convertToLabeledChunks(seqExample.decode(weights).map(labelIndexer.getObject(_)));
  }
  
}

object Conll2003NerSystem {
  val TagSet = IndexedSeq("PER", "ORG", "LOC", "MISC");
  val LabelSetHead = IndexedSeq("B", "I", "O", "E", "S");
  val Conll2003LabelIndexer = new Indexer[String]();
  Conll2003LabelIndexer.add("O");
  for (tag <- TagSet) {
    Conll2003LabelIndexer.add("B-" + tag);
    Conll2003LabelIndexer.add("I-" + tag);
    Conll2003LabelIndexer.add("E-" + tag);
    Conll2003LabelIndexer.add("S-" + tag);
  }

  def getStructuralType(nerSymbol: String) = {
    if (nerSymbol.contains("-")) {
      nerSymbol.substring(0, 1);
    } else if (nerSymbol == "O") {
      "O"
    } else {
      throw new RuntimeException("something wrong tag")
    }
  }
  
  // Designed to handle both ACE classes like "VEH", "FAC", etc. (multiclass), along with
  // OntoNotes tag sequences like "B-ORG", "I-PER", etc.
  // Semantic type is whatever comes after the hyphen or the entire thing if no hyphen
  def getSemanticType(nerSymbol: String) = {
    if (nerSymbol.contains("-")) {
      require(getStructuralType(nerSymbol) != "O");
      nerSymbol.substring(nerSymbol.indexOf("-") + 1);
    } else {
      throw new RuntimeException("something wrong tag type")
    }
  }

  def main(args: Array[String]) {
     //trainEvaluateNerSystem("/home/mc/workplace/rand_search/ner2003/ner/eng.train", 
     //                       "/home/mc/workplace/rand_search/ner2003/ner/eng.testb");
     dumpForLstm();
  }
  
  def dumpForLstm() {
    dumpExmpNoPos("/home/mc/workplace/rand_search/ner2003/ner/eng.train", "/home/mc/workplace/deepsp/sequence_tagging-master/data/train-eng.txt")
    dumpExmpNoPos("/home/mc/workplace/rand_search/ner2003/ner/eng.testa", "/home/mc/workplace/deepsp/sequence_tagging-master/data/testa-eng.txt")
    dumpExmpNoPos("/home/mc/workplace/rand_search/ner2003/ner/eng.testb", "/home/mc/workplace/deepsp/sequence_tagging-master/data/testb-eng.txt")
  }
  
  
  
  def trainEvaluateNerSystem(trainPath: String, testPath: String) {
    val brownPath = "/home/mc/workplace/rand_search/coref/berkfiles/data/bllip-clusters";
    val maybeBrownClusters = Some(BrownClusterInterface.loadBrownClusters(brownPath, 0))
    val system = trainNerSystem(trainPath, maybeBrownClusters, NerDriver.featureSet.split("\\+").toSet, NerDriver.reg, 30, NerDriver.batchSize);//NerDriver.numItrs
    evaluateNerSystem(system, testPath); 
  }
    
  // TRAINING
  def showExamples(trainExpls : Seq[NerExample]) {
    for (ex <- trainExpls) {
      println("=== Example start ===");
      for (i <- (0 until ex.words.size)) {
        println(ex.wordAt(i) + " " + ex.posAt(i) + " " + ex.goldLabels(i));
      }
      println("=== Example end ===");
    }
  }
  
  
  class MySequenceExample(val transitionFeatures: Array[Array[Array[Int]]], 
                          val featuresPerTokenPerState: Array[Array[Array[Int]]], 
                          val goldLabels: Array[Int]) {
    def getNumTokens(): Int = {
      goldLabels.length;
    }
  }
 
  // dump for seq2seq lstm
  def dumpExmpNoPos(trainPath: String, outPath: String) {
    val writer = new PrintWriter(outPath);
    val examples = Conll2003NerInstanceLoader.loadNerExamples(trainPath)
    for (ex <- examples) {
      val exLen = ex.words.size
      for (i <- 0 until exLen) {
        writer.println(ex.words(i) + " " + ex.goldLabels(i))
      }
      writer.println(); // empty line
    }
    writer.close()
    println("Dump " + examples.size + " sentences.")
  }
  
  def trainNerSystem(trainPath: String,//trainDocs: Seq[ConllDoc],
                     maybeBrownClusters: Option[Map[String,String]],
                     nerFeatureSet: Set[String],
                     reg: Double,
                     numItrs: Int, 
                     batchSize: Int) = {
    val labelIndexer = Conll2003LabelIndexer;
    Logger.logss("Extracting training examples");
    val trainExamples = Conll2003NerInstanceLoader.loadNerExamples(trainPath)//extractNerChunksFromConll(trainDocs);
    val maybeWikipediaDB = None;
    val featureIndexer = new Indexer[String]();
    
    val unigramThreshold = 1;
    val bigramThreshold = 10;
    val prefSuffThreshold = 2;
    val nerFeaturizer = NerFeaturizer(nerFeatureSet, featureIndexer, labelIndexer, trainExamples.map(_.words), maybeWikipediaDB, maybeBrownClusters, unigramThreshold, bigramThreshold, prefSuffThreshold);
    // Featurize transitions and then examples
    val featurizedTransitionMatrix = Array.tabulate(labelIndexer.size, labelIndexer.size)((prev, curr) => {
      nerFeaturizer.featurizeTransition(labelIndexer.getObject(prev), labelIndexer.getObject(curr), true);
    });
    Logger.startTrack("Featurizing");
    val trainSequenceExs = for (i <- 0 until trainExamples.size) yield {
      if (i % 1000 == 0) {
        Logger.logss("Featurizing train example " + i);
      }
      val ex = trainExamples(i);
      new SequenceExample(featurizedTransitionMatrix, nerFeaturizer.featurize(ex, true), ex.goldLabels.map(labelIndexer.getIndex(_)).toArray);
    };
    Logger.endTrack();
    val featsByType = featureIndexer.getObjects().asScala.groupBy(str => str.substring(0, str.indexOf("=")));
    Logger.logss(featureIndexer.size + " features");
    // Train
    val weights = new Array[Double](featureIndexer.size);
    val eta = 1.0;
    new GeneralLogisticRegression(true, false).trainWeightsAdagradL1R(trainSequenceExs.asJava, reg, eta, numItrs, batchSize, weights);
    val trainGoldChunks = trainSequenceExs.map(ex => convertToLabeledChunks(ex.goldLabels.map(labelIndexer.getObject(_))));
    val trainPredChunks = trainSequenceExs.map(ex => convertToLabeledChunks(ex.decode(weights).map(labelIndexer.getObject(_))));
    NEEvaluator.evaluateChunksBySent(trainGoldChunks, trainPredChunks);
    val system = new NerSystemConll2003(labelIndexer, featurizedTransitionMatrix, nerFeaturizer, weights);//.pack;
    system;
  }
  
  // EVALUATION
  
  def evaluateNerSystem(nerSystem: NerSystemConll2003, testFile: String) {
    val labelIndexer = nerSystem.labelIndexer;
    Logger.logss("Extracting test examples");
    val testExamples = Conll2003NerInstanceLoader.loadNerExamples(testFile)//extractNerChunksFromConll(testDocs);
    val testSequenceExs = for (i <- 0 until testExamples.size) yield {
      if (i % 1000 == 0) {
        Logger.logss("Featurizing test example " + i);
      }
      val ex = testExamples(i);
      new SequenceExample(nerSystem.featurizedTransitionMatrix, nerSystem.featurizer.featurize(ex, false), ex.goldLabels.map(nerSystem.labelIndexer.getIndex(_)).toArray);
    };
    // Decode and check test set accuracy
    val testGoldChunks = testSequenceExs.map(ex => convertToLabeledChunks(ex.goldLabels.map(labelIndexer.getObject(_))));
    val testPredChunks = testSequenceExs.map(ex => convertToLabeledChunks(ex.decode(nerSystem.weights).map(labelIndexer.getObject(_))));
    NEEvaluator.evaluateChunksBySent(testGoldChunks, testPredChunks);
    runMyEvaluation(testSequenceExs, testExamples, nerSystem, labelIndexer)
  }
  
  def runMyEvaluation(tstExs: IndexedSeq[SequenceExample], nerExs: Seq[NerExample], nerSystem: NerSystemConll2003, labelIndexer: Indexer[String]) {
    for (i <- 0 until tstExs.size) {
      val sqex = tstExs(i)
      val nerEx = nerExs(i)
      val golds = convertToLabeledChunks(sqex.goldLabels.map(labelIndexer.getObject(_)))
      val preds = convertToLabeledChunks(sqex.decode(nerSystem.weights).map(labelIndexer.getObject(_)))
      
      var correct = 0;
      val correctByLabel = new Counter[String];
      var totalPred = 0;
      val totalPredByLabel = new Counter[String];
      var totalGold = 0;
      val totalGoldByLabel = new Counter[String];

      val goldChunks = golds
    	val predChunks = preds
    	totalPred += predChunks.size;
      predChunks.foreach(chunk => totalPredByLabel.incrementCount(chunk.label, 1.0));
      totalGold += goldChunks.size;
      goldChunks.foreach(chunk => totalGoldByLabel.incrementCount(chunk.label, 1.0));
      
      for (predChunk <- predChunks) {
        for (goldChunk <- goldChunks) {
          if (predChunk.start == goldChunk.start && predChunk.end == goldChunk.end) {
            if (predChunk.label == goldChunk.label) {
              correct += 1;
    		      correctByLabel.incrementCount(predChunk.label, 1.0)
            } else {
              println("Wrong tag chunk: " + predChunk + " " + nerEx.getSpan(predChunk.start, predChunk.end) + " (" + goldChunk + ")")
            }
          }
        }
      }
      
      //for (predChunk <- predChunks) {
    	//  if (goldChunks.contains(predChunk)) {
    	//	  correct += 1;
    	//	  correctByLabel.incrementCount(predChunk.label, 1.0)
    	//  } else {
    	//	  println("Wrong chunk: " + predChunk + nerEx.getSpan(predChunk.start, predChunk.end))
    	//  }
     // }
    	  
    	  
      //Logger.logss("Results: " + GUtil.renderPRF1(correct, totalPred, totalGold));
      //if (printFineGrainedResults) {
    	  //for (tag <- totalGoldByLabel.keySet.asScala.toSeq.sorted) {
    		//  Logger.logss("  Results for " + GUtil.padToK(tag, 11) + ": " + GUtil.renderPRF1(correctByLabel.getCount(tag).toInt, totalPredByLabel.getCount(tag).toInt, totalGoldByLabel.getCount(tag).toInt));
    	  //}
      //}
    }
  }
  
  def convertToLabeledChunks(labelSeq: Seq[String]): Seq[Chunk[String]] = {
    val chunks = new ArrayBuffer[Chunk[String]];
    var i = 0;
    var inconsistent = false;
    while (i < labelSeq.size) {
      val structuralType = getStructuralType(labelSeq(i));
      if (structuralType == "B") {
        val semanticType = getSemanticType(labelSeq(i));
        val startIdx = i;
        i += 1;
        while (i < labelSeq.size && (getStructuralType(labelSeq(i)) == "I" || getStructuralType(labelSeq(i)) == "E")) {
          if (getSemanticType(labelSeq(i)) != semanticType) {
            inconsistent = true;
          }
          i += 1;
        }
        chunks += new Chunk[String](startIdx, i, semanticType);
      
      } else if (structuralType == "S") {
        val semanticType = getSemanticType(labelSeq(i));
        val startIdx = i;
        i += 1;
        chunks += new Chunk[String](startIdx, i, semanticType);
      
      } else if (structuralType == "I") {
        inconsistent = true;
        i += 1;
      } else {
        i += 1;
      }
    }
    if (inconsistent) {
      Logger.logss("WARNING: Inconsistent NER sequence: " + labelSeq);
    }
    chunks;
  }
}
