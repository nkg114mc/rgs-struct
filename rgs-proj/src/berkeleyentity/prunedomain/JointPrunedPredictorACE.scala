package berkeleyentity.prunedomain

import berkeleyentity.coref.PairwiseScorer
import berkeleyentity.ner.NerSystemLabeled
import berkeleyentity.coref.CorefPruner
import berkeleyentity.ner.NerPruner
import berkeleyentity.coref.MentionPropertyComputer
import berkeleyentity.lang.Language
import berkeleyentity.coref.CorefSystem
import berkeleyentity.ConllDocReader
import berkeleyentity.coref.CorefDocAssembler
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import berkeleyentity.Chunk
import berkeleyentity.ConllDoc
import berkeleyentity.coref.DocumentGraph
import edu.berkeley.nlp.futile.fig.exec.Execution
import berkeleyentity.coref.CorefEvaluator
import berkeleyentity.wiki.WikipediaInterface
import berkeleyentity.ConllDocWriter
import berkeleyentity.coref.OrderedClustering
import berkeleyentity.ner.NerFeaturizer
import berkeleyentity.ner.NEEvaluator
import edu.berkeley.nlp.futile.fig.basic.IOUtils
import edu.berkeley.nlp.futile.util.Logger
import berkeleyentity.Driver
import berkeleyentity.GUtil
import berkeleyentity.coref.CorefDocAssemblerACE
import berkeleyentity.ner.MCNerFeaturizer
import berkeleyentity.wiki.CorpusWikiAnnots
import berkeleyentity.coref.CorefDoc
import berkeleyentity.wiki.ACEMunger
import berkeleyentity.coref.OrderedClusteringBound
import berkeleyentity.wiki.WikificationEvaluator
import berkeleyentity.wiki._
import berkeleyentity.ilp.HistgramRecord
import scala.collection.mutable.HashSet
import java.io.PrintWriter
import java.util.ArrayList;
import berkeleyentity.joint.JointComputerShared
import berkeleyentity.joint.PrunedGraphFactoryACE
import berkeleyentity.joint.JointDocACE
import berkeleyentity.joint.JointFeaturizerShared

@SerialVersionUID(2L)
class JointPrunedPredictorACE(val jointFeaturizer: JointFeaturizerShared[MCNerFeaturizer],
                              val weights: Array[Float],
                              val corefPruner: CorefPruner,
                              val domainModel: GraphPrunerModelACE) extends Serializable {

  def decodeWriteOutputEvaluate(jointTestDocs: Seq[JointDocACE], maybeWikipediaInterface: Option[WikipediaInterface], doConllPostprocessing: Boolean, wikiLabelsInTrain: Set[String] = Set[String]()) {
    val allPredBackptrsAndClusterings = new ArrayBuffer[(Array[Int],OrderedClustering)];
    val allPredNEChunks = new ArrayBuffer[Seq[Seq[Chunk[String]]]];
    val allPredWikiChunks = new ArrayBuffer[Seq[Chunk[String]]];
    val allPredWikiTitles = new ArrayBuffer[Set[String]];
    
    val histg = new HistgramRecord();
    println("Weight length = " + weights.length);
    val weightFile : String = "myWeight.log";
    //showWeightValues();
    //showJointFeatWeightValues();
    
    println("writeWikiOutput = " + Driver.writeWikiOutput);
    Driver.writeWikiOutput = true;
    val predWriter = if (Driver.outputPath != "") Some(IOUtils.openOutHard(Driver.outputPath)) else None;
    val goldWriter = if (Driver.writeGold) Some(IOUtils.openOutHard(Execution.getFile("gold.conll"))) else None;
    val predWikiWriter = if (Driver.writeWikiOutput) Some(IOUtils.openOutHard(Execution.getFile("wiki.txt"))) else None;
    val predWikiWriterAux = if (Driver.writeWikiOutput) Some(IOUtils.openOutHard(Execution.getFile("wiki-aux.txt"))) else None;
    val maybeRawChunkNames = if (Driver.writeWikiOutput && Driver.allAcePath != "" && Driver.rawWikiGoldPath != "") {
      Some(ACEMunger.mungeACEToGetChunkLabels(Driver.allAcePath, Driver.rawWikiGoldPath))
    } else {
      None
    }
    val outWriter = IOUtils.openOutHard(Execution.getFile("output.conll"))
    val outWikiWriter = IOUtils.openOutHard(Execution.getFile("output-wiki.conll"))
    Logger.startTrack("Decoding");
    for (i <- (0 until jointTestDocs.size)) {
      //Logger.logss("Decoding " + i);
      val jointDevDoc = jointTestDocs(i);
      
      val outDir = Driver.myoutputPath;//"/scratch/EntityLinking2015/berkeley-entity-master/data/ace05/myoutput";
      val outWikiDir = Driver.myoutputPath;//"/scratch/EntityLinking2015/berkeley-entity-master/data/ace05/myoutput_wiki";
      val docPredPath = outDir + "/" + jointDevDoc.rawDoc.docID + ".mypred.conll";
      val docPredWikiPath = outWikiDir + "/" + jointDevDoc.rawDoc.docID + "-wiki.mypred.conll";
      val docPredWriter = IOUtils.openOutHard(docPredPath);
      val docPredWikiWriter = IOUtils.openOutHard(docPredWikiPath);
      
      val (backptrs, clustering, nerChunks, wikiChunks) = decode(jointDevDoc, maybeWikipediaInterface);
      val goldNerChunks = jointDevDoc.goldChunks;
      val predClusteringBound = new OrderedClusteringBound(jointDevDoc.docGraph.getMentions, clustering)
      val goldClusteringBound = new OrderedClusteringBound(jointDevDoc.docGraph.corefDoc.goldMentions, jointDevDoc.docGraph.corefDoc.goldClustering)
      if (predWriter.isDefined) ConllDocWriter.writeDocWithPredAnnotations(predWriter.get, jointDevDoc.rawDoc, nerChunks, predClusteringBound, Some(wikiChunks));
      if (goldWriter.isDefined) {
        val goldWikiAnnotsToWrite = Some(jointDevDoc.goldWikiChunks.map(_.map(chunk => {
          new Chunk[String](chunk.start, chunk.end, if (chunk.label.size == 0) NilToken else chunk.label(0).replace("_", " ").toLowerCase)
        })));
        ConllDocWriter.writeDocWithPredAnnotations(goldWriter.get, jointDevDoc.rawDoc, jointDevDoc.goldChunks, goldClusteringBound, goldWikiAnnotsToWrite);
      }
      
      allPredBackptrsAndClusterings += (backptrs -> clustering);
      allPredNEChunks += nerChunks;
      allPredWikiChunks ++= wikiChunks;
      allPredWikiTitles += WikificationEvaluator.convertChunksToBagOfTitles(wikiChunks);
      if (predWikiWriter.isDefined && maybeRawChunkNames.isDefined) {
        WikificationEvaluator.writeWikificationRightAndWrong(predWikiWriter.get, predWikiWriterAux.get, jointDevDoc, jointDevDoc.goldWikiChunks,
                                                                  maybeRawChunkNames.get, jointDevDoc.rawDoc.docID, wikiChunks, wikiLabelsInTrain);
      }
      ConllDocWriter.writeDocWithPredAnnotationsWikiStandoff(outWriter, outWikiWriter, jointDevDoc.rawDoc, nerChunks, clustering.bind(jointDevDoc.docGraph.getMentions, doConllPostprocessing), wikiChunks);
    
      // by MC
      //ConllDocWriter.writeDocWithPredAnnotations(docPredWriter, jointDevDoc.rawDoc, nerChunks, predClusteringBound, None);//Some(wikiChunks));
      //ConllDocWriter.writeDocWithPredAnnotations(docPredWikiWriter.get, jointDevDoc.rawDoc, nerChunks, predClusteringBound, Some(wikiChunks));
      ConllDocWriter.writeDocWithPredAnnotationsWikiStandoff(docPredWriter, docPredWikiWriter, jointDevDoc.rawDoc, nerChunks, predClusteringBound, wikiChunks);
      docPredWriter.close();
      docPredWikiWriter.close();
    }
    outWriter.close();
    outWikiWriter.close();
    if (Driver.writeNerOutput) {
      NEEvaluator.writeIllinoisNEROutput(Execution.getFile("ner.txt"), jointTestDocs.flatMap(_.rawDoc.words), allPredNEChunks.flatten);
    }
    
    //maybeWikipediaInterface.
    
    Logger.endTrack();
    Logger.logss("MENTION DETECTION")
    CorefDoc.displayMentionPRF1(jointTestDocs.map(_.docGraph.corefDoc));
    Logger.logss("COREF");
    Logger.logss(CorefEvaluator.evaluateAndRender(jointTestDocs.map(_.docGraph), allPredBackptrsAndClusterings.map(_._1), allPredBackptrsAndClusterings.map(_._2),
                                                  Driver.conllEvalScriptPath, "DEV: ", Driver.analysesToPrint));
    Logger.logss("NER");
    NEEvaluator.evaluateChunksBySent(jointTestDocs.flatMap(_.goldChunks), allPredNEChunks.flatten);
    Logger.logss("WIKIFICATION");
    WikificationEvaluator.evaluateWikiChunksBySent(jointTestDocs.flatMap(_.goldWikiChunks), allPredWikiChunks);
    WikificationEvaluator.evaluateBOTF1(jointTestDocs.map(doc => WikificationEvaluator.convertSeqChunksToBagOfTitles(doc.goldWikiChunks)), allPredWikiTitles);
    WikificationEvaluator.evaluateFahrniMetrics(jointTestDocs.flatMap(_.goldWikiChunks), allPredWikiChunks, wikiLabelsInTrain)
    if (predWriter.isDefined) predWriter.get.close
    if (goldWriter.isDefined) goldWriter.get.close
    if (predWikiWriter.isDefined) {
      predWikiWriter.get.close
      predWikiWriterAux.get.close
    }
    
    //histg.printHistgram();
  }
  
  def decode(jointTestDoc: JointDocACE, maybeWikipediaInterface: Option[WikipediaInterface]) = {
    val training = false;
    val fgf = new PrunedGraphFactoryACE(jointFeaturizer, maybeWikipediaInterface, domainModel, training);
    val computer = new JointComputerShared(fgf);
    computer.viterbiDecodeProduceAnnotations(jointTestDoc, weights);
  }
/*
  def getOracleResult(jointTestDoc: JointDocACE, maybeWikipediaInterface: Option[WikipediaInterface]) = {
    val fgfOnto = new FactorGraphFactoryACE(jointFeaturizer, maybeWikipediaInterface);
    val myOracle = new JointOracle(fgfOnto);
    myOracle.runILPinference(jointTestDoc, weights); 
  }
*/
  def pack: JointPrunedPredictorACE = {
    if (jointFeaturizer.canReplaceIndexer) {
      val (newIndexer, newWeights) = GUtil.packFeaturesAndWeights(jointFeaturizer.indexer, weights);
      new JointPrunedPredictorACE(jointFeaturizer.replaceIndexer(newIndexer), newWeights, corefPruner, domainModel);
    } else {
      this;
    }
  }
}

object JointPrunedPredictorACE {
  
  // N.B. Doubles with EntitySystem.preprocessDocs
  // Differences from above: no NER pruning, allows for gold mentions
  def preprocessDocs(path: String, size: Int, suffix: String, mentionPropertyComputer: MentionPropertyComputer, corefPruner: CorefPruner) = {
    val rawDocs = ConllDocReader.loadRawConllDocsWithSuffix(path, size, suffix, Language.ENGLISH);
    val assembler = CorefDocAssembler(Driver.lang, Driver.useGoldMentions);
    val corefDocs = rawDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    val docGraphs = corefDocs.map(new DocumentGraph(_, false));
    CorefSystem.preprocessDocsCacheResources(docGraphs);
    // Prune coref now that we have mentions
    corefPruner.pruneAll(docGraphs);
    // No NER pruning
    JointDocACE.assembleJointDocs(docGraphs, new CorpusWikiAnnots);
  }
  
}