package berkeleyentity.joint

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
import scala.collection.mutable.HashSet
import berkeleyentity.ilp.HistgramRecord

import java.io.PrintWriter;
import java.util.ArrayList;

@SerialVersionUID(1L)
class JointPredictorACE(val jointFeaturizer: JointFeaturizerShared[MCNerFeaturizer],
                        val weights: Array[Float],
                        val corefPruner: CorefPruner) extends Serializable {
  
  def decodeWriteOutput(jointTestDocs: Seq[JointDocACE], maybeWikipediaInterface: Option[WikipediaInterface], doConllPostprocessing: Boolean) {
    val fgfOnto = new FactorGraphFactoryACE(jointFeaturizer, maybeWikipediaInterface);
    val computer = new JointComputerShared(fgfOnto);
    val outWriter = IOUtils.openOutHard(Execution.getFile("output.conll"))
    val outWikiWriter = IOUtils.openOutHard(Execution.getFile("output-wiki.conll"))
    Logger.startTrack("Decoding");
    for (i <- (0 until jointTestDocs.size)) {
      Logger.logss("Decoding " + i);
      val (backptrs, clustering, nerChunks, wikiChunks) = computer.viterbiDecodeProduceAnnotations(jointTestDocs(i), weights);
      ConllDocWriter.writeDocWithPredAnnotationsWikiStandoff(outWriter, outWikiWriter, jointTestDocs(i).rawDoc, nerChunks, clustering.bind(jointTestDocs(i).docGraph.getMentions, doConllPostprocessing), wikiChunks);
    }
    outWriter.close();
    outWikiWriter.close();
    Logger.endTrack();
  }
  
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
      Logger.logss("Decoding " + i);
      val jointDevDoc = jointTestDocs(i);
      
      val outDir = Driver.myoutputPath;//"/scratch/EntityLinking2015/berkeley-entity-master/data/ace05/myoutput";
      val outWikiDir = Driver.myoutputPath;//"/scratch/EntityLinking2015/berkeley-entity-master/data/ace05/myoutput_wiki";
      val docPredPath = outDir + "/" + jointDevDoc.rawDoc.docID + ".mypred.conll";
      val docPredWikiPath = outWikiDir + "/" + jointDevDoc.rawDoc.docID + "-wiki.mypred.conll";
      val docPredWriter = IOUtils.openOutHard(docPredPath);
      val docPredWikiWriter = IOUtils.openOutHard(docPredWikiPath);
      
//      val (backptrs, clustering, nerChunks, wikiChunks) = computer.viterbiDecodeProduceAnnotations(jointDevDoc, finalWeights);
      val (backptrs, clustering, nerChunks, wikiChunks) = decode(jointDevDoc, maybeWikipediaInterface);
      //val (backptrs, clustering, nerChunks, wikiChunks) = decodeILP(jointDevDoc, maybeWikipediaInterface, histg);
      //val (backptrs, clustering, nerChunks, wikiChunks) = getOracleResult(jointDevDoc, maybeWikipediaInterface);
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
      //println("DocName: " + jointDevDoc.rawDoc.docID);
      //WikificationEvaluator.evaluateFahrniMetrics(jointDevDoc.goldWikiChunks, wikiChunks, wikiLabelsInTrain);
      // for (chnk <- jointDevDoc.goldWikiChunks.flatten) {
      //  println(chnk.start + ", " + chnk.end + ", " + chnk.label);
      //}
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
    
    histg.printHistgram();
  }
  
  def decode(jointTestDoc: JointDocACE, maybeWikipediaInterface: Option[WikipediaInterface]) = {
    val fgfOnto = new FactorGraphFactoryACE(jointFeaturizer, maybeWikipediaInterface);
    val computer = new JointComputerShared(fgfOnto);
    computer.viterbiDecodeProduceAnnotations(jointTestDoc, weights);
  }
  
/*
  def decodeILP(jointTestDoc: JointDocACE, maybeWikipediaInterface: Option[WikipediaInterface], histg: HistgramRecord) = {
    val fgf = new FactorGraphFactoryACE(jointFeaturizer, maybeWikipediaInterface);
    val ilpcmptr = new JointIlpComputerAce(fgf);
    ilpcmptr.ilpDecodeProduceAnnotations(jointTestDoc, weights);
    //ilpcmptr.viterbiDecodeProduceAnnotations(jointTestDoc, weights, histg);
  }

  def getOracleResult(jointTestDoc: JointDocACE, maybeWikipediaInterface: Option[WikipediaInterface]) = {
    val fgfOnto = new FactorGraphFactoryACE(jointFeaturizer, maybeWikipediaInterface);
    val myOracle = new JointOracle(fgfOnto);
    myOracle.runILPinference(jointTestDoc, weights);
  }
*/  
  def pack: JointPredictorACE = {
    if (jointFeaturizer.canReplaceIndexer) {
      val (newIndexer, newWeights) = GUtil.packFeaturesAndWeights(jointFeaturizer.indexer, weights);
      new JointPredictorACE(jointFeaturizer.replaceIndexer(newIndexer), newWeights, corefPruner);
    } else {
      this;
    }
  }
/*
  def showWeightValues() {
    val printer = new PrintWriter("weights.log");
    val idxer = jointFeaturizer.indexer;
    for (i <- 0 until idxer.size()) {
      val featContent = idxer.get(i);
      val wval = weights(i);
      printer.println(i + " " + featContent + " " + wval);
    }
    printer.close();
  }
  def showJointFeatWeightValues() {

    val jointFeatSet = new ArrayList[String]();
    
    ///////// Coref + NER
    jointFeatSet.add("TagPair=");

    jointFeatSet.add("PrevHeadCurrSC=");
    jointFeatSet.add("PrevHeadCurrSC=");
    jointFeatSet.add("PrevFirstCurrSC=");
    jointFeatSet.add("PrevPrecedingCurrSC=");
    jointFeatSet.add("PrevFollowingCurrSC=");
    jointFeatSet.add("PrevHeadBrownCurrSC=");
    jointFeatSet.add("PrevFirstBrownCurrSC=");
    jointFeatSet.add("PrevPrecedingBrownCurrSC=");
    jointFeatSet.add("PrevFollowingBrownCurrSC=");

    jointFeatSet.add("CurrHeadPrevSC=");
    jointFeatSet.add("CurrHeadPrevSC=");
    jointFeatSet.add("CurrFirstPrevSC=");
    jointFeatSet.add("CurrPrecedingPrevSC=");
    jointFeatSet.add("CurrFollowingPrevSC=");
    jointFeatSet.add("CurrHeadBrownPrevSC=");
    jointFeatSet.add("CurrFirstBrownPrevSC=");

    jointFeatSet.add("CurrPrecedingBrownPrevSC=");
    jointFeatSet.add("CurrFollowingBrownPrevSC=");
    jointFeatSet.add("ThisHeadContainedAndTypes=");
    jointFeatSet.add("AntHeadContainedAndTypes=");

    /////////////////////// NER + Wiki
/*
    jointFeatSet.add("SemTypeAndCategory=");
    jointFeatSet.add("SemTypeAndInfobox=");
    jointFeatSet.add("SemTypeAndInfoboxHead=");
    jointFeatSet.add("SemTypeAndNil=");
    jointFeatSet.add("HeadAndNil=");
*/
    ////////////////////// Coref + wiki
/*
    jointFeatSet.add("SameWikTitle=");
    jointFeatSet.add("ShareOutLink=");
    jointFeatSet.add("LinkToEachOther=");
    jointFeatSet.add("CurrentAntNil=");
    jointFeatSet.add("STCurrentAntNil=");
    jointFeatSet.add("STSameWikiTitle=");
    */
    val printer = new PrintWriter("weights_joint.log");
    
    val idxer = jointFeaturizer.indexer;
    for (i <- 0 until idxer.size()) {
      val featContent = idxer.get(i);
      val wval = weights(i);
      for (j <- 0 until jointFeatSet.size()) {
        val jfnm : String = jointFeatSet.get(j);
        if (featContent.contains(jfnm)) {
          printer.println(i + " " + featContent + " " + wval);
        }
      }
      
    }
    printer.close();
  }
*/
}

object JointPredictorACE {
  
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