package berkeleyentity.joint

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import berkeleyentity.Chunk
import berkeleyentity.ConllDoc
import berkeleyentity.ConllDocReader
import berkeleyentity.ConllDocWriter
import berkeleyentity.GUtil
import berkeleyentity.coref.CorefDocAssembler
import berkeleyentity.coref.CorefDocAssemblerACE
import berkeleyentity.coref.CorefPruner
import berkeleyentity.coref.CorefSystem
import berkeleyentity.coref.DocumentGraph
import berkeleyentity.coref.MentionPropertyComputer
import berkeleyentity.coref.OrderedClustering
import berkeleyentity.lang.Language
import berkeleyentity.ner.NerFeaturizer
import berkeleyentity.ner.NerPruner
import berkeleyentity.wiki.WikipediaInterface
import edu.berkeley.nlp.futile.fig.basic.IOUtils
import edu.berkeley.nlp.futile.fig.exec.Execution
import edu.berkeley.nlp.futile.util.Logger
import berkeleyentity.ner.NEEvaluator
import berkeleyentity.coref.CorefEvaluator
import berkeleyentity.Driver

@SerialVersionUID(1L)
class JointPredictor(val jointFeaturizer: JointFeaturizerShared[NerFeaturizer],
                     val weights: Array[Float],
                     val corefPruner: CorefPruner,
                     val nerPruner: NerPruner) extends Serializable {
  
  def decodeWriteOutput(jointTestDocs: Seq[JointDoc], maybeWikipediaInterface: Option[WikipediaInterface], doConllPostprocessing: Boolean) {
    decodeWriteOutputMaybeEvaluate(jointTestDocs, maybeWikipediaInterface, doConllPostprocessing, false);
  }
  
  def decodeWriteOutputEvaluate(jointTestDocs: Seq[JointDoc], maybeWikipediaInterface: Option[WikipediaInterface], doConllPostprocessing: Boolean) {
    decodeWriteOutputMaybeEvaluate(jointTestDocs, maybeWikipediaInterface, doConllPostprocessing, true);
  }

  private def decodeWriteOutputMaybeEvaluate(jointTestDocs: Seq[JointDoc], maybeWikipediaInterface: Option[WikipediaInterface], doConllPostprocessing: Boolean, evaluate: Boolean) {
    val fgfOnto = new FactorGraphFactoryOnto(jointFeaturizer, maybeWikipediaInterface);
    val computer = new JointComputerShared(fgfOnto);
    val outWriter = IOUtils.openOutHard(Execution.getFile("output.conll"))
    val outWikiWriter = IOUtils.openOutHard(Execution.getFile("output-wiki.conll"))
    val allPredBackptrsAndClusterings = new ArrayBuffer[(Array[Int],OrderedClustering)];
    val predNEChunks = new ArrayBuffer[Seq[Seq[Chunk[String]]]];
    
    println("Weight length = " + weights.length);
    
    Logger.startTrack("Decoding");
    for (i <- (0 until jointTestDocs.size)) {
      Logger.logss("Decoding " + i);
      val jointDevDoc = jointTestDocs(i);
      val (backptrs, clustering, nerChunks, wikiChunks) = computer.viterbiDecodeProduceAnnotations(jointDevDoc, weights);
      ConllDocWriter.writeDocWithPredAnnotationsWikiStandoff(outWriter, outWikiWriter, jointDevDoc.rawDoc, nerChunks, clustering.bind(jointDevDoc.docGraph.getMentions, Driver.doConllPostprocessing), wikiChunks);
      if (evaluate) {
        allPredBackptrsAndClusterings += (backptrs -> clustering);
        predNEChunks += nerChunks;
      }
    }
    outWriter.close();
    outWikiWriter.close();
    Logger.endTrack();
    if (evaluate) {
      Logger.logss(CorefEvaluator.evaluateAndRender(jointTestDocs.map(_.docGraph), allPredBackptrsAndClusterings.map(_._1), allPredBackptrsAndClusterings.map(_._2),
                                                    Driver.conllEvalScriptPath, "DEV: ", Driver.analysesToPrint));
      NEEvaluator.evaluateChunksBySent(jointTestDocs.flatMap(_.goldNERChunks), predNEChunks.flatten);
      NEEvaluator.evaluateOnConll2011(jointTestDocs, predNEChunks, Driver.conll2011Path.split(",").flatMap(path => ConllDocReader.readDocNames(path)).toSet, if (Driver.writeNerOutput) Execution.getFile("ner.txt") else "");
    }
  }
  
  def pack: JointPredictor = {
    if (jointFeaturizer.canReplaceIndexer) {
      val (newIndexer, newWeights) = GUtil.packFeaturesAndWeights(jointFeaturizer.indexer, weights);
      new JointPredictor(jointFeaturizer.replaceIndexer(newIndexer), newWeights, corefPruner, nerPruner);
    } else {
      this;
    }
  }
}
