package berkeleyentity.prunedomain

import scala.collection.mutable.ArrayBuffer

import berkeleyentity.Driver
import berkeleyentity.ner.MCNerFeaturizer
import edu.berkeley.nlp.futile.fig.basic.Indexer
import berkeleyentity.sem.SemClasser
import berkeleyentity.sem.BrownClusterInterface
import berkeleyentity.sem.QueryCountsBundle
import berkeleyentity.sem.BasicWordNetSemClasser
import berkeleyentity.coref.FeatureSetSpecification
import berkeleyentity.coref.CorefDoc
import berkeleyentity.coref.LexicalCountsBundle
import berkeleyentity.coref.PairwiseIndexingFeaturizerJoint
import berkeleyentity.coref.PairwiseIndexingFeaturizer


class PrunerFeaturizer(val corefFeaturizer: PairwiseIndexingFeaturizer,
                       val nerFeaturizer: MCNerFeaturizer,
                       val maybeBrownClusters: Option[Map[String,String]],
                       val indexer: Indexer[String]) extends Serializable {

  val BrownLen = Driver.corefNerBrownLength
/*
  def replaceIndexer(newIndexer: Indexer[String]): JointFeaturizerShared[T] = {
    if (!canReplaceIndexer) {
      throw new RuntimeException("Error: can't replace indexer in JointFeaturizerShared with the given type parameters, " +
                                 "so don't try to. Catch and prevent this with canReplaceIndexer");
    }
    val newCorefFeaturizer = if (corefFeaturizer.isInstanceOf[PairwiseIndexingFeaturizerJoint]) {
      corefFeaturizer.asInstanceOf[PairwiseIndexingFeaturizerJoint].replaceIndexer(newIndexer)
    } else {
      throw new RuntimeException("Can't replace for " + corefFeaturizer.getClass());
    }
    val newNerFeaturizer = if (nerFeaturizer.isInstanceOf[MCNerFeaturizer]) {
      nerFeaturizer.asInstanceOf[MCNerFeaturizer].replaceIndexer(newIndexer);
    } else if (nerFeaturizer.isInstanceOf[NerFeaturizer]) {
      nerFeaturizer.asInstanceOf[NerFeaturizer].replaceIndexer(newIndexer);
    } else {
      throw new RuntimeException("Can't replace for " + nerFeaturizer.getClass());
    }
    new JointFeaturizerShared(newCorefFeaturizer, newNerFeaturizer.asInstanceOf[T], maybeBrownClusters, corefNerFeatures, corefWikiFeatures, wikiNerFeatures, newIndexer);
  }
*/
  def indexFeature(feat: String, addToIndexer: Boolean): Int = {
    if (addToIndexer) indexer.getIndex(feat) else indexer.indexOf(feat);
  }
  
  def indexFeatures(feats: Array[String], addToIndexer: Boolean): Array[Int] = {
    if (addToIndexer) {
      feats.map(indexer.getIndex(_));
    } else {
      feats.map(indexer.indexOf(_)).filter(_ != -1);
    }
  }
  
  def maybeAddFeat(indexedFeats: ArrayBuffer[Int], feat: String, addToIndexer: Boolean) {
    if (addToIndexer) {
      indexedFeats += indexer.getIndex(feat)
    } else {
      val idx = indexer.indexOf(feat)
      if (idx != -1) indexedFeats += idx;
    }
  }
  
  def maybeAddFeats(indexedFeats: ArrayBuffer[Int], feats: Seq[String], addToIndexer: Boolean) {
    feats.map(maybeAddFeat(indexedFeats, _, addToIndexer));
  }
  
  private def fetchBrownCluster(word: String): String = fetchBrownCluster2(word, BrownLen);
  
  private def fetchBrownCluster2(word: String, length: Int): String = {
    if (maybeBrownClusters.isDefined && maybeBrownClusters.get.contains(word)) {
      val cluster = maybeBrownClusters.get(word);
      cluster.slice(0, Math.min(cluster.size, length));
    } else {
      ""
    }
  }
  
}

object PrunerFeaturizer {
  
  def constructFeaturizer(trainDocs: Seq[CorefDoc]): PrunerFeaturizer = {
    
    val featureIndexer = new Indexer[String]();
    val maybeBrownClusters = if (Driver.brownPath != "") Some(BrownClusterInterface.loadBrownClusters(Driver.brownPath, 0)) else None
    val nerFeaturizer = MCNerFeaturizer(Driver.nerFeatureSet.split("\\+").toSet, featureIndexer, MCNerFeaturizer.StdLabelIndexer, trainDocs.flatMap(_.rawDoc.words), None, maybeBrownClusters)

    featureIndexer.getIndex(PairwiseIndexingFeaturizerJoint.UnkFeatName);
    val queryCounts: Option[QueryCountsBundle] = None;
    val lexicalCounts = LexicalCountsBundle.countLexicalItems(trainDocs, Driver.lexicalFeatCutoff);
    val semClasser: Option[SemClasser] = Some(new BasicWordNetSemClasser);
    val corefFeatureSetSpec = FeatureSetSpecification(Driver.pairwiseFeats, Driver.conjScheme, Driver.conjFeats, Driver.conjMentionTypes, Driver.conjTemplates);
    val corefFeaturizer = new PairwiseIndexingFeaturizerJoint(featureIndexer, corefFeatureSetSpec, lexicalCounts, queryCounts, semClasser);
    val prunerFeatzr = new PrunerFeaturizer(corefFeaturizer, nerFeaturizer, maybeBrownClusters, featureIndexer);
    prunerFeatzr;
  }
  
}
