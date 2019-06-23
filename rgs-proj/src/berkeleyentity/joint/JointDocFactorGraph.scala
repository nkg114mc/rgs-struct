package berkeleyentity.joint

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import berkeleyentity.coref.CorefDoc
import berkeleyentity.coref.DocumentGraph
import berkeleyentity.GUtil
import berkeleyentity.coref.PairwiseIndexingFeaturizer
import berkeleyentity.coref.PairwiseIndexingFeaturizerJoint
import berkeleyentity.coref.PairwiseScorer
import berkeleyentity.ner.NerSystemLabeled
import berkeleyentity.sem.SemClass
import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.futile.util.Logger
import berkeleyentity.ner.NerFeaturizer
import berkeleyentity.Chunk
import berkeleyentity.bp.BetterPropertyFactor
import berkeleyentity.bp.Factor
import berkeleyentity.bp.Node
import berkeleyentity.bp.UnaryFactorOld
import scala.Array.canBuildFrom
import berkeleyentity.bp.Domain
import berkeleyentity.bp.UnaryFactorGeneral
import berkeleyentity.ner.NerExample
import berkeleyentity.bp.BinaryFactorGeneral
import berkeleyentity.Driver
import berkeleyentity.bp.ConstantBinaryFactor
import berkeleyentity.bp.SimpleFactorGraph
import scala.collection.mutable.HashMap

trait JointDocFactorGraph {
  
  def setWeights(weights: Array[Float]);
  
  def computeAndStoreMarginals(weights: Array[Float],
                               exponentiateMessages: Boolean,
                               numBpIters: Int);
  
  def computeLogNormalizerApprox: Double;
  
  def scrubMessages();
  
  def printNodeDomains();
  
  def passMessagesFancy(numItrs: Int, exponentiateMessages: Boolean);
  
  def addExpectedFeatureCountsToGradient(scale: Float, gradient: Array[Float]);
  
  def decodeCorefProduceBackpointers: Array[Int];
  
  def decodeNERProduceChunks: Seq[Seq[Chunk[String]]];
  
  def decodeWikificationProduceChunks: Seq[Seq[Chunk[String]]];
  
  def getRepresentativeFeatures: HashMap[String,String];
  

}
