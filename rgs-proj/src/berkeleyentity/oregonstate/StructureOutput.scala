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


class StructureOutput extends IInstance {
  
  // just store the value indices
  var currentOutput: Array[Int] = null;
  
  def featurize(output: Array[Int]): HashMap[Int,Double] = {
    null;//new HashMap[Int,Double]();
  }

  def infereceIndepBest(wght: Array[Double]): Array[Int] = {
    null;
  }
  
  def infereceIndepGoldBest(wght: Array[Double]): Array[Int] = {
    null;
  }
  
  // do not have be the same as the gold
  // since the gold structure may not be unique 
  def isCorrectOutput(output: Array[Int]): Boolean = {
    false;
  }

}

object StructureOutput {
  
  // return number of different between two output
  def diffCntTwoOutput(out1: Array[Int],
                       out2: Array[Int]) {
    if (out1.length != out2.length) {
      throw new RuntimeException("output length inconsistent: " + out1.length + " != " + out2.length);
    }
    var diffCnt = 0;
    for (i <- 0 until out1.length) {
      if (out1(i) == out2(i)) {
        diffCnt += 1;
      }
    }
    diffCnt;
  }
  
  /*
  def diffCntTwoOutput(out1: Array[Int],
                       out2: Array[Int]) {
    if (out1.length != out2.length) {
      throw new RuntimeException("output length inconsistent: " + out1.length + " != " + out2.length);
    }
    var diffCnt = 0;
    for (i <- 0 until out1.length) {
      if (out1(i) == out2(i)) {
        diffCnt += 1;
      }
    }
    diffCnt;
  }*/
}