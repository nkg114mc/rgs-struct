package berkeleyentity.oregonstate


import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.ObjectInputStream
import java.io.ObjectOutputStream
import scala.Array.canBuildFrom
import scala.collection.JavaConverters.asScalaBufferConverter
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import berkeleyentity.Chunk
import berkeleyentity.Driver.WikifierType
import berkeleyentity.joint.FactorGraphFactoryACE
import berkeleyentity.joint.FactorGraphFactoryOnto
import berkeleyentity.joint.GeneralTrainer
import berkeleyentity.joint.JointComputerShared
import berkeleyentity.joint.JointDoc
import berkeleyentity.joint.JointDocACE
import berkeleyentity.joint.JointFeaturizerShared
import berkeleyentity.joint.JointLossFcns
import berkeleyentity.wiki.WikificationEvaluator
import berkeleyentity.wiki.WikipediaInterface
import berkeleyentity.lang.Language
import berkeleyentity.ner.MCNerFeaturizer
import berkeleyentity.ner.NEEvaluator
import berkeleyentity.ner.NerFeaturizer
import berkeleyentity.ner.NerSystemLabeled
import berkeleyentity.sem.BasicWordNetSemClasser
import berkeleyentity.sem.QueryCountsBundle
import berkeleyentity.sem.SemClasser
import berkeleyentity.wiki._
import edu.berkeley.nlp.futile.fig.basic.IOUtils
import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.futile.util.Logger
import berkeleyentity.xdistrib.CorefComputerDistrib
import berkeleyentity.xdistrib.ComponentFeaturizer
import berkeleyentity.xdistrib.DocumentGraphComponents
import edu.berkeley.nlp.futile.fig.exec.Execution
import berkeleyentity.Driver
import berkeleyentity.GUtil
import berkeleyentity.ConllDoc
import berkeleyentity.WordNetInterfacer
import berkeleyentity.ConllDocWriter
import berkeleyentity.ConllDocReader
import berkeleyentity.sem.BrownClusterInterface
import berkeleyentity.ner.NerPrunerFromMarginals
import berkeleyentity.ner.NerPruner
import berkeleyentity.coref._
import berkeleyentity.joint.JointPredictor
import java.util.ArrayList
import java.io.PrintWriter
import java.nio.file.Files
import java.nio.file.CopyOption._
import java.io.File
import scala.util.control.Breaks._
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
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
import berkeleyentity.coref.CorefEvaluator
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
import berkeleyentity.wiki.WikiAnnotReaderWriter
import berkeleyentity.coref.OrderedClusteringBound
import berkeleyentity.coref.PairwiseIndexingFeaturizerJoint
import berkeleyentity.ner.NerSystemLabeled
import berkeleyentity.coref.PairwiseIndexingFeaturizer
import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.futile.fig.basic.IOUtils
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
import berkeleyentity.ner.MCNerExample
import berkeleyentity.Driver
import berkeleyentity.wiki.ACETester
import berkeleyentity.wiki.WikipediaAuxDB
import berkeleyentity.wiki.WikipediaCategoryDB
import berkeleyentity.wiki.WikipediaInterface
import berkeleyentity.wiki.WikipediaLinkDB
import berkeleyentity.wiki.WikipediaRedirectsDB
import berkeleyentity.wiki.WikipediaTitleGivenSurfaceDB
import berkeleyentity.wiki.JointQueryDenotationChoiceComputer
import berkeleyentity.wiki.JointQueryDenotationChooser
import berkeleyentity.wiki.JointQueryDenotationExample
import berkeleyentity.wiki.Query
import berkeleyentity.wiki.WikiAnnotReaderWriter
import berkeleyentity.wiki.WikificationEvaluator
import berkeleyentity.wiki.CorpusWikiAnnots
import berkeleyentity.wiki.DocWikiAnnots
import berkeleyentity.Driver
import berkeleyentity.lang.Language
import edu.berkeley.nlp.futile.LightRunner
import berkeleyentity.coref.CorefDocAssembler
import berkeleyentity.ConllDocReader
import berkeleyentity.coref.MentionPropertyComputer
import berkeleyentity.GUtil
import edu.berkeley.nlp.futile.fig.basic.Indexer
import berkeleyentity.joint.LikelihoodAndGradientComputer
import scala.collection.mutable.ArrayBuffer
import berkeleyentity.coref.CorefDoc
import edu.berkeley.nlp.futile.math.SloppyMath
import edu.berkeley.nlp.futile.util.Logger
import berkeleyentity.Chunk
import berkeleyentity.joint.GeneralTrainer
import berkeleyentity.coref.NumberGenderComputer
import berkeleyentity.coref.Mention
import berkeleyentity.ilp.SingleDecision
import berkeleyentity.ilp.HistgramRecord
import java.util.Arrays
import edu.berkeley.nlp.futile.util.Counter
import scala.collection.JavaConverters._



///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////


class AceMultiTaskExample(val corefOutput: AceSingleTaskStructExample[Int],
                          val nerOutput: AceSingleTaskStructExample[String],
                          val wikiOutput: AceSingleTaskStructExample[QueryWikiValue],
                          val docGraph: DocumentGraph) extends StructureOutput {
  
  val corefOutputArrStart = 0;
  val corefOutputArrEnd = corefOutputArrStart + corefOutput.variables.size;
  val nerOutputArrStart = corefOutputArrEnd;
  val nerOutputArrEnd = nerOutputArrStart + nerOutput.variables.size;
  val wikiOutputArrStart = nerOutputArrEnd;
  val wikiOutputArrEnd = wikiOutputArrStart + wikiOutput.variables.size;
  
  val numMentions = corefOutput.variables.size;
  val corefNerFactors = Array.ofDim[TernaryTaskFactor[Int, String, String]](numMentions, numMentions);
  val corefWikiFactors = Array.ofDim[TernaryTaskFactor[Int, QueryWikiValue, QueryWikiValue]](numMentions, numMentions);
  val nerWikiFactors = Array.ofDim[BinaryTaskFactor[String, QueryWikiValue]](numMentions, numMentions);
  
  val totalSize = (corefOutput.variables.size +  nerOutput.variables.size + wikiOutput.variables.size);
  
  def getSubArr(start: Int, endPlusOne: Int, arr: Array[Int]): Array[Int] = {
    val retArr = new Array[Int](endPlusOne - start);
        if (arr == null) {
      throw new RuntimeException("null????");
    }
    Array.copy(arr, start, retArr, 0, endPlusOne - start);

    retArr;
  }
  
  override def isCorrectOutput(output: Array[Int]): Boolean = {
    val corefOut = getSubArr(corefOutputArrStart, corefOutputArrEnd, output);
    val nerOut = getSubArr(nerOutputArrStart, nerOutputArrEnd, output);
    val wikiOut = getSubArr(wikiOutputArrStart, wikiOutputArrEnd, output);

    val corefCrrt = corefOutput.isCorrectOutput(corefOut);
    val nerCrrt = nerOutput.isCorrectOutput(nerOut);
    val wikiCrrt = wikiOutput.isCorrectOutput(wikiOut);
    
    (corefCrrt && nerCrrt && wikiCrrt);
  }
  
  override def infereceIndepBest(wght: Array[Double]): Array[Int] = {
    val resultCoref = corefOutput.infereceIndepBest(wght);
    val resultNer = nerOutput.infereceIndepBest(wght);
    val resultWiki = wikiOutput.infereceIndepBest(wght);
    val result = (resultCoref ++ resultNer ++ resultWiki);
    result;
  }
  
  override def infereceIndepGoldBest(wght: Array[Double]): Array[Int] = {
    val resultCoref = corefOutput.infereceIndepGoldBest(wght);
    val resultNer = nerOutput.infereceIndepGoldBest(wght);
    val resultWiki = wikiOutput.infereceIndepGoldBest(wght);
    val result = (resultCoref ++ resultNer ++ resultWiki);
    result;
  }
  
  override def featurize(output: Array[Int]): HashMap[Int, Double] = {
    
    val featMap = new HashMap[Int,Double]();
    
    val corefOut = getSubArr(corefOutputArrStart, corefOutputArrEnd, output);
    val corefFeatVec = corefOutput.featurize(corefOut);
    
    val nerOut = getSubArr(nerOutputArrStart, nerOutputArrEnd, output);
    val nerFeatVec = nerOutput.featurize(nerOut);
    
    val wikiOut = getSubArr(wikiOutputArrStart, wikiOutputArrEnd, output);
    val wikiFeatVec = wikiOutput.featurize(wikiOut);
    
    addSparseFeature(featMap, corefFeatVec);
    addSparseFeature(featMap, nerFeatVec);
    addSparseFeature(featMap, wikiFeatVec);
    
    featMap;
  }
  
  private def inRange(idx: Int, startIdx: Int, endIdx: Int) = {
	  (idx >= startIdx && idx < endIdx);
  }
  def getVariableGivenIndex(idx: Int) = {
	  if (inRange(idx, corefOutputArrStart, corefOutputArrEnd)) {
		  corefOutput.variables(idx - corefOutputArrStart);
	  } else if (inRange(idx, nerOutputArrStart, nerOutputArrEnd)) {
		  nerOutput.variables(idx - nerOutputArrStart);
	  } else if (inRange(idx, wikiOutputArrStart, wikiOutputArrEnd)) {
		  wikiOutput.variables(idx - wikiOutputArrStart);
	  } else {
		  throw new RuntimeException("No such structure slot for index:" + idx);
	  }
  }
  
  // merge the HashMap that represents two sparse vectors
  def addSparseFeature(retv: HashMap[Int, Double], incv: HashMap[Int, Double]): HashMap[Int, Double] = {
    for ((k, v) <- incv) {
      addHMentry(retv, k, v);
    }
    retv;
  }
  
  def addHMentry(myMap: HashMap[Int,Double], index: Int, value: Double) {
    val maybeVal = myMap.get(index);
    if (maybeVal == None) {
      myMap += (index -> value);
    } else {
      val newV = maybeVal.get + value;
      myMap += (index -> newV);
    }
  }
  
  def getZeroOneError(output: Array[Int]) = {
    val corefOut = getSubArr(corefOutputArrStart, corefOutputArrEnd, output);
    val nerOut = getSubArr(nerOutputArrStart, nerOutputArrEnd, output);
    val wikiOut = getSubArr(wikiOutputArrStart, wikiOutputArrEnd, output);
    
    var err1 = 0;
    var err2 = 0;
    var err3 = 0;
    for (i <- 0 until corefOutput.variables.size) {
      if (!corefOutput.variables(i).values(corefOut(i)).isCorrect) err1 += 1;
    }
    for (i <- 0 until nerOutput.variables.size) {
      if (!nerOutput.variables(i).values(nerOut(i)).isCorrect) err2 += 1;
    }
    for (i <- 0 until wikiOutput.variables.size) {
      if (!wikiOutput.variables(i).values(wikiOut(i)).isCorrect) err3 += 1;
    }
    var err = err1 + err2 + err3;
    var total = corefOutput.variables.size  + nerOutput.variables.size + wikiOutput.variables.size;
    //println("Error: [" + err1 + "," + err2 + "," + err3 + "] = " + err + "/" + total);
    err;
  }
  
  def getZeroOneErrorEachTask(output: Array[Int], which: Int) = {
    val corefOut = getSubArr(corefOutputArrStart, corefOutputArrEnd, output);
    val nerOut = getSubArr(nerOutputArrStart, nerOutputArrEnd, output);
    val wikiOut = getSubArr(wikiOutputArrStart, wikiOutputArrEnd, output);
    
    var err1 = 0;
    var err2 = 0;
    var err3 = 0;
    for (i <- 0 until corefOutput.variables.size) {
      if (!corefOutput.variables(i).values(corefOut(i)).isCorrect) err1 += 1;
    }
    for (i <- 0 until nerOutput.variables.size) {
      if (!nerOutput.variables(i).values(nerOut(i)).isCorrect) err2 += 1;
    }
    for (i <- 0 until wikiOutput.variables.size) {
      if (!wikiOutput.variables(i).values(wikiOut(i)).isCorrect) err3 += 1;
    }
    
    (err1, err2, err3);
    //var err = err1 + err2 + err3;
    //var total = corefOutput.variables.size  + nerOutput.variables.size + wikiOutput.variables.size;
    //println("Error: [" + err1 + "," + err2 + "," + err3 + "] = " + err + "/" + total);
    //err;
  }
  
  
}




class AceSingleTaskStructExample[ValType](val variables: Array[IndepVariable[ValType]]) extends StructureOutput {

	// actual structure output?
	//var currentValues = new Array[IndepVariable[ValType]](variables.length);
	//var currentValueIndices = new Array[Int](variables.length);
  //override var currentOutput = new Array[Int](variables.length);
  
  def getVarNumber() = {
    variables.length;
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

	// construct a feature vector that sum over feature vectors on variable values
  def getSparseFeatureVector(output: Array[Int]): HashMap[Int,Double] = { // get value from labelAssignment

		  if (output.length != variables.length) {
			  throw new RuntimeException("Wrong length for output: " + output.length);
		  }

		  val valueMap = new HashMap[Int,Double]();
		  for (i <- 0 until output.length) {
			  val valueIdx = output(i);
			  if (valueIdx >= 0) { // not a invalid value index
				  val feats = variables(i).values(valueIdx).feature;
				  for (idx <- feats) {
					  addValueToVector(valueMap, idx, 1.0);
				  }
			  }
		  }

		  valueMap; // return a sparse feature vector
  }


	override def featurize(output: Array[Int]): HashMap[Int, Double] = {
			getSparseFeatureVector(output);
	}

	override def infereceIndepBest(wght: Array[Double]): Array[Int] = {
    val result = new Array[Int](variables.length);
    for (i <- 0 until variables.length) {
      val bestvIdx = variables(i).getBestValue(wght);
      result(i) = bestvIdx;
    }
    result;
	}
  
	override def infereceIndepGoldBest(wght: Array[Double]): Array[Int] = {
    val result = new Array[Int](variables.length);
    for (i <- 0 until variables.length) {
      val oraclevIdx = variables(i).getCorrectBestValue(wght);
      result(i) = oraclevIdx;
    }
    result;
	}
  
  // return error!
  def getZeroOneError(output: Array[Int]) : Double = {
    var err : Double = 0.0;
    for (i <- 0 until variables.size) {
      if (!variables(i).values(output(i)).isCorrect) err += 1;
    }
    err;
  }
  
  
  override def isCorrectOutput(output: Array[Int]): Boolean = {
		  var isCorrect: Boolean = true;
     breakable {
	   for (i <- 0 until output.length) {
		  if (!variables(i).noCorrect) {
			    if (!variables(i).values(output(i)).isCorrect) {
				    isCorrect = false;
				    break;
			    }
		    }
	    }
    }
    isCorrect;
  }

  def outputCorrectCnt(output: Array[Int]): Int = {
		  var wrongCnt = 0;
		  for (i <- 0 until output.length) {
			  if (!variables(i).noCorrect) {
				  if (!variables(i).values(output(i)).isCorrect) {
					  wrongCnt += 0;
				  }
			  }
		  }
		  wrongCnt;
  }

}
