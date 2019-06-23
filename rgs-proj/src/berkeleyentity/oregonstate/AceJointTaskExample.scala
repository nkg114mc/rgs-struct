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
import berkeleyentity.EntitySystem
import berkeleyentity.joint.MyTrainer
import berkeleyentity.joint.GeneralTrainerCopy
import berkeleyentity.joint.PrunedGraphFactoryACE
import berkeleyentity.prunedomain._
import scala.util.Random


/*
// No joint task factor yet
class SingleTaskStructTesting {

}*/

// factors
class TernaryTaskFactor[A, B, C](val node1: IndepVariable[A], 
                                 val node2: IndepVariable[B], 
                                 val node3: IndepVariable[C],
                                 val feature: Array[Array[Array[Int]]]) {
  var cachedScores = Array.ofDim[Double](node2.domainSize(), node3.domainSize());
  
  def clearCachedScores() {
    for (i <- 0 until cachedScores.size) {
      for (j <- 0 until cachedScores(i).size) {
        cachedScores(i)(j) = Double.NaN;
      }
    }
  }
  
  def computeScore(wght: Array[Double], v2: Int, v3: Int) = {
    SingleTaskStructTesting.computeScore(wght, feature(v2)(v3));
  }
  
  def computeScoreAndCached(wght: Array[Double]) {
    for (i <- 0 until cachedScores.size) {
      for (j <- 0 until cachedScores(i).size) {
        cachedScores(i)(j) = SingleTaskStructTesting.computeScore(wght, feature(i)(j));
      }
    }
  }
  
  def computeScoreAndCachedNonPruned(wght: Array[Double]) {
    for (i <- 0 until cachedScores.size) {
      if (!node2.values(i).isPruned) {
        for (j <- 0 until cachedScores(i).size) {
          if (!node3.values(j).isPruned) {
            cachedScores(i)(j) = SingleTaskStructTesting.computeScore(wght, feature(i)(j));
          }
        }
      }
    }
  }
  
}

// factors
class BinaryTaskFactor[A, B](val node1: IndepVariable[A], 
                             val node2: IndepVariable[B],
                             val feature: Array[Array[Array[Int]]]) {
  var cachedScores = Array.ofDim[Double](node1.domainSize(), node2.domainSize());
  
  def clearCachedScores() {
    for (i <- 0 until cachedScores.size) {
      for (j <- 0 until cachedScores(i).size) {
        cachedScores(i)(j) = Double.NaN;
      }
    }
  }
  
  def computeScore(wght: Array[Double], v1: Int, v2: Int) = {
    SingleTaskStructTesting.computeScore(wght, feature(v1)(v2));
  }
  
  def computeScoreAndCached(wght: Array[Double]) {
	  for (i <- 0 until cachedScores.size) {
		  for (j <- 0 until cachedScores(i).size) {
			  cachedScores(i)(j) = SingleTaskStructTesting.computeScore(wght, feature(i)(j));
		  }
	  }
  }
  
  def computeScoreAndCachedNonPruned(wght: Array[Double]) {
	  for (i <- 0 until cachedScores.size) {
		  if (!node1.values(i).isPruned) {
			  for (j <- 0 until cachedScores(i).size) {
				  if (!node2.values(j).isPruned) {
					  cachedScores(i)(j) = SingleTaskStructTesting.computeScore(wght, feature(i)(j));
				  }
			  }
		  }
	  }
  }
  
}




  // for all three tasks
class AceJointTaskExample(val corefVars: Array[IndepVariable[Int]],
                          val nerVars: Array[IndepVariable[String]],
                          val wikiVars: Array[IndepVariable[QueryWikiValue]],
                          val docGraph: DocumentGraph) extends StructureOutput {
    
    var cachedWeight: Array[Double] = null;
  
    val corefOutputArrStart = 0;
    val corefOutputArrEnd = corefOutputArrStart + corefVars.size;
    val nerOutputArrStart = corefOutputArrEnd;
    val nerOutputArrEnd = nerOutputArrStart + nerVars.size;
    val wikiOutputArrStart = nerOutputArrEnd;
    val wikiOutputArrEnd = wikiOutputArrStart + wikiVars.size;

    val numMentions = corefVars.size;
    val totalSize = (corefVars.size + nerVars.size + wikiVars.size);

    val corefNerFactors = Array.tabulate(numMentions)(i => new Array[TernaryTaskFactor[Int, String, String]](i + 1));//Array.ofDim[TernaryTaskFactor[Int, String, String]](numMentions, numMentions);
    val corefWikiFactors = Array.tabulate(numMentions)(i => new Array[TernaryTaskFactor[Int, QueryWikiValue, QueryWikiValue]](i + 1));//Array.ofDim[TernaryTaskFactor[Int, QueryWikiValue, QueryWikiValue]](numMentions, numMentions);
    val nerWikiFactors = new Array[BinaryTaskFactor[String, QueryWikiValue]](numMentions);//Array.ofDim[BinaryTaskFactor[String, QueryWikiValue]](numMentions, numMentions);

    val corefErrWeight = JointTaskStructTesting.CorefErrorWeight;
    val nerErrWeight = JointTaskStructTesting.NerErrorWeight;
    val wikiErrWeight = JointTaskStructTesting.WikiErrorWeight;
    val errorWeight = initErrorWeight();
    def initErrorWeight() : Array[Double] = {
      val errw = new Array[Double](totalSize);
      for (i1 <- corefOutputArrStart until corefOutputArrEnd) errw(i1) = corefErrWeight;
      for (i2 <- nerOutputArrStart until nerOutputArrEnd) errw(i2) = nerErrWeight;
      for (i3 <- wikiOutputArrStart until wikiOutputArrEnd) errw(i3) = wikiErrWeight;
      errw;
    }
    
    def getSubArr(start: Int, endPlusOne: Int, arr: Array[Int]): Array[Int] = {
        val retArr = new Array[Int](endPlusOne - start);
        if (arr == null) {
          throw new RuntimeException("null????");
        }
        Array.copy(arr, start, retArr, 0, endPlusOne - start);
        retArr;
    }
    
    // 1: coref  2: ner  3: wiki
    def getVariableTaskFromIndex(idx: Int) = {
      if ((idx >= corefOutputArrStart) && (idx < corefOutputArrEnd)) {
        1;
      } else if ((idx >= nerOutputArrStart) && (idx < nerOutputArrEnd)) {
        2;
      } else if ((idx >= wikiOutputArrStart) && (idx < wikiOutputArrEnd)) {
        3;
      } else {
        -1;
      }
    }
    
    def getGlobalStartEndGivenTaskIndex(taskIdx: Int): (Int, Int) = {
      if (taskIdx == 1) {
         return (corefOutputArrStart, corefOutputArrEnd);
      } else if (taskIdx == 2) {
        return (nerOutputArrStart, nerOutputArrEnd);
      } else if (taskIdx == 3) {
        return (wikiOutputArrStart, wikiOutputArrEnd);
      }
      return (-1,-1);
    }
    
    // get coref part of the structure
    def getCorefSubStruct(output: Array[Int]) = {
      val corefOut = getSubArr(corefOutputArrStart, corefOutputArrEnd, output);
      corefOut;
    }
    def getNerSubStruct(output: Array[Int]) = {
      val nerOut = getSubArr(nerOutputArrStart, nerOutputArrEnd, output);
      nerOut;
    }
     def getWikiSubStruct(output: Array[Int]) = {
      val wikiOut = getSubArr(wikiOutputArrStart, wikiOutputArrEnd, output);
      wikiOut;
    }

    /////////////////////////////
    // Structure utils //////////
    /////////////////////////////
    override def isCorrectOutput(output: Array[Int]): Boolean = {
        val corefOut = getSubArr(corefOutputArrStart, corefOutputArrEnd, output);
        val nerOut = getSubArr(nerOutputArrStart, nerOutputArrEnd, output);
        val wikiOut = getSubArr(wikiOutputArrStart, wikiOutputArrEnd, output);

        val corefCrrt = isCorrectOutputSingleTask[Int](corefOut, corefVars);
        val nerCrrt = isCorrectOutputSingleTask[String](nerOut, nerVars);
        val wikiCrrt = isCorrectOutputSingleTask[QueryWikiValue](wikiOut, wikiVars);

        (corefCrrt && nerCrrt && wikiCrrt);
    }
    
    
    // get variable given index
    private def inRange(idx: Int, startIdx: Int, endIdx: Int) = {
      (idx >= startIdx && idx < endIdx);
    }
    def getSingleTaskIndex(idx: Int) = { // covert global index to inside task index from 0 to nMentions
      if (inRange(idx, corefOutputArrStart, corefOutputArrEnd)) {
        (idx - corefOutputArrStart);
      } else if (inRange(idx, nerOutputArrStart, nerOutputArrEnd)) {
        (idx - nerOutputArrStart);
      } else if (inRange(idx, wikiOutputArrStart, wikiOutputArrEnd)) {
        (idx - wikiOutputArrStart);
      } else {
        throw new RuntimeException("No such structure slot for index:" + idx);
      }
    }
    def getCorefGlobalIndex(singleTaskIdx: Int) = { // get a globle task index from single task idx
      (singleTaskIdx + corefOutputArrStart);
    }
    def getNerGlobalIndex(singleTaskIdx: Int) = { // get a globle task index from single task idx
      (singleTaskIdx + nerOutputArrStart);
    }
    def getWikiGlobalIndex(singleTaskIdx: Int) = { // get a globle task index from single task idx
      (singleTaskIdx + wikiOutputArrStart);
    }
    //// given a single
    def getCorefGlobalOutputValue(singleTaskIdx: Int, output: Array[Int]) = { // get a globle task value from a output array
      output(singleTaskIdx + corefOutputArrStart);
    }
    def getNerGlobalOutputValue(singleTaskIdx: Int, output: Array[Int]) = { // get a globle task value from a output array
      output(singleTaskIdx + nerOutputArrStart);
    }
    def getWikiGlobalOutputValue(singleTaskIdx: Int, output: Array[Int]) = { // get a globle task value from a output array
      output(singleTaskIdx + wikiOutputArrStart);
    }

    def getCorefVar(idx: Int) = { // get a coref variable
      corefVars(idx);
    }
    def getNerVar(idx: Int) = { // get a ner variable
      nerVars(idx);
    }
    def getWikiVar(idx: Int) = { // get a wiki variable
      wikiVars(idx);
    }
    def getVariableGivenIndex(idx: Int) = {
      if (inRange(idx, corefOutputArrStart, corefOutputArrEnd)) {
        corefVars(idx - corefOutputArrStart);
      } else if (inRange(idx, nerOutputArrStart, nerOutputArrEnd)) {
        nerVars(idx - nerOutputArrStart);
      } else if (inRange(idx, wikiOutputArrStart, wikiOutputArrEnd)) {
        wikiVars(idx - wikiOutputArrStart);
      } else {
        throw new RuntimeException("No such structure slot for index:" + idx);
      }
    }
    def getVariableDomainSizeGivenIndex(idx: Int) = {
      //val singleTskIdx = getVariableGivenIndex(idx);
      val variable = getVariableGivenIndex(idx);
      variable.values.length;
    }

    // TODO
    override def infereceIndepBest(wght: Array[Double]): Array[Int] = {
      /*
        val resultCoref = corefOutput.infereceIndepBest(wght);
        val resultNer = nerOutput.infereceIndepBest(wght);
        val resultWiki = wikiOutput.infereceIndepBest(wght);
        val result = (resultCoref ++ resultNer ++ resultWiki);*/
        val result = new Array[Int](totalSize);
        result;
    }

    // TODO
    override def infereceIndepGoldBest(wght: Array[Double]): Array[Int] = {
      /*
        val resultCoref = corefOutput.infereceIndepGoldBest(wght);
        val resultNer = nerOutput.infereceIndepGoldBest(wght);
        val resultWiki = wikiOutput.infereceIndepGoldBest(wght);
        val result = (resultCoref ++ resultNer ++ resultWiki);*/
        val result = new Array[Int](totalSize);
        result;
    }

    
    
    def updateValueScoreGivenWeight(wght: Array[Double]) {
    	// unary factors
      
    	for (i <- 0 until numMentions) {
    		corefVars(i).values.map(x => x.computeScoreAndCachedNonPruned(wght))
    		nerVars(i).values.map(x => x.computeScoreAndCachedNonPruned(wght))
    		wikiVars(i).values.map( x => x.computeScoreAndCachedNonPruned(wght))
    	}

    	// joint factors
      
      // ternary factors
      for (i <- 0 until numMentions) {
        val jdomainIdxs = corefVars(i).getAllNonPruningValueIndices();
        for (jvIdx <- jdomainIdxs) {
          val j = corefVars(i).values(jvIdx).value;
          corefNerFactors(i)(j).computeScoreAndCachedNonPruned(wght);
          corefWikiFactors(i)(j).computeScoreAndCachedNonPruned(wght);
        }
      }
      // binary factors
      for (i <- 0 until numMentions) {
    	  nerWikiFactors(i).computeScoreAndCachedNonPruned(wght);
      }
    }
    
    // Won't consider pruned values...
    def updateValueScoreGivenWeightNonPruned(wght: Array[Double]) {
    	// unary factors
      
    	for (i <- 0 until numMentions) {
    		corefVars(i).values.map(x => x.computeScoreAndCachedNonPruned(wght))
    		nerVars(i).values.map(x => x.computeScoreAndCachedNonPruned(wght))
    		wikiVars(i).values.map( x => x.computeScoreAndCachedNonPruned(wght))
    	}

    	// joint factors
      
      // ternary factors
      for (i <- 0 until numMentions) {
        val jdomainIdxs = corefVars(i).getAllNonPruningValueIndices();
        for (jvIdx <- jdomainIdxs) {
          val j = corefVars(i).values(jvIdx).value;
          corefNerFactors(i)(j).computeScoreAndCachedNonPruned(wght);
          corefWikiFactors(i)(j).computeScoreAndCachedNonPruned(wght);
        }
      }
      // binary factors
      for (i <- 0 until numMentions) {
    	  nerWikiFactors(i).computeScoreAndCachedNonPruned(wght);
      }
    }
    
    def clearAllValueScores() {
      // unary factors
      
      
      // joint factors
      
    }
    
    def setCachedWeight(wght: Array[Double]) {
      cachedWeight = wght;
      updateValueScoreGivenWeight(wght);
    } 

    def clearCachedWeight() {
      cachedWeight = null;
      clearAllValueScores();
    }

    def getScoreGivenWeight(output: Array[Int], wght: Array[Double]) : Double = {
    		//val corefOut = getSubArr(corefOutputArrStart, corefOutputArrEnd, output);
    		//val nerOut = getSubArr(nerOutputArrStart, nerOutputArrEnd, output);
    		//val wikiOut = getSubArr(wikiOutputArrStart, wikiOutputArrEnd, output);

    		// unary

      var corefUnaryScore : Double = 0;
      var nerUnaryScore : Double = 0;
      var wikiUnaryScore : Double = 0;
      for (i <- 0 until numMentions) {
      	corefUnaryScore += corefVars(i).values(output(i + corefOutputArrStart)).computeScore(wght);
      	nerUnaryScore += nerVars(i).values(output(i + nerOutputArrStart)).computeScore(wght);
      	wikiUnaryScore+= wikiVars(i).values(output(i + wikiOutputArrStart)).computeScore(wght);
      }

      // ternary & binary

      var corefNerScore : Double = 0;
      var corefWikiScore : Double = 0;
      var nerWikiScore : Double = 0;
      for (i <- 0 until numMentions) {
    	  //val j = output(i + corefOutputArrStart);
    	  val jvalIndex = output(i + corefOutputArrStart);
        val jval = corefVars(i).values(jvalIndex).value;
        //// coref-ner
        val nv1 = output(i + nerOutputArrStart);
        val nv2 = output(jval + nerOutputArrStart);
    	  corefNerScore += corefNerFactors(i)(jval).computeScore(wght, nv1, nv2);
        ////
        val wv1 = output(i + wikiOutputArrStart);
        val wv2 = output(jval + wikiOutputArrStart);
    	  wikiUnaryScore += corefWikiFactors(i)(jval).computeScore(wght, wv1, wv2);
      }

      // binary factors
      for (i <- 0 until numMentions) {
      	val v1 = output(i + nerOutputArrStart);
      	val v2 = output(i + wikiOutputArrStart);
      	nerWikiScore += nerWikiFactors(i).computeScore(wght, v1, v2);
      }

      val totalSc = corefUnaryScore + nerUnaryScore + wikiUnaryScore + corefNerScore + corefWikiScore + nerWikiScore;
      totalSc;
    }
    
    def getScoreFast(output: Array[Int]) : Double = {

      // unary

      var corefUnaryScore : Double = 0;
      var nerUnaryScore : Double = 0;
      var wikiUnaryScore : Double = 0;
      for (i <- 0 until numMentions) {
        corefUnaryScore += corefVars(i).values(output(i + corefOutputArrStart)).cachedScore;
        nerUnaryScore += nerVars(i).values(output(i + nerOutputArrStart)).cachedScore;
        wikiUnaryScore+= wikiVars(i).values(output(i + wikiOutputArrStart)).cachedScore;
      }

      // ternary & binary

      var corefNerScore : Double = 0;
      var corefWikiScore : Double = 0;
      var nerWikiScore : Double = 0;

      for (i <- 0 until numMentions) {
        val jvalIndex = output(i + corefOutputArrStart);
        val jval = corefVars(i).values(jvalIndex).value;
        //// coref-ner
        val nv1 = output(i + nerOutputArrStart);
        val nv2 = output(jval + nerOutputArrStart);
        corefNerScore += corefNerFactors(i)(jval).cachedScores(nv1)(nv2);
        //// coref-wiki
        val wv1 = output(i + wikiOutputArrStart);
        val wv2 = output(jval + wikiOutputArrStart);
        corefWikiScore += corefWikiFactors(i)(jval).cachedScores(wv1)(wv2);
      }

      // binary factors
      //// ner-wiki
      for (i <- 0 until numMentions) {
        val ival = i;
        val v1 = output(ival + nerOutputArrStart);
        val v2 = output(ival + wikiOutputArrStart);
        nerWikiScore += nerWikiFactors(i).cachedScores(v1)(v2);
      }

      val totalSc = corefUnaryScore + nerUnaryScore + wikiUnaryScore + corefNerScore + corefWikiScore + nerWikiScore;
      totalSc;
    }
    
    def getScoreFastIncrementalByAction(output: Array[Int], action: SearchAction, corefRightLink: Array[ArrayBuffer[Int]]) : Double = {
      
      val involvedIndex = action.idx;
      val oldValIdx = action.undoOldValue;
      val newValIdx = action.newValIdx;
      
      val taskType = getVariableTaskFromIndex(involvedIndex);
      

      var unaryDiff : Double = 0;
      var tenaryDiff : Double = 0;
      var binaryDiff : Double = 0;
      
      // 1: coref  2: ner  3: wiki
      if (taskType == 1) { // coref
        
         // unary
        val i = getSingleTaskIndex(involvedIndex);
        unaryDiff = corefVars(i).values(newValIdx).cachedScore - corefVars(i).values(oldValIdx).cachedScore;
        // tenary
        val jvalOld = corefVars(i).values(oldValIdx).value;
        val jvalNew = corefVars(i).values(newValIdx).value;
        // coref-ner
        val nvi = output(i + nerOutputArrStart);
        val nvjOld = output(jvalOld + nerOutputArrStart);
        val nvjNew = output(jvalNew + nerOutputArrStart);
        val cnOld = corefNerFactors(i)(jvalOld).cachedScores(nvi)(nvjOld);
        val cnNew = corefNerFactors(i)(jvalNew).cachedScores(nvi)(nvjNew);
        tenaryDiff += (cnNew - cnOld);
        // coref-wiki
        val wvi = output(i + wikiOutputArrStart);
        val wvjOld = output(jvalOld + wikiOutputArrStart);
        val wvjNew = output(jvalNew + wikiOutputArrStart);
        val cwOld = corefWikiFactors(i)(jvalOld).cachedScores(wvi)(wvjOld);
        val cwNew = corefWikiFactors(i)(jvalNew).cachedScores(wvi)(wvjNew);
        tenaryDiff += (cwNew - cwOld);
        
      } else if (taskType == 2) {// ner
        
        // unary
        val i = getSingleTaskIndex(involvedIndex);
        unaryDiff = nerVars(i).values(newValIdx).cachedScore - nerVars(i).values(oldValIdx).cachedScore;
        // coref-ner
        // left link edge
        val ciIndex = output(i + corefOutputArrStart);
        val jval = corefVars(i).values(ciIndex).value;
        val nvj = output(jval + nerOutputArrStart);
        tenaryDiff += (corefNerFactors(i)(jval).cachedScores(newValIdx)(nvj) - corefNerFactors(i)(jval).cachedScores(oldValIdx)(nvj));
        // right link edge
        val rightNodesToi = corefRightLink(i); 
        for (righti <- rightNodesToi) { // corefVars(i).values(output(righti + nerOutputArrStart)) = i
          val nvRighti = output(righti + nerOutputArrStart);
          tenaryDiff += (corefNerFactors(righti)(i).cachedScores(nvRighti)(newValIdx) - corefNerFactors(righti)(i).cachedScores(nvRighti)(oldValIdx));
        }
        // wiki-ner
        val wvi = output(i + wikiOutputArrStart);
        binaryDiff = nerWikiFactors(i).cachedScores(newValIdx)(wvi) - nerWikiFactors(i).cachedScores(oldValIdx)(wvi);

      } else if (taskType == 3) {// wiki
        
         // unary
        val i = getSingleTaskIndex(involvedIndex);
        unaryDiff = wikiVars(i).values(newValIdx).cachedScore - wikiVars(i).values(oldValIdx).cachedScore;
        // coref-wiki
        
        // left link edge
        val ciIndex = output(i + corefOutputArrStart);
        val jval = corefVars(i).values(ciIndex).value;
        val wvj = output(jval + wikiOutputArrStart);
        tenaryDiff += (corefWikiFactors(i)(jval).cachedScores(newValIdx)(wvj) - corefWikiFactors(i)(jval).cachedScores(oldValIdx)(wvj));
        // right link edge
        val rightNodesToi = corefRightLink(i); 
        for (righti <- rightNodesToi) { // corefVars(i).values(output(righti + nerOutputArrStart)) = i
          val wvRighti = output(righti + wikiOutputArrStart);
          tenaryDiff += (corefWikiFactors(righti)(i).cachedScores(wvRighti)(newValIdx) - corefWikiFactors(righti)(i).cachedScores(wvRighti)(oldValIdx));
        }
        // wiki-ner
        val nvi = output(i + nerOutputArrStart);
        binaryDiff = nerWikiFactors(i).cachedScores(nvi)(newValIdx) - nerWikiFactors(i).cachedScores(nvi)(oldValIdx);
      }
/*
      var corefUnaryScore : Double = 0;
      var nerUnaryScore : Double = 0;
      var wikiUnaryScore : Double = 0;
      for (i <- 0 until numMentions) {
        corefUnaryScore += corefVars(i).values(output(i + corefOutputArrStart)).cachedScore;
        nerUnaryScore += nerVars(i).values(output(i + nerOutputArrStart)).cachedScore;
        wikiUnaryScore+= wikiVars(i).values(output(i + wikiOutputArrStart)).cachedScore;
      }

      // ternary & binary

      var corefNerScore : Double = 0;
      var corefWikiScore : Double = 0;
      var nerWikiScore : Double = 0;

      for (i <- 0 until numMentions) {
        val jvalIndex = output(i + corefOutputArrStart);
        val jval = corefVars(i).values(jvalIndex).value;
        //// coref-ner
        val nv1 = output(i + nerOutputArrStart);
        val nv2 = output(jval + nerOutputArrStart);
        corefNerScore += corefNerFactors(i)(jval).cachedScores(nv1)(nv2);
        //// coref-wiki
        val wv1 = output(i + wikiOutputArrStart);
        val wv2 = output(jval + wikiOutputArrStart);
        corefWikiScore += corefWikiFactors(i)(jval).cachedScores(wv1)(wv2);
      }

      // binary factors
      //// ner-wiki
      for (i <- 0 until numMentions) {
        val ival = i;
        val v1 = output(ival + nerOutputArrStart);
        val v2 = output(ival + wikiOutputArrStart);
        nerWikiScore += nerWikiFactors(i).cachedScores(v1)(v2);
      }
*/
      val totalDiff = unaryDiff + tenaryDiff + binaryDiff;
      totalDiff;
    }
    
    // the inverse edge of each left-link edge
    def computeCorefRightLink(output: Array[Int]): Array[ArrayBuffer[Int]] = {
      val corefOut = getSubArr(corefOutputArrStart, corefOutputArrEnd, output);
      val result = new Array[ArrayBuffer[Int]](corefOut.length);
      for (i <- 0 until result.size) {
        result(i) = new ArrayBuffer[Int]();
      }
      for (i <- 0 until corefOut.length) {
        //val j = ;
        val jval = corefVars(i).values(corefOut(i)).value;
        if (i != jval) { // i should on the right of j
          result(jval) += i;
        }
      }
      result;
    }
    
    def getIncrementalFeatureByAction(output: Array[Int], action: SearchAction, corefRightLink: Array[ArrayBuffer[Int]]) : HashMap[Int,Double] = {
      
      val involvedIndex = action.idx;
      val oldValIdx = action.undoOldValue;
      val newValIdx = action.newValIdx;
      
      val taskType = getVariableTaskFromIndex(involvedIndex);
      

      val featDiff = new HashMap[Int,Double](); // all zero
      //var unaryDiff = new HashMap[Int,Double]();
      //var tenaryDiff = new HashMap[Int,Double]();
      //var binaryDiff = new HashMap[Int,Double]();
      
      // 1: coref  2: ner  3: wiki
      if (taskType == 1) { // coref
        
         // unary
        val i = getSingleTaskIndex(involvedIndex);
        //unaryDiff = corefVars(i).values(newValIdx).feature - corefVars(i).values(oldValIdx).cachedScore;
        addIndexedVector(featDiff, corefVars(i).values(newValIdx).feature);
        subtractIndexedVector(featDiff, corefVars(i).values(oldValIdx).feature);
        // tenary
        val jvalOld = corefVars(i).values(oldValIdx).value;
        val jvalNew = corefVars(i).values(newValIdx).value;
        // coref-ner
        val nvi = output(i + nerOutputArrStart);
        val nvjOld = output(jvalOld + nerOutputArrStart);
        val nvjNew = output(jvalNew + nerOutputArrStart);
        //val cnOld = corefNerFactors(i)(jvalOld).cachedScores(nvi)(nvjOld);
        //val cnNew = corefNerFactors(i)(jvalNew).cachedScores(nvi)(nvjNew);
        //tenaryDiff += (cnNew - cnOld);
        addIndexedVector(featDiff, corefNerFactors(i)(jvalNew).feature(nvi)(nvjNew));
        subtractIndexedVector(featDiff, corefNerFactors(i)(jvalOld).feature(nvi)(nvjOld));
        
        // coref-wiki
        val wvi = output(i + wikiOutputArrStart);
        val wvjOld = output(jvalOld + wikiOutputArrStart);
        val wvjNew = output(jvalNew + wikiOutputArrStart);
        //val cwOld = corefWikiFactors(i)(jvalOld).cachedScores(wvi)(wvjOld);
        //val cwNew = corefWikiFactors(i)(jvalNew).cachedScores(wvi)(wvjNew);
        //tenaryDiff += (cwNew - cwOld);
        addIndexedVector(featDiff, corefWikiFactors(i)(jvalNew).feature(wvi)(wvjNew));
        subtractIndexedVector(featDiff, corefWikiFactors(i)(jvalOld).feature(wvi)(wvjOld));
        
      } else if (taskType == 2) {// ner
        
        // unary
        val i = getSingleTaskIndex(involvedIndex);
        //unaryDiff = nerVars(i).values(newValIdx).cachedScore - nerVars(i).values(oldValIdx).cachedScore;
        addIndexedVector(featDiff, nerVars(i).values(newValIdx).feature);
        subtractIndexedVector(featDiff, nerVars(i).values(oldValIdx).feature);
        // coref-ner
        // left link edge
        val ciIndex = output(i + corefOutputArrStart);
        val jval = corefVars(i).values(ciIndex).value;
        val nvj = output(jval + nerOutputArrStart);
        //tenaryDiff += (corefNerFactors(i)(jval).cachedScores(newValIdx)(nvj) - corefNerFactors(i)(jval).cachedScores(oldValIdx)(nvj));
        addIndexedVector(featDiff, corefNerFactors(i)(jval).feature(newValIdx)(nvj));
        subtractIndexedVector(featDiff, corefNerFactors(i)(jval).feature(oldValIdx)(nvj));
        // right link edge
        val rightNodesToi = corefRightLink(i); 
        for (righti <- rightNodesToi) { // corefVars(i).values(output(righti + nerOutputArrStart)) = i
          val nvRighti = output(righti + nerOutputArrStart);
          //tenaryDiff += (corefNerFactors(righti)(i).cachedScores(nvRighti)(newValIdx) - corefNerFactors(righti)(i).cachedScores(nvRighti)(oldValIdx));
          addIndexedVector(featDiff, corefNerFactors(righti)(i).feature(nvRighti)(newValIdx));
          subtractIndexedVector(featDiff, corefNerFactors(righti)(i).feature(nvRighti)(oldValIdx));
        }
        // wiki-ner
        val wvi = output(i + wikiOutputArrStart);
        //binaryDiff = nerWikiFactors(i).cachedScores(newValIdx)(wvi) - nerWikiFactors(i).cachedScores(oldValIdx)(wvi);
        addIndexedVector(featDiff, nerWikiFactors(i).feature(newValIdx)(wvi));
        subtractIndexedVector(featDiff, nerWikiFactors(i).feature(oldValIdx)(wvi));

      } else if (taskType == 3) {// wiki
        
         // unary
        val i = getSingleTaskIndex(involvedIndex);
        //unaryDiff = wikiVars(i).values(newValIdx).cachedScore - wikiVars(i).values(oldValIdx).cachedScore;
        addIndexedVector(featDiff, wikiVars(i).values(newValIdx).feature);
        subtractIndexedVector(featDiff, wikiVars(i).values(oldValIdx).feature);
        // coref-wiki
        
        // left link edge
        val ciIndex = output(i + corefOutputArrStart);
        val jval = corefVars(i).values(ciIndex).value;
        val wvj = output(jval + wikiOutputArrStart);
        //tenaryDiff += (corefWikiFactors(i)(jval).cachedScores(newValIdx)(wvj) - corefWikiFactors(i)(jval).cachedScores(oldValIdx)(wvj));
        addIndexedVector(featDiff, corefWikiFactors(i)(jval).feature(newValIdx)(wvj));
        subtractIndexedVector(featDiff, corefWikiFactors(i)(jval).feature(oldValIdx)(wvj));
        // right link edge
        val rightNodesToi = corefRightLink(i); 
        for (righti <- rightNodesToi) { // corefVars(i).values(output(righti + nerOutputArrStart)) = i
          val wvRighti = output(righti + wikiOutputArrStart);
          //tenaryDiff += (corefWikiFactors(righti)(i).cachedScores(wvRighti)(newValIdx) - corefWikiFactors(righti)(i).cachedScores(wvRighti)(oldValIdx));
          addIndexedVector(featDiff, corefWikiFactors(righti)(i).feature(wvRighti)(newValIdx));
          subtractIndexedVector(featDiff, corefWikiFactors(righti)(i).feature(wvRighti)(oldValIdx));
        }
        // wiki-ner
        val nvi = output(i + nerOutputArrStart);
        //binaryDiff = nerWikiFactors(i).cachedScores(nvi)(newValIdx) - nerWikiFactors(i).cachedScores(nvi)(oldValIdx);
        addIndexedVector(featDiff, nerWikiFactors(i).feature(nvi)(newValIdx));
        subtractIndexedVector(featDiff, nerWikiFactors(i).feature(nvi)(oldValIdx));
      }

      //val totalDiff = unaryDiff + tenaryDiff + binaryDiff;
      //totalDiff;
      featDiff;
    }
    
    

    override def featurize(output: Array[Int]): HashMap[Int, Double] = {

    		val featMap = new HashMap[Int,Double]();


    		////// Unary factor features ////
    		val corefOut = getSubArr(corefOutputArrStart, corefOutputArrEnd, output);
    		val corefFeatVec = getSparseFeatVecFromVarible[Int](corefOut, corefVars);

    		val nerOut = getSubArr(nerOutputArrStart, nerOutputArrEnd, output);
    		val nerFeatVec = getSparseFeatVecFromVarible[String](nerOut, nerVars);

    		val wikiOut = getSubArr(wikiOutputArrStart, wikiOutputArrEnd, output);
    		val wikiFeatVec = getSparseFeatVecFromVarible[QueryWikiValue](wikiOut, wikiVars);

    		////// Joint (Binary, Ternary) factor features ////
    		val corefNerFeatVec = getSparseFeatVecFromTerFactor[Int, String, String](corefOut, nerOut,nerOut,corefNerFactors);
    		val corefWikiFeatVec = getSparseFeatVecFromTerFactor[Int, QueryWikiValue, QueryWikiValue](corefOut, wikiOut,wikiOut,corefWikiFactors);
    		val wikiNerFeatVec = getSparseFeatVecFromBinFactor[String, QueryWikiValue](nerOut,wikiOut,nerWikiFactors);

    		////////////////////////////////////////
    		////////////////////////////////////////

    		addSparseFeature(featMap, corefFeatVec);
    		addSparseFeature(featMap, nerFeatVec);
    		addSparseFeature(featMap, wikiFeatVec);

    		addSparseFeature(featMap, corefNerFeatVec);
    		addSparseFeature(featMap, corefWikiFeatVec);
    		addSparseFeature(featMap, wikiNerFeatVec);

    		featMap;
    }


    def isCorrectOutputSingleTask[T](output: Array[Int], variables: Array[IndepVariable[T]]): Boolean = {
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

    /////////////////////////
    ///// Feature utils /////
    /////////////////////////
    
    // a indexed vector is an array that contains only index, the corresponding values on the index is 1.0, 
    // all other index that does not included in the array is 0 (sparse feature representation)
    def subtractIndexedVector(myMap: HashMap[Int,Double], indexedVec: Array[Int]) { 
      for (idx <- indexedVec) {
        //subtractValueFromVector(myMap, idx, 1.0);
        subtrValueFromVector(myMap, idx, 1.0);
      }
    }
    
    def subtrValueFromVector(myMap: HashMap[Int,Double], indx: Int, value: Double) { // no checking version
      val maybeVal = myMap.get(indx);
      if (maybeVal == None) {
        myMap += (indx -> -1.0);
      } else {
        myMap(indx) -= 1.0;
        if (myMap(indx) == 0) {
          myMap.remove(indx); // remove zero items
        }
      }
    }
    
    def subtractValueFromVector(myMap: HashMap[Int,Double], indx: Int, value: Double) {
      val maybeVal = myMap.get(indx);
      if (maybeVal == None) {
        //myMap += (index -> 1.0);
        throw new RuntimeException("already zero at index " + indx + ", can not subtract!");
      } else {
        //val newV = maybeVal.get + 1.0;
        //myMap += (index -> newV);
        myMap(indx) -= 1.0;
        if (myMap(indx) == 0) {
          myMap.remove(indx); // remove zero items
        }
      }
    }
    
    def addIndexedVector(valueMap: HashMap[Int,Double], indexedVec: Array[Int]) {
      for (idx <- indexedVec) {
        addValueToVector(valueMap, idx, 1.0);
      }
    }
    
    def addValueToVector(myMap: HashMap[Int,Double], indx: Int, value: Double) {
      val maybeVal = myMap.get(indx);
      if (maybeVal == None) {
        myMap += (indx -> 1.0);
      } else {
        //val newV = maybeVal.get + 1.0;
        //myMap += (index -> newV);
        myMap(indx) += 1.0;
        if (myMap(indx) == 0) {
          myMap.remove(indx); // remove zero items
        }
      }
    }

    // construct a feature vector that sum over feature vectors on variable values
    def getSparseFeatVecFromVarible[T](output: Array[Int], variables: Array[IndepVariable[T]]): HashMap[Int,Double] = { // get value from labelAssignment

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

    // construct a feature vector that sum over feature vectors on all factor combined values
    def getSparseFeatVecFromTerFactor[A,B,C](outputA: Array[Int], // edge
        outputB: Array[Int], // node1
        outputC: Array[Int], // node2
        factorArrs: Array[Array[TernaryTaskFactor[A,B,C]]]): HashMap[Int,Double] = {

        val valueMap = new HashMap[Int,Double]();
        
        for (i <- 0 until factorArrs.length) {
        	val ifactors = factorArrs(i);
        	val j = outputA(i);
        	if (j >= 0 && j != i) { // a invalid value index
        		val factor = ifactors(j); 
        		if (factor != null) {
        			val ivIdx = outputB(i);
        			val jvIdx = outputC(j);
        			val feats = factor.feature(ivIdx)(jvIdx);
        			for (idx <- feats) {
        				addValueToVector(valueMap, idx, 1.0);
        			}
        		}
        	}
        }

        valueMap; // return a sparse feature vector
    }  
    def getSparseFeatVecFromBinFactor[B,C](outputB: Array[Int], 
                                           outputC: Array[Int],
                                           factorArrs: Array[BinaryTaskFactor[B,C]]): HashMap[Int,Double] = {

    		val valueMap = new HashMap[Int,Double]();
    		for (i <- 0 until factorArrs.length) {
    			val factor = factorArrs(i); 
    			if (factor != null) {
    				val ivIdx = outputB(i);
    				val jvIdx = outputC(i);
    				val feats = factor.feature(ivIdx)(jvIdx);
    				for (idx <- feats) {
    					addValueToVector(valueMap, idx, 1.0);
    				}
    			}
    		}

    		valueMap; // return a sparse feature vector
    }

    /////////////////////////////////////////////////////

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
        //val newV = maybeVal.get + value;
        //myMap += (index -> newV);
        myMap(index) += value;
      }
    }
    
    def minusSparseFeature(retv: HashMap[Int, Double], decv: HashMap[Int, Double]): HashMap[Int, Double] = {
        for ((k, v) <- decv) {
          addHMentry(retv, k, -v); // a - b = a + (-b)
        }
        retv;
    }

    
    def getErrFullScore(): Double = {
      totalSize.toDouble;
    }

    def getZeroOneError(output: Array[Int]): Double = {
      val corefOut = getSubArr(corefOutputArrStart, corefOutputArrEnd, output);
      val nerOut = getSubArr(nerOutputArrStart, nerOutputArrEnd, output);
      val wikiOut = getSubArr(wikiOutputArrStart, wikiOutputArrEnd, output);

      var err1 = 0;
      var err2 = 0;
      var err3 = 0;
      for (i <- 0 until corefVars.size) {
        if (!corefVars(i).values(corefOut(i)).isCorrect) err1 += 1;
      }
      for (i <- 0 until nerVars.size) {
        if (!nerVars(i).values(nerOut(i)).isCorrect) err2 += 1;
      }
      for (i <- 0 until wikiVars.size) {
        if (!wikiVars(i).values(wikiOut(i)).isCorrect) err3 += 1;
      }
      var err = (err1 + err2 + err3);
      var total = (corefVars.size  + nerVars.size + wikiVars.size);
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
      for (i <- 0 until corefVars.size) {
        if (!corefVars(i).values(corefOut(i)).isCorrect) err1 += 1;
      }
      for (i <- 0 until nerVars.size) {
        if (!nerVars(i).values(nerOut(i)).isCorrect) err2 += 1;
      }
      for (i <- 0 until wikiVars.size) {
        if (!wikiVars(i).values(wikiOut(i)).isCorrect) err3 += 1;
      }

      (err1, err2, err3);
      //var err = err1 + err2 + err3;
      //var total = corefOutput.variables.size  + nerOutput.variables.size + wikiOutput.variables.size;
      //println("Error: [" + err1 + "," + err2 + "," + err3 + "] = " + err + "/" + total);
      //err;
    }
    
    def getWeightedZeroOneError(output: Array[Int]): Double = {
      val corefOut = getSubArr(corefOutputArrStart, corefOutputArrEnd, output);
      val nerOut = getSubArr(nerOutputArrStart, nerOutputArrEnd, output);
      val wikiOut = getSubArr(wikiOutputArrStart, wikiOutputArrEnd, output);

      var err1: Double = 0;
      var err2: Double = 0;
      var err3: Double = 0;
      for (i <- 0 until corefVars.size) {
        if (!corefVars(i).values(corefOut(i)).isCorrect) err1 += corefErrWeight;
      }
      for (i <- 0 until nerVars.size) {
        if (!nerVars(i).values(nerOut(i)).isCorrect) err2 += nerErrWeight;
      }
      for (i <- 0 until wikiVars.size) {
        if (!wikiVars(i).values(wikiOut(i)).isCorrect) err3 += wikiErrWeight;
      }
      var err: Double = (err1 + err2 + err3);
      //var total = (corefVars.size  + nerVars.size + wikiVars.size);
      //println("Error: [" + err1 + "," + err2 + "," + err3 + "] = " + err + "/" + total);
      err;
    }
    
    
    ////
    def toMultiTaskStructs() = {
    	val corefOutput = new AceSingleTaskStructExample[Int](corefVars);
    	val nerOutput = new AceSingleTaskStructExample[String](nerVars);
    	val wikiOutput = new AceSingleTaskStructExample[QueryWikiValue](wikiVars);
    	val mtex = new AceMultiTaskExample(corefOutput, nerOutput, wikiOutput, docGraph);
    	if (this.currentOutput != null) {
    		val corefOut = getSubArr(corefOutputArrStart, corefOutputArrEnd, this.currentOutput);
    		val nerOut = getSubArr(nerOutputArrStart, nerOutputArrEnd, this.currentOutput);
    		val wikiOut = getSubArr(wikiOutputArrStart, wikiOutputArrEnd, this.currentOutput);

    		mtex.currentOutput = this.currentOutput;
    		mtex.corefOutput.currentOutput = corefOut;
    		mtex.nerOutput.currentOutput = nerOut;
    		mtex.wikiOutput.currentOutput = wikiOut;
    	} else {
    	  throw new RuntimeException("currentOutput is null!");
    	}
    	mtex;
    }
    
    def printTaskOutput() {
      val corefOut = getSubArr(corefOutputArrStart, corefOutputArrEnd, this.currentOutput);
      val nerOut = getSubArr(nerOutputArrStart, nerOutputArrEnd, this.currentOutput);
      val wikiOut = getSubArr(wikiOutputArrStart, wikiOutputArrEnd, this.currentOutput);
      
      println("Coref: ");
      for (i <- 0 until corefOut.length) {
        print(i +":"+corefOut(i) + " ");
      } 
      println();
      println("Ner: ");
      for (i <- 0 until nerOut.length) {
        print(i +":"+nerOut(i) + " ");
      } 
      println();
      println("Wiki: ");
      for (i <- 0 until wikiOut.length) {
        print(i +":"+wikiOut(i) + " ");
      } 
      println();
    }
    
}
