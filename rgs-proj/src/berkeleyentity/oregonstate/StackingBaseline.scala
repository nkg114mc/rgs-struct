package berkeleyentity.oregonstate

import berkeleyentity.sem.BrownClusterInterface
import berkeleyentity.coref.MentionPropertyComputer
import berkeleyentity.sem.QueryCountsBundle
import berkeleyentity.coref.NumberGenderComputer
import berkeleyentity.sem.BasicWordNetSemClasser
import berkeleyentity.coref.FeatureSetSpecification
import berkeleyentity.GUtil
import berkeleyentity.wiki.WikipediaInterface
import berkeleyentity.wiki.JointQueryDenotationChoiceComputer
import berkeleyentity.ner.MCNerFeaturizer
import berkeleyentity.oregonstate.pruner.LinearDomainPruner
import berkeleyentity.lang.Language
import berkeleyentity.joint.JointFeaturizerShared
import berkeleyentity.sem.SemClasser
import berkeleyentity.ConllDocReader
import berkeleyentity.coref.CorefDocAssembler
import berkeleyentity.ilp.HistgramRecord
import berkeleyentity.oregonstate.pruner.NonePruner
import berkeleyentity.coref.LexicalCountsBundle
import berkeleyentity.wiki.WikiAnnotReaderWriter
import berkeleyentity.oregonstate.pruner.StaticDomainPruner
import berkeleyentity.coref.CorefFeaturizerTrainer
import berkeleyentity.coref.PairwiseIndexingFeaturizerJoint
import edu.berkeley.nlp.futile.fig.basic.Indexer
import berkeleyentity.Driver
import berkeleyentity.coref.CorefPruner
import berkeleyentity.coref.DocumentGraph
import berkeleyentity.coref.CorefDoc
import berkeleyentity.wiki.DocWikiAnnots
import java.util.Arrays
import java.io.File
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import ml.dmlc.xgboost4j.scala.XGBoost

class StackingValue(val value: Int, 
                    val isCrr: Boolean) {
  
}

class StackingDecision(val taskIndex: Int,
                       val values: Array[StackingValue],
                       val originIndex: Int) {
  
  val idxToVal = initValueMap();
  val goldCount = getCorrectCount();
  
  def initValueMap() = {
    val itov = new HashMap[Int, StackingValue]();
    for (v <- values) {
      itov += (v.value -> v);
    }
    itov;
  }
  
  def checkCorrectness(vIdx: Int): Boolean = {
    val v = idxToVal(vIdx);
    return v.isCrr;
  }
  
  def getCorrectCount() = {
    var crrCnt = 0;
    for (v <- values) {
      if (v.isCrr) crrCnt += 1;
    }
    crrCnt;
  }

}

class StackingStruct(val ex: AceJointTaskExample, 
                     val decisions: Array[StackingDecision],
                     val existOutput: Array[Int],
                     val taskIndex: Int,
                     val doCoref: Boolean, val doNer: Boolean, val doWiki: Boolean) {
  
  def copyArr(src: Array[Int]) = {
    val dest = new Array[Int](src.length);
    Array.copy(src, 0, dest, 0, src.length)
    dest;
  }
  
  def assignOutput(result: Array[Int], output: Array[Int]) {
    for (i <- 0 until decisions.length) {
      output(decisions(i).originIndex) = result(i);
    }
  }
  
  /////////////////////
  
  def sequentialInference(weight: Array[Double]): Array[Int] = {
    val existOutputCopy = copyArr(existOutput);
    val result = new Array[Int](decisions.length);
    for (i <- 0 until decisions.length) {
      result(i) = inferenceBest(decisions(i), weight, existOutputCopy);
      existOutputCopy(decisions(i).originIndex) = result(i);
    }
    return result;
  }

  def sequentialInferenceCorrect(weight: Array[Double]): Array[Int] = {
    val existOutputCopy = copyArr(existOutput);
    val result = new Array[Int](decisions.length);
    for (i <- 0 until decisions.length) {
      result(i) = inferenceCorrectBest(decisions(i), weight, existOutputCopy);
      existOutputCopy(decisions(i).originIndex) = result(i);
    }
    return result;
  }

  /////////////////////
  
    

  def inferenceBest(decision: StackingDecision, weight: Array[Double], existOutput: Array[Int]): Int = {
    var bestIdx = -1;
    var bestScore = -Double.MaxValue;
    for (ivalue <- decision.values) {
      val vfeat = StackingBaseline.featurizeGivenValue(ex, taskIndex, decision.originIndex, ivalue.value, doCoref, doNer, doWiki, existOutput);
      val score = SingleTaskStructTesting.computeScoreSparse(weight, vfeat);
      if (score > bestScore) {
        bestScore = score;
        bestIdx = ivalue.value;
      }
    }
    return bestIdx;
  }
  
  def inferenceCorrectBest(decision: StackingDecision, weight: Array[Double], existOutput: Array[Int]): Int = {
		  var bestLbl = -1;
		  var bestScore = -Double.MaxValue;
		  var bestCorrectLbl = -1; // latent best
		  var bestCorrectScore = -Double.MaxValue;
		  for (ivalue <- decision.values) {
			  val vfeat = StackingBaseline.featurizeGivenValue(ex, taskIndex, decision.originIndex, ivalue.value, doCoref, doNer, doWiki, existOutput);
        val score = SingleTaskStructTesting.computeScoreSparse(weight, vfeat);
			  if (score > bestScore) {
				  bestScore = score;
				  bestLbl = ivalue.value;
			  }
			  if (ivalue.isCrr) {
				  if (score > bestCorrectScore) {
					  bestCorrectScore = score;
					  bestCorrectLbl = ivalue.value;
				  }
			  }
		  }

		  // no correct
		  if (bestCorrectLbl == -1) {
			  bestCorrectLbl = bestLbl;
		  }

		  return bestCorrectLbl;
  }

  def featurize(result: Array[Int]): HashMap[Int,Double] = {
    val existOutputCopy = copyArr(existOutput);
    //val result = new Array[Int](decisions.length);
    for (i <- 0 until decisions.length) {
      existOutputCopy(decisions(i).originIndex) = result(i);
    }
    ////////
    val feat = new HashMap[Int,Double]();
    for (i <- 0 until decisions.length) {
      val vfeat = StackingBaseline.featurizeGivenValue(ex, taskIndex, decisions(i).originIndex, result(i), doCoref, doNer, doWiki, existOutputCopy);
      ex.addSparseFeature(feat, vfeat);
    }
    return feat;
  }
  
  def computeTotalAndCorrect(result: Array[Int]) = {
    var total = 0;
    var crr = 0;
    for (i <- 0 until result.length) {
    	val decision = decisions(i);
    	val bestValue = result(i);
    	////////////
    	total += 1;
    	if (decision.checkCorrectness(bestValue)) {
    		crr += 1;
    	}
    }
    (total, crr);
  }
  
  def isCorrectOutput(result: Array[Int]): Boolean = {
    val (total, crr) = computeTotalAndCorrect(result);
    if (crr == total) {
      return true;
    } else {
      return false;
    }
  }
  
  
}

class StackingModel[NerFeaturizerType](val ordering: Array[Int],
                                       val weight1: Array[Double],
                                       val weight2: Array[Double],
                                       val weight3: Array[Double],
                                       val wikiDB: WikipediaInterface,
                                       val jointFeaturizer: JointFeaturizerShared[NerFeaturizerType]) {
  
}

object StackingBaseline {

  
  def main(args: Array[String]) {

	  // set some configs
	  Driver.numberGenderDataPath = "data/gender.data";
	  Driver.brownPath = "data/bllip-clusters";
	  //Driver.wordNetPath = "";//data/dict";
	  Driver.useGoldMentions = true;
	  Driver.doConllPostprocessing = false;
	  Driver.pruningStrategy = "build:./corefpruner-ace.ser.gz:-5:5";
    Driver.lossFcn = "customLoss-1-1-1";

	  Driver.corefNerFeatures = "indicators+currlex+antlex";
	  Driver.wikiNerFeatures = "categories+infoboxes+appositives";
	  Driver.corefWikiFeatures = "basic+lastnames";

	  trainStackingTasks();
  }

  
  def extractJointExmpGivenMultiExampleNoFeaturize(independentTaskStrucExmp: AceMultiTaskExample) = {
    val docGraph = independentTaskStrucExmp.docGraph;

    val corefVars = independentTaskStrucExmp.corefOutput.variables;
    val nerVars = independentTaskStrucExmp.nerOutput.variables;
    val wikiVars = independentTaskStrucExmp.wikiOutput.variables;
    
    val jointStructExmp = new AceJointTaskExample(corefVars, nerVars, wikiVars, docGraph);
	  jointStructExmp;
  }
  
  def extractJointNoFeature(multiExs: Seq[AceMultiTaskExample]) = {
    val result = ArrayBuffer[AceJointTaskExample]();
    for (ex <- multiExs) {
      result += (extractJointExmpGivenMultiExampleNoFeaturize(ex));
    }
    result;
  }
  
  def featurizeAblationBatch[NerFeaturizerType](exs: Seq[AceJointTaskExample],
                                           wikiDB: WikipediaInterface,
                                           jointFeaturizer: JointFeaturizerShared[NerFeaturizerType],
                                           doCoref: Boolean,
                                           doNer: Boolean,
                                           doWiki: Boolean) {
    for (ex <- exs) {
      featurizeAblation[NerFeaturizerType](ex, wikiDB, jointFeaturizer,  doCoref, doNer, doWiki);
    }
  }
    
  def featurizeAblation[NerFeaturizerType](jointStructExmp: AceJointTaskExample,
                                           wikiDB: WikipediaInterface,
                                           jointFeaturizer: JointFeaturizerShared[NerFeaturizerType],
                                           doCoref: Boolean,
                                           doNer: Boolean,
                                           doWiki: Boolean) {

    
    val docGraph = jointStructExmp.docGraph;

    val corefVars = jointStructExmp.corefVars;
    val nerVars = jointStructExmp.nerVars;
    val wikiVars = jointStructExmp.wikiVars;

    /////////////////////////////////////////
    
    val corefDoc: CorefDoc = docGraph.corefDoc;
    val rawDoc = corefDoc.rawDoc;
    val docName = rawDoc.docID;
    val addToIdxer: Boolean = docGraph.addToFeaturizer;
    
    val doCorefNer = (doCoref && doNer);
    val doCorefWiki = (doCoref && doWiki);
    val doNerWiki = (doNer && doWiki);
    
    // NER+COREF FACTORS
    if (jointFeaturizer.corefNerFeatures != "") {
    	for (i <- 0 until docGraph.size) {
    		val domain = corefVars(i).values;
    		val currNerNode = nerVars(i);
    		for (jval <- domain) {
    			val j = jval.value;
    			if (!jval.isPruned) {
    				if (j != i) {
    					val antNerNode = nerVars(j);
    					val featsIndexed: Array[Array[Array[Int]]] = Array.tabulate(currNerNode.values.size, antNerNode.values.size)((currNerValIdx, antNerValIdx) => {
    						if (doCorefNer) {
    						  jointFeaturizer.getCorefNerFeatures(docGraph, i, j, currNerNode.values(currNerValIdx).value, antNerNode.values(antNerValIdx).value, addToIdxer);
    						} else {
    						  Array();
    						}
    					});
    					//agreementFactors(i)(j) = addAndReturnFactor(new BetterPropertyFactor[String](j, currNerNode, corefNodes(i), antNerNode, featsIndexed), true);
    					jointStructExmp.corefNerFactors(i)(j) = new TernaryTaskFactor[Int, String, String](corefVars(i), currNerNode, antNerNode, featsIndexed);
    				} else {
    					val antNerNode = nerVars(j);
    					val emptyFeat: Array[Array[Array[Int]]] = Array.tabulate(currNerNode.values.size, antNerNode.values.size)((currNerValIdx, antNerValIdx) => Array() );
    					jointStructExmp.corefNerFactors(i)(j) = new TernaryTaskFactor[Int, String, String](corefVars(i), currNerNode, antNerNode, emptyFeat);
    				}
    			}
    		}
    	}
    }

    // COREF+WIKIFICATION FACTORS
    if (jointFeaturizer.corefWikiFeatures != "") {
    	for (i <- 0 until docGraph.size) {
    		val domain = corefVars(i).values;
    		val currWikiNode = wikiVars(i);
    		for (jval <- domain) {
    			val j = jval.value;
    			if (!jval.isPruned) {
    				if (j != i) {
    					val antWikiNode = wikiVars(j);
    					val featsIndexed: Array[Array[Array[Int]]] = Array.tabulate(currWikiNode.values.size, antWikiNode.values.size)((currWikiValIdx, antWikiValIdx) => {
    						if (doCorefWiki) {
    						  jointFeaturizer.getCorefWikiFeatures(docGraph, i, j, currWikiNode.values(currWikiValIdx).value.wiki, antWikiNode.values(antWikiValIdx).value.wiki, Some(wikiDB), addToIdxer);
    						} else {
    						  Array();
    						}
    					});
    					//corefWikiFactors(i)(j) = addAndReturnFactor(new BetterPropertyFactor[String](j, currWikiNode, corefNodes(i), antWikiNode, featsIndexed), true);
    					jointStructExmp.corefWikiFactors(i)(j) = new TernaryTaskFactor[Int, QueryWikiValue, QueryWikiValue](corefVars(i), currWikiNode, antWikiNode, featsIndexed);
    				} else {
    					val antWikiNode = wikiVars(j);
    					val emptyFeat: Array[Array[Array[Int]]] = Array.tabulate(currWikiNode.values.size, antWikiNode.values.size)((currWikiValIdx, antWikiValIdx) => Array() );
    					jointStructExmp.corefWikiFactors(i)(j) = new TernaryTaskFactor[Int, QueryWikiValue, QueryWikiValue](corefVars(i), currWikiNode, antWikiNode, emptyFeat);
    				}
    			}
    		}
    	}
    }

    // NER+WIKIFICATION FACTORS
    if (jointFeaturizer.wikiNerFeatures != "") {
    	for (i <- 0 until docGraph.size) {
    		val nerNode = nerVars(i);
    		val wikiNode = wikiVars(i);
    		val featsIndexed: Array[Array[Array[Int]]] = Array.tabulate(nerNode.values.size, wikiNode.values.size)((nerValIdx, wikiValIdx) => {
    			if (doNerWiki) {
    				jointFeaturizer.getWikiNerFeatures(docGraph, i, wikiNode.values(wikiValIdx).value.wiki, nerNode.values(nerValIdx).value, Some(wikiDB), addToIdxer);
    			} else {
    				Array();
    			}
    		});
    		jointStructExmp.nerWikiFactors(i) = new BinaryTaskFactor[String, QueryWikiValue](nerNode, wikiNode, featsIndexed);
    	}
    }
    
  }
  
  
  val COREF_TASK_INDEX = 1;
  val NER_TASK_INDEX = 2;
  val WIKI_TASK_INDEX = 3;
  
/*
  def trainStackingTasks() {

    val trainDataPath = "data/ace05/train";
    val devDataPath = "data/ace05/dev";
    //val testDataPath = "data/ace05/test";
    val testDataPath = devDataPath;

    //val trainDataPath = "data/ace05/train_1";
    //val testDataPath = "data/ace05/test_1";
    
    val wikiPath = "data/ace05/ace05-all-conll-wiki"
    val wikiDBPath = "models/wiki-db-ace.ser.gz"
    
    val berkeleyCorefPrunerDumpPath = "models:./corefpruner-ace.ser.gz:-5";
    //val berkeleyCorefPrunerDumpPath = "build:./corefpruner-ace.ser.gz:-5:5";
    

    val featIndexer = new Indexer[String]();
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    //val mentTypePredictor = AceMentionTypePredictor.TrainMentionTypeACE();
    //val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer), Some(mentTypePredictor));
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    val assembler = CorefDocAssembler(Language.ENGLISH, true); //use gold mentions
    val trainDocs = ConllDocReader.loadRawConllDocsWithSuffix(trainDataPath, -1, "", Language.ENGLISH);
    val trainCorefDocs = trainDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    
    val maybeBrownClusters = Some(BrownClusterInterface.loadBrownClusters(Driver.brownPath, 0));
    val nerFeaturizer = MCNerFeaturizer(Driver.nerFeatureSet.split("\\+").toSet, featIndexer, MCNerFeaturizer.StdLabelIndexer, trainDocs.flatMap(_.words), None, maybeBrownClusters);


    // load pruner
    val berkPruner = CorefPruner.buildPrunerArguments(berkeleyCorefPrunerDumpPath, trainDataPath, -1);
    
    
    // Read in gold Wikification labels
    val goldWikification = WikiAnnotReaderWriter.readStandoffAnnotsAsCorpusAnnots(wikiPath)
    // Read in the title given surface database
    val wikiDB = GUtil.load(wikiDBPath).asInstanceOf[WikipediaInterface];
    val jqdcomputer = new JointQueryDenotationChoiceComputer(wikiDB, featIndexer);

    featIndexer.getIndex(PairwiseIndexingFeaturizerJoint.UnkFeatName);
    val queryCounts: Option[QueryCountsBundle] = None;
    val lexicalCounts = LexicalCountsBundle.countLexicalItems(trainCorefDocs, Driver.lexicalFeatCutoff);
    val semClasser: Option[SemClasser] = Driver.semClasserType match {
      case "basic" => Some(new BasicWordNetSemClasser);
      case e => throw new RuntimeException("Other semclassers not implemented");
    }
    val trainDocGraphs = trainCorefDocs.map(new DocumentGraph(_, true));

    // run berkeley pruning
    CorefStructUtils.preprocessDocsCacheResources(trainDocGraphs);
    berkPruner.pruneAll(trainDocGraphs);

    

    val featureSetSpec = FeatureSetSpecification(Driver.pairwiseFeats, Driver.conjScheme, Driver.conjFeats, Driver.conjMentionTypes, Driver.conjTemplates);
    val basicFeaturizer = new PairwiseIndexingFeaturizerJoint(featIndexer, featureSetSpec, lexicalCounts, queryCounts, semClasser);
    val featurizerTrainer = new CorefFeaturizerTrainer();

    // joint featurizer
    val jointFeaturier = new JointFeaturizerShared[MCNerFeaturizer](basicFeaturizer, nerFeaturizer, maybeBrownClusters, Driver.corefNerFeatures, Driver.corefWikiFeatures, Driver.wikiNerFeatures, featIndexer);

    /// extract independent task examples
    val trainIndepExs = SingleTaskStructTesting.extractAllTaskExamples(trainDocGraphs, 
                                              nerFeaturizer,
                                              goldWikification, wikiDB, true, jqdcomputer,
                                              basicFeaturizer);
    
    println("FeatureCnt1 = " + featIndexer.size);
    
    // test examples
    val testDocGraphs = SingleTaskStructTesting.testGetdocgraphs(testDataPath, -1, mentionPropertyComputer);
    val testDocs = testDocGraphs.map { dg => dg.corefDoc };
    berkPruner.pruneAll(testDocGraphs);
    
    val testIndepExs = SingleTaskStructTesting.extractAllTaskExamples(testDocGraphs, 
                                           nerFeaturizer,
                                           goldWikification, wikiDB, false, jqdcomputer,
                                           basicFeaturizer);

    println("FeatureCnt1 = " + featIndexer.size);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////


    // extract joint structural examples!
    val trainStructs = extractJointNoFeature(trainIndepExs);
    // test examples
    val testStructs = extractJointNoFeature(testIndepExs);
    println("FeatureCnt1.1 = " + featIndexer.size);
    
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    
    //val orderedTasks = Array[Int](COREF_TASK_INDEX, NER_TASK_INDEX, WIKI_TASK_INDEX);
    //val orderedTasks = Array[Int](COREF_TASK_INDEX, WIKI_TASK_INDEX, NER_TASK_INDEX);
    
    //val orderedTasks = Array[Int](NER_TASK_INDEX, COREF_TASK_INDEX, WIKI_TASK_INDEX);
    //val orderedTasks = Array[Int](NER_TASK_INDEX, WIKI_TASK_INDEX, COREF_TASK_INDEX);
    
    //val orderedTasks = Array[Int](WIKI_TASK_INDEX, NER_TASK_INDEX, COREF_TASK_INDEX);
    val orderedTasks = Array[Int](WIKI_TASK_INDEX, COREF_TASK_INDEX, NER_TASK_INDEX);

    val stackingModel = stackTraining[MCNerFeaturizer](orderedTasks, trainStructs, testStructs, wikiDB, jointFeaturier);
    stackTesting[MCNerFeaturizer](testStructs, stackingModel);
    
    // evaluate
    JointTaskStructTesting.evaluateAceStructsJoint(testStructs, goldWikification);

  }
*/
  
   def trainStackingTasks() {

    val trainDataPath = "data/ace05/train";
    val devDataPath = "data/ace05/dev";
    val testDataPath = "data/ace05/test";
    //val testDataPath = devDataPath;

    //val trainDataPath = "data/ace05/train_1";
    //val testDataPath = "data/ace05/test_1";
    
    val wikiPath = "data/ace05/ace05-all-conll-wiki"
    val wikiDBPath = "models/wiki-db-ace.ser.gz"
    
    val berkeleyCorefPrunerDumpPath = "models:./corefpruner-ace.ser.gz:-5";
    //val berkeleyCorefPrunerDumpPath = "build:./corefpruner-ace.ser.gz:-5:5";
    

    val featIndexer = new Indexer[String]();
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    //val mentTypePredictor = AceMentionTypePredictor.TrainMentionTypeACE();
    //val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer), Some(mentTypePredictor));
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    val assembler = CorefDocAssembler(Language.ENGLISH, true); //use gold mentions
    val trainDocs = ConllDocReader.loadRawConllDocsWithSuffix(trainDataPath, -1, "", Language.ENGLISH);
    val trainCorefDocs = trainDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    
    val maybeBrownClusters = Some(BrownClusterInterface.loadBrownClusters(Driver.brownPath, 0));
    val nerFeaturizer = MCNerFeaturizer(Driver.nerFeatureSet.split("\\+").toSet, featIndexer, MCNerFeaturizer.StdLabelIndexer, trainDocs.flatMap(_.words), None, maybeBrownClusters);


    // load pruner
    val berkPruner = CorefPruner.buildPrunerArguments(berkeleyCorefPrunerDumpPath, trainDataPath, -1);
    
    
    // Read in gold Wikification labels
    val goldWikification = WikiAnnotReaderWriter.readStandoffAnnotsAsCorpusAnnots(wikiPath)
    // Read in the title given surface database
    val wikiDB = GUtil.load(wikiDBPath).asInstanceOf[WikipediaInterface];
    val jqdcomputer = new JointQueryDenotationChoiceComputer(wikiDB, featIndexer);

    featIndexer.getIndex(PairwiseIndexingFeaturizerJoint.UnkFeatName);
    val queryCounts: Option[QueryCountsBundle] = None;
    val lexicalCounts = LexicalCountsBundle.countLexicalItems(trainCorefDocs, Driver.lexicalFeatCutoff);
    val semClasser: Option[SemClasser] = Driver.semClasserType match {
      case "basic" => Some(new BasicWordNetSemClasser);
      case e => throw new RuntimeException("Other semclassers not implemented");
    }
    val trainDocGraphs = trainCorefDocs.map(new DocumentGraph(_, true));

    // run berkeley pruning
    CorefStructUtils.preprocessDocsCacheResources(trainDocGraphs);
    berkPruner.pruneAll(trainDocGraphs);

    

    val featureSetSpec = FeatureSetSpecification(Driver.pairwiseFeats, Driver.conjScheme, Driver.conjFeats, Driver.conjMentionTypes, Driver.conjTemplates);
    val basicFeaturizer = new PairwiseIndexingFeaturizerJoint(featIndexer, featureSetSpec, lexicalCounts, queryCounts, semClasser);
    val featurizerTrainer = new CorefFeaturizerTrainer();

    // joint featurizer
    val jointFeaturier = new JointFeaturizerShared[MCNerFeaturizer](basicFeaturizer, nerFeaturizer, maybeBrownClusters, Driver.corefNerFeatures, Driver.corefWikiFeatures, Driver.wikiNerFeatures, featIndexer);

    /// extract independent task examples
    val trainIndepExs = SingleTaskStructTesting.extractAllTaskExamples(trainDocGraphs, 
                                              nerFeaturizer,
                                              goldWikification, wikiDB, true, jqdcomputer,
                                              basicFeaturizer);
    
    println("FeatureCnt1 = " + featIndexer.size);
    
    // extract joint structural examples!
    val trainStructs = extractJointNoFeature(trainIndepExs);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    
    //val orderedTasks = Array[Int](COREF_TASK_INDEX, NER_TASK_INDEX, WIKI_TASK_INDEX);
    //val orderedTasks = Array[Int](COREF_TASK_INDEX, WIKI_TASK_INDEX, NER_TASK_INDEX);
    
    //val orderedTasks = Array[Int](NER_TASK_INDEX, COREF_TASK_INDEX, WIKI_TASK_INDEX);
    //val orderedTasks = Array[Int](NER_TASK_INDEX, WIKI_TASK_INDEX, COREF_TASK_INDEX);
    
    //val orderedTasks = Array[Int](WIKI_TASK_INDEX, NER_TASK_INDEX, COREF_TASK_INDEX);
    val orderedTasks = Array[Int](WIKI_TASK_INDEX, COREF_TASK_INDEX, NER_TASK_INDEX);

    val stackingModel = stackTraining[MCNerFeaturizer](orderedTasks, trainStructs, wikiDB, jointFeaturier);
    
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // test
    testGivenDocPath[MCNerFeaturizer]("TestDocs", 
    		testDataPath, 
    		mentionPropertyComputer,
    		berkPruner,
    		nerFeaturizer,
    		goldWikification,
    		wikiDB,
    		jqdcomputer,
    		basicFeaturizer,
    		featIndexer,
    		stackingModel);
    
    // dev
    testGivenDocPath[MCNerFeaturizer]("DevDocs", 
    		devDataPath, 
    		mentionPropertyComputer,
    		berkPruner,
    		nerFeaturizer,
    		goldWikification,
    		wikiDB,
    		jqdcomputer,
    		basicFeaturizer,
    		featIndexer,
    		stackingModel);

  }
   
  def testGivenDocPath[NerFeaturizerTp](note: String,
                       testDataPath: String, 
                       mentionPropertyComputer: MentionPropertyComputer,
                       berkPruner: CorefPruner,
                       nerFeaturizer: MCNerFeaturizer,
                       goldWikification: HashMap[String, DocWikiAnnots], 
                       wikiDB: WikipediaInterface,
                       jqdcomputer: JointQueryDenotationChoiceComputer,
                       basicFeaturizer: PairwiseIndexingFeaturizerJoint,
                       featIndexer: Indexer[String],
                       stackingModel: StackingModel[NerFeaturizerTp]) {
    
    println();
    println("========" + note + "========");
    
    // test examples
    val testDocGraphs = SingleTaskStructTesting.testGetdocgraphs(testDataPath, -1, mentionPropertyComputer);
    val testDocs = testDocGraphs.map { dg => dg.corefDoc };
    berkPruner.pruneAll(testDocGraphs);
    
    val testIndepExs = SingleTaskStructTesting.extractAllTaskExamples(testDocGraphs, 
                                               nerFeaturizer,
                                               goldWikification, wikiDB, false, jqdcomputer,
                                               basicFeaturizer);

    // test examples
    val testStructs = extractJointNoFeature(testIndepExs);
    println("FeatureCnt1.1 = " + featIndexer.size);
    
    stackTesting[NerFeaturizerTp](testStructs, stackingModel);

    // evaluate
    JointTaskStructTesting.evaluateAceStructsJoint(testStructs, goldWikification);
  }
  
  def initOutputPrediction(exs: Seq[AceJointTaskExample]) = {
    val predicts = new HashMap[AceJointTaskExample, Array[Int]]();
    for (ex <- exs) {
      val output = new Array[Int](ex.totalSize);
      Arrays.fill(output, -1); // -1 indicates not assigned value
      predicts += (ex -> output);
    }
    predicts;
  }
      
  def stackTraining[NerFeaturizerType](taskOrdering: Array[Int],
                    trainStructs: Seq[AceJointTaskExample],
                    //testStructs: Seq[AceJointTaskExample],
                    wikiDB: WikipediaInterface,
                    jointFeaturizer: JointFeaturizerShared[NerFeaturizerType]) = {
    
    checkOrdering(taskOrdering); // have a look at ordering
    
    val featIndexer = jointFeaturizer.indexer;
    
    // init mask
    val taskMask = new Array[Boolean](3);
    Arrays.fill(taskMask, false);
    
    
    val trainOutputResults = initOutputPrediction(trainStructs);
   //val testOutputResults  = initOutputPrediction(testStructs);
    
    ///////////////// first task //////////////////////
    
    taskMask(taskOrdering(0) - 1) = true;
    featurizeAblationBatch[NerFeaturizerType](trainStructs, wikiDB, jointFeaturizer,
                                              taskMask(COREF_TASK_INDEX - 1),
                                              taskMask(NER_TASK_INDEX - 1),
                                              taskMask(WIKI_TASK_INDEX - 1));

    System.gc();
    println("FeatureCntTask1 = " + featIndexer.size);
    
    val wght1 = trainAndInferenceSingleTaskStacking(taskOrdering(0),
                                        taskMask(COREF_TASK_INDEX - 1),
                                        taskMask(NER_TASK_INDEX - 1),
                                        taskMask(WIKI_TASK_INDEX - 1),
                                        trainStructs,
                                        trainOutputResults,
                                        featIndexer);
    
    
    ///////////////// Second task /////////////////////
    
    taskMask(taskOrdering(1) - 1) = true;
    featurizeAblationBatch[NerFeaturizerType](trainStructs, wikiDB, jointFeaturizer,
                                              taskMask(COREF_TASK_INDEX - 1),
                                              taskMask(NER_TASK_INDEX - 1),
                                              taskMask(WIKI_TASK_INDEX - 1));

    
    System.gc();
    println("FeatureCntTask2 = " + featIndexer.size);
    
    val wght2 = trainAndInferenceSingleTaskStacking(taskOrdering(1),
    		taskMask(COREF_TASK_INDEX - 1),
    		taskMask(NER_TASK_INDEX - 1),
    		taskMask(WIKI_TASK_INDEX - 1),
    		trainStructs,
    		trainOutputResults,
    		featIndexer);
    
    ///////////////// Third task /////////////////////
    
    taskMask(taskOrdering(2) - 1) = true;
    featurizeAblationBatch[NerFeaturizerType](trainStructs, wikiDB, jointFeaturizer,
                                              taskMask(COREF_TASK_INDEX - 1),
                                              taskMask(NER_TASK_INDEX - 1),
                                              taskMask(WIKI_TASK_INDEX - 1));

    System.gc();
    println("FeatureCntTask3 = " + featIndexer.size);
    
    val wght3 = trainAndInferenceSingleTaskStacking(taskOrdering(2),
    		taskMask(COREF_TASK_INDEX - 1),
    		taskMask(NER_TASK_INDEX - 1),
    		taskMask(WIKI_TASK_INDEX - 1),
    		trainStructs,
    		trainOutputResults,
    		featIndexer);
    
    
    /// assemble model
    val md = new StackingModel[NerFeaturizerType](taskOrdering, wght1, wght2, wght3, wikiDB, jointFeaturizer);
    md;
  }
  
  
  def stackTesting[NerFeaturizerType](//taskOrdering: Array[Int],
                    //trainStructs: Seq[AceJointTaskExample],
                    testStructs: Seq[AceJointTaskExample],
                    //wikiDB: WikipediaInterface,
                    //jointFeaturizer: JointFeaturizerShared[NerFeaturizerType],
                    model: StackingModel[NerFeaturizerType]) {
    
    System.gc();
    //////////////////////////////////////////
    checkOrdering(model.ordering); // have a look at ordering
    
    val featIndexer = model.jointFeaturizer.indexer;
    
    // init mask
    val taskMask = new Array[Boolean](3);
    Arrays.fill(taskMask, false);
    
    val testOutputResults  = initOutputPrediction(testStructs);
    
    ///////////////// first task //////////////////////
    
    taskMask(model.ordering(0) - 1) = true;
    featurizeAblationBatch[NerFeaturizerType](testStructs, model.wikiDB, model.jointFeaturizer,
                                              taskMask(COREF_TASK_INDEX - 1),
                                              taskMask(NER_TASK_INDEX - 1),
                                              taskMask(WIKI_TASK_INDEX - 1));
    

    System.gc();
    println("FeatureCntTask1 = " + featIndexer.size);
    
    checkPredictPercetage(testOutputResults); // before
    testSingleTaskStacking(model.weight1, model.ordering(0), taskMask(COREF_TASK_INDEX - 1),
                                                             taskMask(NER_TASK_INDEX - 1),
                                                             taskMask(WIKI_TASK_INDEX - 1), testStructs, testOutputResults);
    checkPredictPercetage(testOutputResults); // after

    
    
    
    
    ///////////////// Second task /////////////////////
    
    taskMask(model.ordering(1) - 1) = true;
    featurizeAblationBatch[NerFeaturizerType](testStructs, model.wikiDB, model.jointFeaturizer,
                                              taskMask(COREF_TASK_INDEX - 1),
                                              taskMask(NER_TASK_INDEX - 1),
                                              taskMask(WIKI_TASK_INDEX - 1));

    
    System.gc();
    println("FeatureCntTask2 = " + featIndexer.size);
    
    checkPredictPercetage(testOutputResults); // before
    testSingleTaskStacking(model.weight2, model.ordering(1), taskMask(COREF_TASK_INDEX - 1),
                                                             taskMask(NER_TASK_INDEX - 1),
                                                             taskMask(WIKI_TASK_INDEX - 1), testStructs, testOutputResults);
    checkPredictPercetage(testOutputResults); // after
    
    
    
    
    ///////////////// Third task /////////////////////
    
    taskMask(model.ordering(2) - 1) = true;
    featurizeAblationBatch[NerFeaturizerType](testStructs, model.wikiDB, model.jointFeaturizer,
                                              taskMask(COREF_TASK_INDEX - 1),
                                              taskMask(NER_TASK_INDEX - 1),
                                              taskMask(WIKI_TASK_INDEX - 1));

    
    System.gc();
    println("FeatureCntTask3 = " + featIndexer.size);
    
    checkPredictPercetage(testOutputResults); // before
    testSingleTaskStacking(model.weight3, model.ordering(2), taskMask(COREF_TASK_INDEX - 1),
                                                             taskMask(NER_TASK_INDEX - 1),
                                                             taskMask(WIKI_TASK_INDEX - 1), testStructs, testOutputResults);
    checkPredictPercetage(testOutputResults); // after

    
    //////////////////////////////////////////
    for ((ex, outp) <- testOutputResults) {
      ex.currentOutput = outp; // assign the predict result
    }
  }
  
  def checkOrdering(taskOrdering: Array[Int]) {
    val names = new HashMap[Int, String]();
    names += (COREF_TASK_INDEX -> "Coref");
    names += (NER_TASK_INDEX -> "Ner");
    names += (WIKI_TASK_INDEX -> "Wiki");
    val str = names(taskOrdering(0)) + " --> " + names(taskOrdering(1)) + " --> " + names(taskOrdering(2));
    println("");
    println("TaskOrdering: " + str);
  }
  
  def featurizeGivenValue(ex: AceJointTaskExample, currentTaskIndex: Int, varIdx: Int, valueIdx: Int,
                          doCoref: Boolean, doNer: Boolean, doWiki: Boolean,
                          predict: Array[Int]): HashMap[Int,Double] = {
    
    val feat: HashMap[Int,Double] = new HashMap[Int,Double](); // empty vector
    
    val ivariable = ex.getVariableGivenIndex(varIdx);
    val valueClas = ivariable.values(valueIdx);
    
    val useJointFeat = true;
    
    
    if (currentTaskIndex == COREF_TASK_INDEX) {
      
      // single variable feature
      ex.addIndexedVector(feat, valueClas.feature);
      
      val ival = ex.getSingleTaskIndex(varIdx);
      val corefVar = ex.getCorefVar(ival);
      val jval = corefVar.values(valueIdx).value;
      
if (useJointFeat) {
      if (doNer) {
    	  if (jval != ival) {
    		  val factor = ex.corefNerFactors(ival)(jval); 
    		  if (factor != null) {
    			  val invIdx = ex.getNerGlobalOutputValue(ival, predict);//outputB(i);
    			  val jnvIdx = ex.getNerGlobalOutputValue(jval, predict);//outputC(j);
    			  val feats = factor.feature(invIdx)(jnvIdx);
    			  ex.addIndexedVector(feat, feats);
    		  }
    	  }
      }

      if (doWiki) {
        if (jval != ival) {
    		  val factor = ex.corefWikiFactors(ival)(jval); 
    		  if (factor != null) {
    			  val invIdx = ex.getWikiGlobalOutputValue(ival, predict);//outputB(i);
    			  val jnvIdx = ex.getWikiGlobalOutputValue(jval, predict);//outputC(j);
    			  val feats = factor.feature(invIdx)(jnvIdx);
    			  ex.addIndexedVector(feat, feats);
    		  }
    	  }
      }
}

      return feat;
      
    } else if (currentTaskIndex == NER_TASK_INDEX) {
      
      // single variable feature
      ex.addIndexedVector(feat, valueClas.feature);
      
      val ival = ex.getSingleTaskIndex(varIdx);
      val nerVar = ex.getNerVar(ival);

if (useJointFeat) {
      if (doCoref) {
        val corefVar = ex.getCorefVar(ival);
        
        val jvalueIdx = ex.getCorefGlobalOutputValue(ival, predict);
        val jval = corefVar.values(jvalueIdx).value;
    	  val factor = ex.corefNerFactors(ival)(jval);
    	  if (ival != jval) {
    		  if (factor != null) {
    			  val invIdx = valueIdx;//ex.getNerGlobalOutputValue(ival, predict);
    			  val jnvIdx = ex.getNerGlobalOutputValue(jval, predict);
    			  val feats = factor.feature(invIdx)(jnvIdx);
    			  ex.addIndexedVector(feat, feats);
    		  }
    	  }
      }
      
      if (doWiki) {
    	  val factor = ex.nerWikiFactors(ival); 
    	  if (factor != null) {
    		  val iwvIdx = ex.getWikiGlobalOutputValue(ival, predict);
    		  val feats = factor.feature(valueIdx)(iwvIdx);
    		  ex.addIndexedVector(feat, feats);
    	  }
      }
}
      return feat;
      
      
    } else if (currentTaskIndex == WIKI_TASK_INDEX) {
      
      // single variable feature
      ex.addIndexedVector(feat, valueClas.feature);
      
      val ival = ex.getSingleTaskIndex(varIdx);

if (useJointFeat) {
      if (doCoref) {
    	  val corefVar = ex.getCorefVar(ival);

    	  val jvalueIdx = ex.getCorefGlobalOutputValue(ival, predict);
    	  val jval = corefVar.values(jvalueIdx).value;
        
    	  //val jval = ex.getCorefGlobalOutputValue(ival, predict);
    	  val factor = ex.corefWikiFactors(ival)(jval);
    	  if (ival != jval) {
    		  if (factor != null) {
    			  val invIdx = valueIdx;//ex.getWikiGlobalOutputValue(ival, predict);
    			  val jnvIdx = ex.getWikiGlobalOutputValue(jval, predict);
    			  val feats = factor.feature(invIdx)(jnvIdx);
    			  ex.addIndexedVector(feat, feats);
    		  }
    	  }
      }
      
      if (doNer) {
    	  val factor = ex.nerWikiFactors(ival);
    	  if (factor != null) {
    		  val invIdx = ex.getNerGlobalOutputValue(ival, predict);
    		  val feats = factor.feature(invIdx)(valueIdx);
    		  ex.addIndexedVector(feat, feats);
    	  }
      }
}
      return feat;
      
    }
    
    return null;
  }
  

  def buildValues(ex: AceJointTaskExample, currentTaskIndex: Int, varIdx: Int,
                  doCoref: Boolean, doNer: Boolean, doWiki: Boolean, applyPrune: Boolean,
                  output: Array[Int]): Array[StackingValue] = {
    
    val svalues = new ArrayBuffer[StackingValue]();
    
    val ivariable = ex.getVariableGivenIndex(varIdx);
    val indices = if (applyPrune) { 
    	ivariable.getAllNonPruningValueIndices();
    } else {
    	ivariable.getAllValueIndices();
    }

    for (vIdx <- indices) {
    	val varValue = ivariable.values(vIdx);
    	//val vfeat = featurizeGivenValue(ex, currentTaskIndex, varIdx, vIdx, doCoref, doNer, doWiki, output);
    	//val newv = new StackingValue(vIdx, vfeat, varValue.isCorrect);
    	val newv = new StackingValue(vIdx, varValue.isCorrect);
    	svalues += newv;
    }

    return svalues.toArray;
  }

  
  def constructDecisions(taskIndex: Int,
                         doCoref: Boolean, doNer: Boolean, doWiki: Boolean,
                         ex: AceJointTaskExample, output: Array[Int]): ArrayBuffer[StackingDecision] = {
    
    val decisions = new ArrayBuffer[StackingDecision]();
    val (startIdx, endIdx) = ex.getGlobalStartEndGivenTaskIndex(taskIndex);
    for (idx <- startIdx until endIdx) {
      val stackValues = buildValues(ex, taskIndex, idx, doCoref, doNer, doWiki, true, output);
      val decs = new StackingDecision(taskIndex, stackValues, idx);
      decisions += decs;
    }
    
    return decisions;
  }
  
  def constructStruct(taskIndex: Int,
                      doCoref: Boolean, doNer: Boolean, doWiki: Boolean,
                      ex: AceJointTaskExample, output: Array[Int]): StackingStruct = {
    
    val decisions = new ArrayBuffer[StackingDecision]();
    val (startIdx, endIdx) = ex.getGlobalStartEndGivenTaskIndex(taskIndex);
    for (idx <- startIdx until endIdx) {
      val stackValues = buildValues(ex, taskIndex, idx, doCoref, doNer, doWiki, true, output);
      val decs = new StackingDecision(taskIndex, stackValues, idx);
      decisions += decs;
    }
    
    val stackingStruct = new StackingStruct(ex, decisions.toArray, output, taskIndex, doCoref, doNer, doWiki);
    return stackingStruct;
  }
  
  def trainAndInferenceSingleTaskStacking(taskIndex: Int,
                                          doCoref: Boolean, doNer: Boolean, doWiki: Boolean,
                                          trainStructs: Seq[AceJointTaskExample],
                                          trnOutputs: HashMap[AceJointTaskExample, Array[Int]],
                                          featIndexer: Indexer[String]) = {
    
    // extract decisions and featurization
    val stackingExs = new ArrayBuffer[StackingStruct]();
    for (ex <- trainStructs) {
      val predict = trnOutputs(ex);
      val stackingStruct = constructStruct(taskIndex, doCoref, doNer, doWiki, ex, predict);
      stackingExs += stackingStruct;
    }
    
    // train model
    val weight: Array[Double] = trainPerceptron(stackingExs, featIndexer);
    
    // predict on train examples
    checkPredictPercetage(trnOutputs); // before
    testSingleTaskStacking(weight, taskIndex, doCoref, doNer, doWiki, trainStructs, trnOutputs);
    checkPredictPercetage(trnOutputs); // after
    
    weight;
  }
  
  // do inference
  def testSingleTaskStacking(weight: Array[Double],
                             taskIndex: Int,
                             doCoref: Boolean, doNer: Boolean, doWiki: Boolean,
                             testStructs: Seq[AceJointTaskExample],
                             exOutputs: HashMap[AceJointTaskExample, Array[Int]]) {
    
    var total = 0;
    var crr = 0;
    for (ex <- testStructs) {
      val output = exOutputs(ex);
      val stackingStruct = constructStruct(taskIndex, doCoref, doNer, doWiki, ex, output);
      val results = stackingStruct.sequentialInference(weight);
      //val results = stackingStruct.sequentialInferenceCorrect(weight);
      stackingStruct.assignOutput(results, output);
      val (exTotal, exCrr) = stackingStruct.computeTotalAndCorrect(results);
      total += exTotal;
      crr += exCrr;
    }
    
    val rate = crr.toDouble / total.toDouble;
    println("Accuracy: " + crr + " / " + total + " = " + rate);
  }
  
  def checkPredictPercetage(structOutputs: HashMap[AceJointTaskExample, Array[Int]]) {
    var totalCnt = 0;
    var predictCnt = 0;
    for ((ex,y) <- structOutputs) {
      for (yi <- y) {
        totalCnt += 1;
        if (yi >= 0) {
          predictCnt += 1;
        }
      }
    }
    val perc = predictCnt.toDouble / totalCnt.toDouble;
    println(predictCnt + " / " + totalCnt + " = " + perc);
  }
  
  
  ////////////////////
  
/*
  def trainPerceptron(allDecisions: Seq[StackingDecision], featIndexer: Indexer[String]) = {
    
      var weight = Array.fill[Double](featIndexer.size)(0);
      var weightSum = Array.fill[Double](featIndexer.size)(0);
      var lastWeight = Array.fill[Double](featIndexer.size)(0);

      val Iteration = 5;
      val learnRate = 0.1;
      val lambda = 1e-8;

      var updateCnt = 0;
      var lastUpdtCnt = 0;

      for (iter <- 0 until Iteration) {
        lastUpdtCnt = updateCnt;
        Array.copy(weight, 0, lastWeight, 0, weight.length);

        println("Iter " + iter);
        var exId = 0;
        for (example <- allDecisions) {

          exId += 1;
          val domains = example.values;
          val prunOk = (example.getCorrectCount() > 0);
          if (prunOk) {

            var bestLbl: StackingValue = null;
            var bestScore = -Double.MaxValue;
            var bestCorrectLbl: StackingValue = null; // latent best
            var bestCorrectScore = -Double.MaxValue;
            for (l <-domains) {
              var score = SingleTaskStructTesting.computeScoreSparse(weight, l.feature);//computeScore(weight, example.features(j));
              if (score > bestScore) {
                bestScore = score;
                bestLbl = l;
              }
              if (l.isCrr) {
                if (score > bestCorrectScore) {
                  bestCorrectScore = score;
                  bestCorrectLbl = l;
                }
              }
            }

            //println("size = " + domains.size + " pred = " + bestLbl + " correct = " + bestCorrectLbl)

            // update?
            if (!(bestLbl.isCrr)) {
              updateCnt += 1;
              if (updateCnt % 1000 == 0) println("Update " + updateCnt);
              SearchBasedLearner.updateWeight(weight, 
                           bestCorrectLbl.feature,
                           bestLbl.feature,
                           learnRate,
                           lambda);
              SingleTaskStructTesting.sumWeight(weightSum, weight);
            }
          }
        }

        ///////////////////////////////////////////////////
        // have a test after each iteration (for learning curve)
        val tmpAvg = new Array[Double](weightSum.size)
        Array.copy(weightSum, 0, tmpAvg, 0, weightSum.size);
        SingleTaskStructTesting.divdeNumber(tmpAvg, updateCnt.toDouble);

        //quickTest(allTrains, tmpAvg, wikiDB);
        println("Iter Update Cnt = " + (updateCnt - lastUpdtCnt));

      }

      SingleTaskStructTesting.divdeNumber(weightSum, updateCnt.toDouble);
      weightSum;
  }
*/
  
    def trainPerceptron(allDecisions: Seq[StackingStruct], featIndexer: Indexer[String]) = {

      var weight = Array.fill[Double](featIndexer.size)(0);
      var weightSum = Array.fill[Double](featIndexer.size)(0);

      val Iteration = 10;
      val learnRate = 0.1;
      val lambda = 1e-8;

      var updateCnt = 0;
      var lastUpdtCnt = 0;

      for (iter <- 0 until Iteration) {
        lastUpdtCnt = updateCnt;

        for (example <- allDecisions) {
          
          val predBestOutput = example.sequentialInference(weight);
          val goldBestOutput = example.sequentialInferenceCorrect(weight);

          // update?
          if (!example.isCorrectOutput(predBestOutput)) {
            updateCnt += 1;
            if (updateCnt % 1000 == 0) println("Update " + updateCnt);
            
            val featGold = example.featurize(goldBestOutput);
            val featPred = example.featurize(predBestOutput);
            
            //checkWeight(weight, featGold, featPred);
            
            SearchBasedLearner.updateWeight(weight, featGold, featPred, learnRate, lambda);
            SingleTaskStructTesting.sumWeight(weightSum, weight);
          }
        }
        
        println("Iter Update Cnt = " + (updateCnt - lastUpdtCnt));
      
        ///////////////////////////////////////////////////
        // have a test after each iteration (for learning curve)
        val tmpAvg = new Array[Double](weightSum.size)
        Array.copy(weightSum, 0, tmpAvg, 0, weightSum.size);
        SingleTaskStructTesting.divdeNumber(tmpAvg, updateCnt.toDouble);
        quickTestStacking(tmpAvg, allDecisions);
      }
      
      
      

      SingleTaskStructTesting.divdeNumber(weightSum, updateCnt.toDouble);
      weightSum;
  }
    
  def checkWeight(currentWeight: Array[Double], 
                  featGold: HashMap[Int,Double],
                  featPred: HashMap[Int,Double]) {
    var gradient = Array.fill[Double](currentWeight.length)(0);
    for ((i, vgold) <- featGold) {
      gradient(i) += (vgold);
    }
    for ((j, vpred) <- featPred) {
       gradient(j) -= (vpred);
    }

    var nonZero = 0;
    for (i2 <- 0 until currentWeight.length) {
      if (gradient(i2) != 0) {
        nonZero += 1;
      }
    }
    
    println("nonZero = " + nonZero);
  }
  
  def quickTestStacking(weight: Array[Double],
                        testDecs: Seq[StackingStruct]){
                        //exOutputs: HashMap[AceJointTaskExample, Array[Int]]) {
    
    var total = 0;
    var crr = 0;
    for (stackingStruct <- testDecs) {
      val results = stackingStruct.sequentialInference(weight);
      val (exTotal, exCrr) = stackingStruct.computeTotalAndCorrect(results);
      total += exTotal;
      crr += exCrr;
    }
    
    val rate = crr.toDouble / total.toDouble;
    println("===> Quick accuracy: " + crr + " / " + total + " = " + rate);
  }
  
}