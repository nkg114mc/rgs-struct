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
import berkeleyentity.ner.MCNerFeaturizerBase
import berkeleyentity.oregonstate.pruner.StaticDomainPruner
import berkeleyentity.oregonstate.pruner.LinearDomainPruner
import berkeleyentity.oregonstate.pruner.NonePruner
import berkeleyentity.oregonstate.pruner.InitPruner
import ml.dmlc.xgboost4j.scala.XGBoost
import ml.dmlc.xgboost4j.scala.Booster



/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////


object JointTaskStructTesting {
  
  // weight error should be between [0, 1], and the sum of three tasks should be 
  val CorefErrorWeight: Double = 1.0;
  val NerErrorWeight: Double   = 1.0;
  val WikiErrorWeight: Double  = 1.0;

  
  def extractJointExamples[NerFeaturizerType](docGraphs: Seq[DocumentGraph], 
                           nerFeaturizer: MCNerFeaturizerBase,//MCNerFeaturizer,
                           goldWikification: CorpusWikiAnnots, wikiDB: WikipediaInterface, filterImpossible: Boolean = false, jqdcomputer: JointQueryDenotationChoiceComputer,
                           pairwiseIndexingFeaturizer: PairwiseIndexingFeaturizer,
                           jointFeaturizer: JointFeaturizerShared[NerFeaturizerType]) = {
    
    val result = ArrayBuffer[AceJointTaskExample]();
    for (d <- docGraphs) {
      val jointOneStructExmp = extractJointTaskOneExample(d, nerFeaturizer, goldWikification, wikiDB, filterImpossible, jqdcomputer, pairwiseIndexingFeaturizer, jointFeaturizer);
      result += jointOneStructExmp;
    }
    result;
  }
  
  def extractJointExamplesFromMultiExmps[NerFeaturizerType](multiExs: Seq[AceMultiTaskExample], 
                           nerFeaturizer: MCNerFeaturizerBase,//MCNerFeaturizer,
                           goldWikification: CorpusWikiAnnots, wikiDB: WikipediaInterface, filterImpossible: Boolean = false, jqdcomputer: JointQueryDenotationChoiceComputer,
                           pairwiseIndexingFeaturizer: PairwiseIndexingFeaturizer,
                           jointFeaturizer: JointFeaturizerShared[NerFeaturizerType]) = {
    
    val result = ArrayBuffer[AceJointTaskExample]();
    for (ex <- multiExs) {
      val jointOneStructExmp = extractJointExmpGivenMultiExample(ex, nerFeaturizer, goldWikification, wikiDB, filterImpossible, jqdcomputer, pairwiseIndexingFeaturizer, jointFeaturizer);
      result += jointOneStructExmp;
    }
    result;
  }

  def extractJointTaskOneExample[NerFeaturizerType](docGraph: DocumentGraph, 
                           nerFeaturizer: MCNerFeaturizerBase,//MCNerFeaturizer,
                           goldWikification: CorpusWikiAnnots, wikiDB: WikipediaInterface, filterImpossible: Boolean = false, jqdcomputer: JointQueryDenotationChoiceComputer,
                           pairwiseIndexingFeaturizer: PairwiseIndexingFeaturizer,
                           jointFeaturizer: JointFeaturizerShared[NerFeaturizerType] ) = {
         val indepTaskStrucExmp = SingleTaskStructTesting.extractAllTaskOneExample(docGraph, nerFeaturizer, goldWikification, wikiDB, filterImpossible, jqdcomputer, pairwiseIndexingFeaturizer);
         val jointOneStructExmp = extractJointExmpGivenMultiExample(indepTaskStrucExmp, nerFeaturizer, goldWikification, wikiDB, filterImpossible, jqdcomputer, pairwiseIndexingFeaturizer, jointFeaturizer);
         jointOneStructExmp;
   }
  
  def extractJointExmpGivenMultiExample[NerFeaturizerType](independentTaskStrucExmp: AceMultiTaskExample,
                           nerFeaturizer: MCNerFeaturizerBase,//MCNerFeaturizer,
                           goldWikification: CorpusWikiAnnots, wikiDB: WikipediaInterface, filterImpossible: Boolean = false, jqdcomputer: JointQueryDenotationChoiceComputer,
                           pairwiseIndexingFeaturizer: PairwiseIndexingFeaturizer,
                           jointFeaturizer: JointFeaturizerShared[NerFeaturizerType] ) = {


	  ////////////////////////////////
    
    val docGraph = independentTaskStrucExmp.docGraph;

    val corefVars = independentTaskStrucExmp.corefOutput.variables;
    val nerVars = independentTaskStrucExmp.nerOutput.variables;
    val wikiVars = independentTaskStrucExmp.wikiOutput.variables;
    
    val jointStructExmp = new AceJointTaskExample(corefVars, nerVars, wikiVars, docGraph);

    
    /////////////////////////////////////////
    
    
    val corefDoc: CorefDoc = docGraph.corefDoc;
    val rawDoc = corefDoc.rawDoc;
    val docName = rawDoc.docID;
    val addToIdxer: Boolean = docGraph.addToFeaturizer;
    
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
    					jointFeaturizer.getCorefNerFeatures(docGraph, i, j, currNerNode.values(currNerValIdx).value, antNerNode.values(antNerValIdx).value, addToIdxer);
    				  //Array();
    				});
    				//agreementFactors(i)(j) = addAndReturnFactor(new BetterPropertyFactor[String](j, currNerNode, corefNodes(i), antNerNode, featsIndexed), true);
    				jointStructExmp.corefNerFactors(i)(j) = new TernaryTaskFactor[Int, String, String](corefVars(i), currNerNode, antNerNode, featsIndexed);// BetterPropertyFactor[String](j, currNerNode, corefNodes(i), antNerNode, featsIndexed), true);
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
    					jointFeaturizer.getCorefWikiFeatures(docGraph, i, j, currWikiNode.values(currWikiValIdx).value.wiki, antWikiNode.values(antWikiValIdx).value.wiki, Some(wikiDB), addToIdxer);
    				  //Array();
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
    		jointFeaturizer.getWikiNerFeatures(docGraph, i, wikiNode.values(wikiValIdx).value.wiki, nerNode.values(nerValIdx).value, Some(wikiDB), addToIdxer);
    	  //Array();
    	});
    	jointStructExmp.nerWikiFactors(i) = new BinaryTaskFactor[String, QueryWikiValue](nerNode, wikiNode, featsIndexed);
    }
  }
	  /////////////////////////////////
	  jointStructExmp;
  }
  
  
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

	  trainStructuralJointTasks();
	  //runStandardBerkTrainTestWithPruner();
    //runSelfConstructBerkeley();
  }
  
  def trainStructuralJointTasks() {
    
    val trainDataPath = "data/ace05/train";
    val devDataPath = "data/ace05/dev";
    //val testDataPath = "data/ace05/test";
    val testDataPath = devDataPath;
    
    //val trainDataPath = "data/ace05/train_1";
    //val testDataPath = "data/ace05/test_1";
    
    val dictNer = NerTesting.constructDictionary(trainDataPath, devDataPath, testDataPath);
    
    val wikiPath = "data/ace05/ace05-all-conll-wiki"
    val wikiDBPath = "models/wiki-db-ace.ser.gz"
    
    val berkeleyCorefPrunerDumpPath = "models:./corefpruner-ace.ser.gz:-5";
    //val berkeleyCorefPrunerDumpPath = "build:./corefpruner-ace.ser.gz:-5:5";
    
    // independent model only
    //Driver.corefNerFeatures = "";
    //Driver.wikiNerFeatures = "";
    //Driver.corefWikiFeatures = "";
    
    val featIndexer = new Indexer[String]();
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentTypePredictor = AceMentionTypePredictor.TrainMentionTypeACE();
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer), Some(mentTypePredictor));
    //val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    val assembler = CorefDocAssembler(Language.ENGLISH, true); //use gold mentions
    val trainDocs = ConllDocReader.loadRawConllDocsWithSuffix(trainDataPath, -1, "", Language.ENGLISH);
    val trainCorefDocs = trainDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    
    val maybeBrownClusters = Some(BrownClusterInterface.loadBrownClusters(Driver.brownPath, 0));
    val nerFeaturizer = MCNerFeaturizer(Driver.nerFeatureSet.split("\\+").toSet, featIndexer, MCNerFeaturizer.StdLabelIndexer, trainDocs.flatMap(_.words), None, maybeBrownClusters);
    val nerNewFeatr = new NerAdditionalFeature(featIndexer, MCNerFeaturizer.StdLabelIndexer, dictNer);
    val nerAddonFeaturizer = new MCNerAddonFeaturizer(nerFeaturizer, nerNewFeatr);
    
    
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
    //val jointFeaturier = new JointFeaturizerShared[MCNerFeaturizer](basicFeaturizer, nerFeaturizer, maybeBrownClusters, Driver.corefNerFeatures, Driver.corefWikiFeatures, Driver.wikiNerFeatures, featIndexer);
    val jointFeaturier = new JointFeaturizerShared[MCNerAddonFeaturizer](basicFeaturizer, nerAddonFeaturizer, maybeBrownClusters, Driver.corefNerFeatures, Driver.corefWikiFeatures, Driver.wikiNerFeatures, featIndexer);

    
    /// extract independent task examples
    val trainIndepExs = SingleTaskStructTesting.extractAllTaskExamples(trainDocGraphs, 
                                              nerFeaturizer,
                                              goldWikification, wikiDB, true, jqdcomputer,
                                              basicFeaturizer);
    
    // test examples
    val testDocGraphs = SingleTaskStructTesting.testGetdocgraphs(testDataPath, -1, mentionPropertyComputer);
    val testDocs = testDocGraphs.map { dg => dg.corefDoc };
    berkPruner.pruneAll(testDocGraphs);
    
    val testIndepExs = SingleTaskStructTesting.extractAllTaskExamples(testDocGraphs, 
                                           nerFeaturizer,
                                           goldWikification, wikiDB, false, jqdcomputer,
                                           basicFeaturizer);
    
    
    CorefAdditionalFeature.testPruningCoverage(trainDocGraphs);
    CorefAdditionalFeature.testPruningCoverage(testDocGraphs);
    CorefAdditionalFeature.testMentionTypeAcc(testDocGraphs, mentionPropertyComputer);
    //throw new RuntimeException("Exit early ...");
    
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
////// Perform Domain Pruning Here ///////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //val prunWghtFile = new File("pruner_weight_ace05.weight_full");
    val prunWghtFile = new File("pruner_develop_ace05.weight_full");
    /*
    // mention count!
    //SearchDomainPruner.mentionCount(trainIndepExs, testIndepExs);
    //SearchDomainPruner.mentionCount2(trainIndepExs, testIndepExs);
  
    /// PRUNER
    // train a unary potential as pruner
    //val pruneWeight = runIndependentTaskTraining(trainStructs, testStructs, featIndexer,  goldWikification);
    //val prunWghtFile = new File("pruner_weight_ace05.weight");
    val prunWghtFile = new File("pruner_weight_ace05.weight_full");
    val pruneWeight = if (!prunWghtFile.exists()) {
      val pw = runUnaryPrunerTraining(trainIndepExs, testIndepExs, featIndexer,  goldWikification);
      SearchDomainPruner.savePrunerWeight(pw, prunWghtFile.getAbsolutePath);
      pw;  
    } else {
    	val pw = new Array[Double](featIndexer.size);
    	SearchDomainPruner.loadPrunerWeight(pw, prunWghtFile.getAbsolutePath);
    	val histp = new HistgramRecord();
    	SingleTaskStructTesting.testStructuralAllTasks(testIndepExs, pw, histp);
    	histp.printHistgram();
    	pw;
    }
    //val unaryPruner = new SearchDomainPruner(pruneWeight);
    //val unaryPruner = new SyntheticPruner(pruneWeight);
    val unaryPruner = new NonePruner(pruneWeight);
    //unaryPruner.getBerkeleyCorefPruner(Driver.pruningStrategy, trainDataPath);
    val corefTopK = 5; 
    val nerTopK = 5;
    val wikiTopK = 4;
    unaryPruner.pruneDomainIndepBatch(trainIndepExs, corefTopK, nerTopK, wikiTopK);
    unaryPruner.pruneDomainIndepBatch(testIndepExs, corefTopK, nerTopK, wikiTopK);
    //unaryPruner.pruneDomainIndepWithBerkCorefPrunerBatch(trainIndepExs, corefTopK, nerTopK, wikiTopK);
    //unaryPruner.pruneDomainIndepWithBerkCorefPrunerBatch(testIndepExs, corefTopK, nerTopK, wikiTopK);
    SearchDomainPruner.checkPruningLoss(trainIndepExs, testIndepExs);
   
    // ERROR FINDER?
    //val errFinder = new ErrorFinder();
    //errFinder.errorFindingClassifierLearn(trainIndepExs, testIndepExs, featIndexer);
     */
    
    val pwght = if (!prunWghtFile.exists()) {
      //val pw = runUnaryPrunerTraining(trainIndepExs, testIndepExs, featIndexer,  goldWikification);
      val pw = LinearDomainPruner.runIndependentTaskTraining(trainIndepExs, testIndepExs, featIndexer);
      StaticDomainPruner.savePrunerWeight(pw, prunWghtFile.getAbsolutePath);
      pw;  
    } else {
      /*
    	val pw = new Array[Double](featIndexer.size);
    	SearchDomainPruner.loadPrunerWeight(pw, prunWghtFile.getAbsolutePath);
    	val histp = new HistgramRecord();
    	SingleTaskStructTesting.testStructuralAllTasks(testIndepExs, pw, histp);
    	histp.printHistgram();
    	*/
      val fpw = StaticDomainPruner.loadPrunerWeightUnknownLen(prunWghtFile.getAbsolutePath);
      if (fpw == null || fpw.length != featIndexer.size) { // need retrain...
        val trpw = LinearDomainPruner.runIndependentTaskTraining(trainIndepExs, testIndepExs, featIndexer);
        StaticDomainPruner.savePrunerWeight(trpw, prunWghtFile.getAbsolutePath);
        trpw;
      } else {
        fpw;
      }
    }
    val histp = new HistgramRecord();
    SingleTaskStructTesting.testStructuralAllTasks(testIndepExs, pwght, histp);
    SingleTaskStructTesting.evaluateAceStructs(testIndepExs, goldWikification);
    histp.printHistgram();
    ////////////////////////////////////
    val histp2 = new HistgramRecord();
    SingleTaskStructTesting.testStructuralAllTasks(trainIndepExs, pwght, histp2);
    SingleTaskStructTesting.evaluateAceStructs(trainIndepExs, goldWikification);
    histp2.printHistgram();
    ////////////////////////////////////
    //val linearPruner = new LinearDomainPruner(pwght, 5, 7, 12);
    //linearPruner.pruneDomainIndepBatch(trainIndepExs,5, 7, 12);
    //linearPruner.pruneDomainIndepBatch(testIndepExs,5, 7, 12);
    // check pruning loss?
    //SearchDomainPruner.checkPruningLoss(trainIndepExs, testIndepExs);
    //val unaryPruner = linearPruner;
    //unaryPruner.pruneDomainIndepBatch(trainIndepExs, -1, -1, -1);
    //unaryPruner.pruneDomainIndepBatch(testIndepExs, -1, -1, -1);
    val nonePruner = new NonePruner(new Array[Double](featIndexer.size));
    nonePruner.pruneDomainIndepBatch(trainIndepExs, -1, -1, -1);
    nonePruner.pruneDomainIndepBatch(testIndepExs, -1, -1, -1);
    
    //SingleTaskStructTesting.nerErrorAnalysis(testIndepExs, testDocs);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //reassignMentType(cdocs: Seq[CorefDoc], mentPredictor: AceMentionTypePredictor)
    
    // extract joint structural examples!
    val trainStructs = extractJointExamples(trainDocGraphs, 
                                            nerAddonFeaturizer,//nerFeaturizer,//
                                            goldWikification, wikiDB, true, jqdcomputer,
                                            basicFeaturizer,
                                            jointFeaturier);
    // test examples
    val testStructs = extractJointExamples(testDocGraphs, 
                                           nerAddonFeaturizer,//nerFeaturizer,//
                                           goldWikification, wikiDB, false, jqdcomputer,
                                           basicFeaturizer,
                                           jointFeaturier);

    println("FeatureCnt = " + featIndexer.size);
    
    val extendedPrunerWeight = SingleTaskStructTesting.extendWeight(pwght, featIndexer.size);
    val unaryPruner = new LinearDomainPruner(extendedPrunerWeight, 5, 7, 12);

    
    // Learning!
    val beamSize = 40;
    val restart = 10;
    val IterNum = 10;//100;
    val zbKeys = new ZobristKeys(3000, 1000);
    val greedySearcher = new SearchBasedLearner(zbKeys);
    
    val topAlpha = 0.5;//0.9;
    
    //InitPruner.runInitActionPrunerTraining(trainStructs, testStructs, featIndexer, unaryPruner, greedySearcher, 0.5);
    //InitPruner.dumpInitActionPrunerTrainingFiles(trainStructs ++ testStructs, testStructs, featIndexer, unaryPruner, greedySearcher, 0.5);
    //val prunBooster = XGBoost.loadModel("./model/xgb-initpruner-100.model");
    val prunBooster = XGBoost.loadModel("./model/xgb-initpruner-dev.model");
    //InitPruner.runInitActionPrunerTesting(trainStructs, featIndexer, unaryPruner, greedySearcher, topAlpha, prunBooster);
    //InitPruner.runInitActionPrunerTesting(testStructs, featIndexer, unaryPruner, greedySearcher, topAlpha, prunBooster);
    //throw new RuntimeException("Exit early ...");
    
    //val greedySearcher = new NerWikiOnlyLearner();
    //val weight = greedySearcher.runLearningDelayUpdate(trainStructs, featIndexer, testStructs, beamSize, restart, unaryPruner, IterNum);
    //val weight = greedySearcher.CSearchLearning(trainStructs, featIndexer, testStructs, unaryPruner, IterNum);

    val weight = StructuralSVMLearner.uiucStructLearning(trainStructs, featIndexer, testStructs, beamSize, restart, greedySearcher, unaryPruner);
    //StaticDomainPruner.savePrunerWeight(weight, "./model/cost_ace05_alpha" + String.valueOf(topAlpha) + ".weight");
    //val weight = StructuralSVMLearner.svmLoadModel();
    
    //greedySearcher.unaryScoreChecking(trainStructs, featIndexer, testStructs, unaryPruner);
    //val weight = getRandomWeight(featIndexer.size);//new Array[Double](featIndexer.size);

    //val ILPsolver = new ILPJointInferencer();
    //val weight = ILPsolver.runLearningPerceptron(trainStructs, featIndexer, testStructs, unaryPruner, IterNum);
    
    // test as oracle?
    //testPrunedOracle(testStructs, featIndexer.size());
    
    // test!
    //testHillClimb(testStructs, weight, greedySearcher, unaryPruner)
    testBeamSearch(testStructs, beamSize, restart, weight, greedySearcher, unaryPruner)
    //testGroundTruthBeamSearch(testStructs, beamSize, restart, weight, greedySearcher, unaryPruner);
    
    //val unaryWeight = new Array[Double](featIndexer.size);
    //Array.copy(unaryPruner.weight, 0, unaryWeight , 0, unaryPruner.weight.length);
    //testILPsolver(testStructs, weight, ILPsolver, unaryPruner.weight);
    
    //testNerWikiFactorOnly(testStructs, weight, greedySearcher, unaryPruner.weight);
    

    // evaluate
    StaticDomainPruner.noCorrectCountJoint(testStructs);
    evaluateAceStructsJoint(testStructs, goldWikification);
    //SingleTaskStructTesting.nerErrorAnalysisJoint(testStructs, testDocs);
    
    //println("==onTrain====================================================\n");
    //testBeamSearch(trainStructs, beamSize, restart, weight, greedySearcher, unaryPruner);
    //evaluateAceStructsJoint(trainStructs, goldWikification);
    
    //testPrunerUpperBounds(testStructs, beamSize, restart, featIndexer, greedySearcher, unaryPruner, prunBooster, goldWikification)
  }
  
  def testPrunerUpperBounds(testStructs: Seq[AceJointTaskExample], bsize: Int, restrt: Int, fIndexer: Indexer[String], searcher: SearchBasedLearner, pruner: StaticDomainPruner, 
                            booster: Booster, gWikification: HashMap[String, DocWikiAnnots]) {
    
	  val cAlphas: Array[Double] = Array( 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 );
    for (alpha <- cAlphas) {
	    println("============> Pruner ALPHA = " + alpha + " <====================");
	    InitPruner.runInitActionPrunerTesting(testStructs, fIndexer, pruner, searcher, alpha, booster);
	    val weight = getRandomWeight(fIndexer.size);//new Array[Double](featIndexer.size);
	    testGroundTruthBeamSearch(testStructs, bsize, 1, weight, searcher, pruner)
	    evaluateAceStructsJoint(testStructs, gWikification);
    }
  }
/*
  def reassignMentType(cdocs: Seq[CorefDoc], mentPredictor: AceMentionTypePredictor) {
    for (rawDoc <- cdocs) {
      val ments = rawDoc.predMentions;
      for (pm <- ments) {
        pm.mentionType = mentPredictor.predictType(pm);
      }
    }
  }
*/
  def getRandomWeight(len: Int): Array[Double] = {
    val w = new Array[Double](len);
    val rnd = new Random();
    for (i <- 0 until w.length) {
      w(i) = rnd.nextDouble(); 
    }
    w;
  }
  /*
  def testILPsolver(testStructs: Seq[AceJointTaskExample], wght: Array[Double], ilpSolver: ILPJointInferencer, unaryWght: Array[Double]) {
	  var finalCnt = 0;
    for (ex <- testStructs) {
      finalCnt += 1;
      println("Final Test: " + finalCnt);
		  ex.currentOutput = ilpSolver.runILPinference(ex, wght, false, (Array.fill[Int](ex.totalSize)(-1)), false);
		  ex.printTaskOutput(); 
	  }
  }
  */
  //def testHillClimb(testStructs: Seq[AceJointTaskExample], wght: Array[Double], searcher: SearchBasedLearner, unaryWght: Array[Double]) {
  def testHillClimb(testStructs: Seq[AceJointTaskExample], wght: Array[Double], searcher: SearchBasedLearner, pruner: StaticDomainPruner) {
    searcher.greedySearchQuickTest(testStructs, wght, pruner);
    for (ex <- testStructs) {
    	//val initState = SearchBasedLearner.getRandomInitState(ex);
    	val initState = SearchBasedLearner.getInitStateWithUnaryScore(ex, pruner, true);
    	val initWithMusk = searcher.prunedActionSpaceMusk(ex, initState);
    	val returnedState = searcher.hillClimbing(ex, initWithMusk, wght, false, false);
    	//val returnedState = searcher.hillClimbing(ex, initState, wght, false);
    	ex.currentOutput = returnedState.output;
    	//ex.printTaskOutput(); 
    }
  }
  
  def testBeamSearch(testStructs: Seq[AceJointTaskExample], bsize: Int, restrt: Int, wght: Array[Double], searcher: SearchBasedLearner, pruner: StaticDomainPruner) {
    searcher.greedySearchQuickTest(testStructs, wght, pruner);
    for (ex <- testStructs) {
    	//val initState = SearchBasedLearner.getRandomInitState(ex);
      //val initState = SearchBasedLearner.getZeroInitState(ex);
    	val initState = SearchBasedLearner.getInitStateWithUnaryScore(ex, pruner, true);
    	val initWithMusk = searcher.prunedActionSpaceMusk(ex, initState);
    	//val returnedState = searcher.hillClimbing(ex, initWithMusk, wght, true, false);
    	val returnedState = searcher.beamSearch(ex, initWithMusk, bsize, wght, false, false);
    	//val returnedState = searcher.searchWithRandomRestart(ex, bsize, wght, false, false, restrt);
    	ex.currentOutput = returnedState.output;
    	//ex.printTaskOutput(); 
    }
  }
  
  def testGreedyWithRestarts(testStructs: Seq[AceJointTaskExample], bsize: Int, restrt: Int, wght: Array[Double], searcher: SearchBasedLearner, pruner: StaticDomainPruner) {
    searcher.greedySearchQuickTest(testStructs, wght, pruner);
    for (ex <- testStructs) {
    	val initState = SearchBasedLearner.getInitStateWithUnaryScore(ex, pruner, true);
    	val initWithMusk = searcher.prunedActionSpaceMusk(ex, initState);
    	//val returnedState = searcher.beamSearch(ex, initWithMusk, bsize, wght, false, false);
      val (returnedState, bestRnk) = searcher.searchWithRandomRestart(ex, initWithMusk, bsize, wght, false, false, restrt);
    	ex.currentOutput = returnedState.output;
    }
  }
  
  def testGroundTruthBeamSearch(testStructs: Seq[AceJointTaskExample], bsize: Int, restrt: Int, wght: Array[Double], searcher: SearchBasedLearner, pruner: StaticDomainPruner) {
    for (ex <- testStructs) {
      val initState = SearchBasedLearner.getInitStateWithUnaryScore(ex, pruner, true);
    	//val gdinit = SearchBasedLearner.getGoldInitState(ex);
      val gdinit = searcher.constructGoldMuskNoPredict(ex, initState);
      val returnedState = searcher.beamSearch(ex, gdinit, bsize, wght, true, false);
    	ex.currentOutput = returnedState.output;
    }
  }
  
/*
 def testNerWikiFactorOnly(testStructs: Seq[AceJointTaskExample], wght: Array[Double], searcher: NerWikiOnlyLearner, unaryWght: Array[Double]) {
      for (ex <- testStructs) {
        //val initState = SearchBasedLearner.getRandomInitState(ex);
        val initState = SearchBasedLearner.getInitStateWithUnaryScore(ex, unaryWght);
        val returned = searcher.nerwikiTravialInference(ex,  wght, false);
        ex.currentOutput = returned;
        //ex.printTaskOutput(); 
      }
  }
*/
  
  def evaluateAceStructsJoint(testStructs: Seq[AceJointTaskExample], 
                              goldWikification: HashMap[String, DocWikiAnnots]) {
    val multiStructs = testStructs.map( struct => { struct.toMultiTaskStructs() } );
    SingleTaskStructTesting.evaluateAceStructs(multiStructs, goldWikification);
  }
  
  
  def testPrunedOracle(testStructs: Seq[AceJointTaskExample]) {
     for (ex <- testStructs) {
       for (i <- 0 until ex.totalSize) {
         val corrvIdxs = ex.getVariableDomainSizeGivenIndex(i)
       }
    }
  }
  //////////////////////////////////////////////////////

  
  def runIndependentTaskTraining(trainExs: Seq[AceJointTaskExample],
                                 testExs: Seq[AceJointTaskExample],
                                 featIndexer: Indexer[String],
                                 goldWikification: HashMap[String, DocWikiAnnots]) = {
    
    val trainStructs = (new ArrayBuffer[AceMultiTaskExample]()) ++ (trainExs.map(struct => { struct.toMultiTaskStructs() }));
    val testStructs = (new ArrayBuffer[AceMultiTaskExample]()) ++ (testExs.map(struct => { struct.toMultiTaskStructs() }));

    // Learning!
    val weight = SingleTaskStructTesting.structurePerceptrion(trainStructs, featIndexer, testStructs);
    
    // test as oracle?
    //testStructuraAllTaskOracle(testStructs, featIndexer.size());
    
    // test!
    val histgram = new HistgramRecord();
    SingleTaskStructTesting.testStructuralAllTasks(testStructs, weight, histgram);
    histgram.printHistgram();
    
    // evaluate
    SingleTaskStructTesting.evaluateAceStructs(testStructs, goldWikification);
    
    // return 
    weight;
  }
  
  // train the pruner by taking the AceMultiTaskExample as input
  def runUnaryPrunerTraining(trains: Seq[AceMultiTaskExample],
                             tests: Seq[AceMultiTaskExample],
                             featIndexer: Indexer[String],
                             goldWikification: HashMap[String, DocWikiAnnots]) = {
    
    val trainStructs = (new ArrayBuffer[AceMultiTaskExample]()) ++ trains;
    val testStructs = (new ArrayBuffer[AceMultiTaskExample]()) ++ tests
    // Learning!
    val weight = SingleTaskStructTesting.structurePerceptrion(trainStructs, featIndexer, testStructs);
    //val weight = SingleTaskStructTesting.trainAdagrad(trainStructs.toSeq, testStructs.toSeq, featIndexer.size, Driver.eta, Driver.reg,  Driver.batchSize, Driver.numItrs)

    // test as oracle?
    //testStructuraAllTaskOracle(testStructs, featIndexer.size());
    
    // test!
    val histgram = new HistgramRecord();
    SingleTaskStructTesting.testStructuralAllTasks(testStructs, weight, histgram);
    histgram.printHistgram();
    
    // evaluate train!
    val histgramTrain = new HistgramRecord();
    SingleTaskStructTesting.testStructuralAllTasks(trainStructs, weight, histgramTrain);
    SingleTaskStructTesting.evaluateAceStructs(trainStructs, goldWikification);
    println("===============================\n===============================\n===============================\n");

    // evaluate
    SingleTaskStructTesting.evaluateAceStructs(testStructs, goldWikification);

    // return 
    weight;
  }
  
  ////////////////////////////////////
/*
  def runStandardBerkTrainTest() {
    
    val trainDataPath = "data/ace05/train";
    val devDataPath = "data/ace05/dev";
    val testDataPath = "data/ace05/test";
    val wikiPath = "data/ace05/ace05-all-conll-wiki"
    val wikiDBPath = "models/wiki-db-ace.ser.gz"
    
    Driver.trainPath = trainDataPath;
    Driver.testPath = testDataPath;
    Driver.modelPath = "./berk-ace05-joint.ser.gz";
    
    // independent model only
    Driver.corefNerFeatures = "";
    //Driver.wikiNerFeatures = "";
    Driver.corefWikiFeatures = "";
    
    // Resources needed for document assembly: number/gender computer, NER marginals, coref models and mapping of documents to folds
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    
    // Load coref models
    val corefPruner = CorefPruner.buildPruner(Driver.pruningStrategy)
    val jointDocs = EntitySystem.preprocessACEDocsForTrainEval(trainDataPath, -1, mentionPropertyComputer, corefPruner, wikiPath, true);

    ///////////////////////
    // Build the featurizer, which involves building specific featurizers for each task
    val featureIndexer = new Indexer[String]();
    val maybeBrownClusters = if (Driver.brownPath != "") Some(BrownClusterInterface.loadBrownClusters(Driver.brownPath, 0)) else None
    val nerFeaturizer = MCNerFeaturizer(Driver.nerFeatureSet.split("\\+").toSet, featureIndexer, MCNerFeaturizer.StdLabelIndexer, jointDocs.flatMap(_.rawDoc.words), None, maybeBrownClusters)
    val jointFeaturizer = EntitySystem.buildFeaturizerShared(jointDocs.map(_.docGraph.corefDoc), featureIndexer, nerFeaturizer, maybeBrownClusters);
    val maybeWikipediaInterface: Option[WikipediaInterface] = Some(GUtil.load(wikiDBPath).asInstanceOf[WikipediaInterface]);
    
    ///////////////////////
    // Cache features
    val fgfAce = new FactorGraphFactoryACE(jointFeaturizer, maybeWikipediaInterface);
    val computer = new JointComputerShared(fgfAce);
    jointDocs.foreach(jointDoc => {
      fgfAce.getDocFactorGraph(jointDoc, true, true, true, PairwiseLossFunctions(Driver.lossFcn), JointLossFcns.nerLossFcn, JointLossFcns.wikiLossFcn);
      fgfAce.getDocFactorGraph(jointDoc, false, true, true, PairwiseLossFunctions(Driver.lossFcn), JointLossFcns.nerLossFcn, JointLossFcns.wikiLossFcn);
    });
    PairwiseIndexingFeaturizer.printFeatureTemplateCounts(featureIndexer)
    Logger.logss(featureIndexer.size + " total features");
    
    val finalWeights = new GeneralTrainer[JointDocACE].trainAdagrad(jointDocs, computer, featureIndexer.size, Driver.eta.toFloat, Driver.reg.toFloat, Driver.batchSize, Driver.numItrs);
    //val finalWeights = new GeneralTrainerCopy[JointDocACE].trainAdagrad(jointDocs, computer, featureIndexer.size, Driver.eta.toFloat, Driver.reg.toFloat, Driver.batchSize, Driver.numItrs);
    //val finalWeights = new MyTrainer[JointDocACE].train(jointDocs, computer, featureIndexer.size, Driver.eta.toFloat, Driver.reg.toFloat, Driver.batchSize, Driver.numItrs);
    
    val model = new JointPredictorACE(jointFeaturizer, finalWeights, corefPruner).pack;
    if (Driver.modelPath != "") GUtil.save(model, Driver.modelPath);
    
    ///////////////////////
    // Evaluation of each part of the model
    // Build dev docs
    val jointDevDocs = EntitySystem.preprocessACEDocsForTrainEval(testDataPath, -1, mentionPropertyComputer, corefPruner, wikiPath, false);
    val wikiLabelsInTrain: Set[String] = jointDocs.flatMap(_.goldWikiChunks.flatMap(_.flatMap(_.label)).toSet).toSet;
    model.decodeWriteOutputEvaluate(jointDevDocs, maybeWikipediaInterface, Driver.doConllPostprocessing, wikiLabelsInTrain)
  }
*/
  
  def runStandardBerkTrainTestWithPruner() {
    
    val trainDataPath = "data/ace05/train";
    val devDataPath = "data/ace05/dev";
    val testDataPath = "data/ace05/test";
    val wikiPath = "data/ace05/ace05-all-conll-wiki"
    val wikiDBPath = "models/wiki-db-ace.ser.gz"
    
    Driver.trainPath = trainDataPath;
    Driver.testPath = testDataPath;
    Driver.modelPath = "./berk-ace05-joint.ser.gz";
    
    // independent model only
    //Driver.corefNerFeatures = "";
    //Driver.wikiNerFeatures = "";
    //Driver.corefWikiFeatures = "";
    
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentTypePredictor = AceMentionTypePredictor.TrainMentionTypeACE();
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer), Some(mentTypePredictor));
    
    // Load coref models
    //val existCorefPrunerDumpPath = "models:./corefpruner-ace.ser.gz:-5";
    val existCorefPrunerDumpPath = "models:./corefpruner-ace.ser.gz:-5";
    val corefPruner = CorefPruner.buildPruner(existCorefPrunerDumpPath)
    val jointDocs = EntitySystem.preprocessACEDocsForTrainEval(trainDataPath, -1, mentionPropertyComputer, corefPruner, wikiPath, true);
    val jointDevDocs = EntitySystem.preprocessACEDocsForTrainEval(testDataPath, -1, mentionPropertyComputer, corefPruner, wikiPath, false);

    ///////////////////////
    // Build the featurizer, which involves building specific featurizers for each task
    val featureIndexer = new Indexer[String]();
    val maybeBrownClusters = if (Driver.brownPath != "") Some(BrownClusterInterface.loadBrownClusters(Driver.brownPath, 0)) else None
    val nerFeaturizer = MCNerFeaturizer(Driver.nerFeatureSet.split("\\+").toSet, featureIndexer, MCNerFeaturizer.StdLabelIndexer, jointDocs.flatMap(_.rawDoc.words), None, maybeBrownClusters)
    val jointFeaturizer = EntitySystem.buildFeaturizerShared(jointDocs.map(_.docGraph.corefDoc), featureIndexer, nerFeaturizer, maybeBrownClusters);
    val maybeWikipediaInterface: Option[WikipediaInterface] = Some(GUtil.load(wikiDBPath).asInstanceOf[WikipediaInterface]);
    
    // construct coref doc list
    //val trainCorefDocs = jointDocs.map { jdoc => jdoc.docGraph.corefDoc }; 
    //val testCorefDocs = jointDevDocs.map { jdoc => jdoc.docGraph.corefDoc };
    
    ///////////////////////
    // Cache features
    val domainModel = GraphPrunerTrainer.trainPruner(numberGenderComputer, mentionPropertyComputer, maybeBrownClusters, trainDataPath, testDataPath);
    //val domainModel = new GraphPrunerModelACE(new NerPrunerModelACE(null, null, null));
    //throw new RuntimeException("sstop!");
    
    val fgfPruner = new PrunedGraphFactoryACE(jointFeaturizer, maybeWikipediaInterface, domainModel, true);//new FactorGraphFactoryACE(jointFeaturizer, maybeWikipediaInterface);
    val computer = new JointComputerShared(fgfPruner);
    jointDocs.foreach(jointDoc => {
      fgfPruner.getDocFactorGraph(jointDoc, true, true, true, PairwiseLossFunctions(Driver.lossFcn), JointLossFcns.nerLossFcn, JointLossFcns.wikiLossFcn);
      fgfPruner.getDocFactorGraph(jointDoc, false, true, true, PairwiseLossFunctions(Driver.lossFcn), JointLossFcns.nerLossFcn, JointLossFcns.wikiLossFcn);
    });
    PairwiseIndexingFeaturizer.printFeatureTemplateCounts(featureIndexer)
    Logger.logss(featureIndexer.size + " total features");
    
    val finalWeights = new GeneralTrainer[JointDocACE].trainAdagrad(jointDocs, computer, featureIndexer.size, Driver.eta.toFloat, Driver.reg.toFloat, Driver.batchSize, Driver.numItrs);
    //val finalWeights = new GeneralTrainerCopy[JointDocACE].trainAdagrad(jointDocs, computer, featureIndexer.size, Driver.eta.toFloat, Driver.reg.toFloat, Driver.batchSize, Driver.numItrs);
    //val finalWeights = new MyTrainer[JointDocACE].train(jointDocs, computer, featureIndexer.size, Driver.eta.toFloat, Driver.reg.toFloat, Driver.batchSize, Driver.numItrs);
    
    //val model = new JointPredictorACE(jointFeaturizer, finalWeights, corefPruner).pack;
    val model = new JointPrunedPredictorACE(jointFeaturizer, finalWeights, corefPruner, domainModel);//.pack;
    //if (Driver.modelPath != "") GUtil.save(model, Driver.modelPath);
    
    ///////////////////////
    // Evaluation of each part of the model
    // Build dev docs
    //val jointDevDocs = EntitySystem.preprocessACEDocsForTrainEval(testDataPath, -1, mentionPropertyComputer, corefPruner, wikiPath, false);
    val wikiLabelsInTrain: Set[String] = jointDocs.flatMap(_.goldWikiChunks.flatMap(_.flatMap(_.label)).toSet).toSet;
    model.decodeWriteOutputEvaluate(jointDevDocs, maybeWikipediaInterface, Driver.doConllPostprocessing, wikiLabelsInTrain)
    
    /////
    //println("==Train==================================\n");
    //model.decodeWriteOutputEvaluate(jointDocs, maybeWikipediaInterface, Driver.doConllPostprocessing, wikiLabelsInTrain)
  }
  
  
  def runSelfConstructBerkeley() {
    
    val trainDataPath = "data/ace05/train";
    val devDataPath = "data/ace05/dev";
    val testDataPath = "data/ace05/test";
    val wikiPath = "data/ace05/ace05-all-conll-wiki"
    val wikiDBPath = "models/wiki-db-ace.ser.gz"
    
    // independent model only
    //Driver.corefNerFeatures = "";
    //Driver.wikiNerFeatures = "";
    //Driver.corefWikiFeatures = "";
    
    
    val featIndexer = new Indexer[String]();
    
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    val assembler = CorefDocAssembler(Language.ENGLISH, true); //use gold mentions
    val trainDocs = ConllDocReader.loadRawConllDocsWithSuffix(trainDataPath, -1, "", Language.ENGLISH);
    val trainCorefDocs = trainDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    
    val maybeBrownClusters = if (Driver.brownPath != "") Some(BrownClusterInterface.loadBrownClusters(Driver.brownPath, 0)) else None;
    val nerFeaturizer = MCNerFeaturizer(Driver.nerFeatureSet.split("\\+").toSet, featIndexer, MCNerFeaturizer.StdLabelIndexer, trainDocs.flatMap(_.words), None, maybeBrownClusters);
    
    
    // Read in gold Wikification labels
    val goldWikification = WikiAnnotReaderWriter.readStandoffAnnotsAsCorpusAnnots(wikiPath)
    // Read in the title given surface database
    val wikiDB = GUtil.load(wikiDBPath).asInstanceOf[WikipediaInterface];
    val jqdcomputer = new JointQueryDenotationChoiceComputer(wikiDB, featIndexer);
    
    val queryCounts: Option[QueryCountsBundle] = None;
    val lexicalCounts = LexicalCountsBundle.countLexicalItems(trainCorefDocs, Driver.lexicalFeatCutoff);
    val semClasser: Option[SemClasser] = Driver.semClasserType match {
      case "basic" => Some(new BasicWordNetSemClasser);
      case e => throw new RuntimeException("Other semclassers not implemented");
    }
    val trainDocGraphs = trainCorefDocs.map(new DocumentGraph(_, true));

    CorefStructUtils.preprocessDocsCacheResources(trainDocGraphs);
    val berkPruner = CorefPruner.buildPrunerArguments(Driver.pruningStrategy, trainDataPath, -1);//.buildPruner(Driver.pruningStrategy);
    berkPruner.pruneAll(trainDocGraphs);

    featIndexer.getIndex(PairwiseIndexingFeaturizerJoint.UnkFeatName);
    val featureSetSpec = FeatureSetSpecification(Driver.pairwiseFeats, Driver.conjScheme, Driver.conjFeats, Driver.conjMentionTypes, Driver.conjTemplates);
    val basicFeaturizer = new PairwiseIndexingFeaturizerJoint(featIndexer, featureSetSpec, lexicalCounts, queryCounts, semClasser);
    val featurizerTrainer = new CorefFeaturizerTrainer();

    // joint featurizer
    val jointFeaturizer = new JointFeaturizerShared[MCNerFeaturizer](basicFeaturizer, nerFeaturizer, maybeBrownClusters, Driver.corefNerFeatures, Driver.corefWikiFeatures, Driver.wikiNerFeatures, featIndexer);

    
    val jointDocs = EntitySystem.preprocessACEDocsForTrainEval(trainDataPath, -1, mentionPropertyComputer, berkPruner, wikiPath, true);
    
    ///////////////////////
    // Cache features
    val fgfAce = new FactorGraphFactoryACE(jointFeaturizer, Some(wikiDB));
    val computer = new JointComputerShared(fgfAce);
    jointDocs.foreach(jointDoc => {
      fgfAce.getDocFactorGraph(jointDoc, true, true, true, PairwiseLossFunctions(Driver.lossFcn), JointLossFcns.nerLossFcn, JointLossFcns.wikiLossFcn);
      fgfAce.getDocFactorGraph(jointDoc, false, true, true, PairwiseLossFunctions(Driver.lossFcn), JointLossFcns.nerLossFcn, JointLossFcns.wikiLossFcn);
    });
    PairwiseIndexingFeaturizer.printFeatureTemplateCounts(featIndexer)
    Logger.logss(featIndexer.size + " total features");
    
    val finalWeights = new GeneralTrainer[JointDocACE].trainAdagrad(jointDocs, computer, featIndexer.size, Driver.eta.toFloat, Driver.reg.toFloat, Driver.batchSize, Driver.numItrs);
    //val finalWeights = new GeneralTrainerCopy[JointDocACE].trainAdagrad(jointDocs, computer, featureIndexer.size, Driver.eta.toFloat, Driver.reg.toFloat, Driver.batchSize, Driver.numItrs);
    //val finalWeights = new MyTrainer[JointDocACE].train(jointDocs, computer, featIndexer.size, Driver.eta.toFloat, Driver.reg.toFloat, Driver.batchSize, Driver.numItrs);
    
    val model = new JointPredictorACE(jointFeaturizer, finalWeights, berkPruner).pack;
    if (Driver.modelPath != "") GUtil.save(model, Driver.modelPath);
    
    ///////////////////////
    // Evaluation of each part of the model
    // Build dev docs
    val jointDevDocs = EntitySystem.preprocessACEDocsForTrainEval(testDataPath, -1, mentionPropertyComputer, berkPruner, wikiPath, false);
    val wikiLabelsInTrain: Set[String] = jointDocs.flatMap(_.goldWikiChunks.flatMap(_.flatMap(_.label)).toSet).toSet;
    
    val jointTrainDocs = EntitySystem.preprocessACEDocsForTrainEval(trainDataPath, -1, mentionPropertyComputer, berkPruner, wikiPath, false);
    model.decodeWriteOutputEvaluate(jointTrainDocs, Some(wikiDB), Driver.doConllPostprocessing, wikiLabelsInTrain)
    println("===================================\n===================================\n===================================\n");
    model.decodeWriteOutputEvaluate(jointDevDocs, Some(wikiDB), Driver.doConllPostprocessing, wikiLabelsInTrain)
  }

}