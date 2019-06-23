package berkeleyentity.randsearch

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
import java.util.ArrayList
import java.io.PrintWriter
import java.nio.file.Files
import java.nio.file.CopyOption._
import java.io.File
import scala.util.control.Breaks._
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import berkeleyentity.sem.BrownClusterInterface
import berkeleyentity.oregonstate.CorefStructUtils
import berkeleyentity.oregonstate.SingleTaskStructTesting
import berkeleyentity.coref.MentionPropertyComputer
import berkeleyentity.sem.QueryCountsBundle
import berkeleyentity.coref.NumberGenderComputer
import berkeleyentity.oregonstate.pruner.InitPruner
import berkeleyentity.oregonstate.CorefAdditionalFeature
import berkeleyentity.sem.BasicWordNetSemClasser
import berkeleyentity.coref.FeatureSetSpecification
import berkeleyentity.GUtil
import berkeleyentity.oregonstate.MCNerAddonFeaturizer
import berkeleyentity.wiki.WikipediaInterface
import berkeleyentity.wiki.JointQueryDenotationChoiceComputer
import berkeleyentity.oregonstate.NerTesting
import berkeleyentity.ner.MCNerFeaturizer
import berkeleyentity.oregonstate.pruner.LinearDomainPruner
import berkeleyentity.lang.Language
import berkeleyentity.oregonstate.AceMultiTaskExample
import berkeleyentity.joint.JointFeaturizerShared
import berkeleyentity.sem.SemClasser
import berkeleyentity.ConllDocReader
import berkeleyentity.oregonstate.StructuralSVMLearner
import berkeleyentity.coref.CorefDocAssembler
import berkeleyentity.ilp.HistgramRecord
import berkeleyentity.oregonstate.pruner.NonePruner
import berkeleyentity.coref.LexicalCountsBundle
import berkeleyentity.oregonstate.ZobristKeys
import berkeleyentity.oregonstate.AceJointTaskExample
import berkeleyentity.oregonstate.AceMentionTypePredictor
import berkeleyentity.wiki.WikiAnnotReaderWriter
import berkeleyentity.oregonstate.NerAdditionalFeature
import berkeleyentity.oregonstate.pruner.StaticDomainPruner
import berkeleyentity.coref.CorefFeaturizerTrainer
import berkeleyentity.coref.PairwiseIndexingFeaturizerJoint
import berkeleyentity.oregonstate.SearchBasedLearner
import scala.util.Random
import edu.berkeley.nlp.futile.fig.basic.Indexer
import berkeleyentity.Driver
import berkeleyentity.coref.CorefPruner
import ml.dmlc.xgboost4j.scala.Booster
import ml.dmlc.xgboost4j.scala.XGBoost
import berkeleyentity.sem.BrownClusterInterface
import berkeleyentity.oregonstate.CorefStructUtils
import berkeleyentity.oregonstate.SingleTaskStructTesting
import berkeleyentity.coref.MentionPropertyComputer
import berkeleyentity.sem.QueryCountsBundle
import berkeleyentity.coref.NumberGenderComputer
import berkeleyentity.oregonstate.pruner.InitPruner
import berkeleyentity.oregonstate.CorefAdditionalFeature
import berkeleyentity.sem.BasicWordNetSemClasser
import berkeleyentity.coref.FeatureSetSpecification
import berkeleyentity.GUtil
import berkeleyentity.oregonstate.MCNerAddonFeaturizer
import berkeleyentity.wiki.WikipediaInterface
import berkeleyentity.wiki.JointQueryDenotationChoiceComputer
import berkeleyentity.oregonstate.NerTesting
import berkeleyentity.ner.MCNerFeaturizer
import berkeleyentity.oregonstate.pruner.LinearDomainPruner
import berkeleyentity.lang.Language
import berkeleyentity.oregonstate.AceMultiTaskExample
import berkeleyentity.joint.JointFeaturizerShared
import berkeleyentity.sem.SemClasser
import berkeleyentity.ConllDocReader
import berkeleyentity.oregonstate.StructuralSVMLearner
import berkeleyentity.coref.CorefDocAssembler
import berkeleyentity.ilp.HistgramRecord
import berkeleyentity.oregonstate.pruner.NonePruner
import berkeleyentity.coref.LexicalCountsBundle
import berkeleyentity.oregonstate.ZobristKeys
import berkeleyentity.oregonstate.AceJointTaskExample
import berkeleyentity.oregonstate.AceMentionTypePredictor
import berkeleyentity.wiki.WikiAnnotReaderWriter
import berkeleyentity.oregonstate.NerAdditionalFeature
import berkeleyentity.oregonstate.pruner.StaticDomainPruner
import berkeleyentity.coref.CorefFeaturizerTrainer
import berkeleyentity.coref.PairwiseIndexingFeaturizerJoint
import berkeleyentity.oregonstate.SearchBasedLearner
import berkeleyentity.coref.DocumentGraph
import berkeleyentity.oregonstate.JointTaskStructTesting
import berkeleyentity.wiki.DocWikiAnnots
import berkeleyentity.oregonstate.IndepVariable
import berkeleyentity.oregonstate.AceSingleTaskStructExample
import berkeleyentity.oregonstate.VarValue
import berkeleyentity.coref.CorefDoc
import berkeleyentity.coref.PairwiseIndexingFeaturizer


object CorefRandSearch {

  def main(args: Array[String]) {

	  // set some configs
	  Driver.numberGenderDataPath = "../coref/berkfiles/data/gender.data";
	  Driver.brownPath = "../coref/berkfiles/data/bllip-clusters";
	  Driver.useGoldMentions = true;
	  Driver.doConllPostprocessing = false;
	  //Driver.pruningStrategy = "build:./corefpruner-ace.ser.gz:-5:5";
    Driver.lossFcn = "customLoss-1-1-1";

	  Driver.corefNerFeatures = "indicators+currlex+antlex";
	  Driver.wikiNerFeatures = "categories+infoboxes+appositives";
	  Driver.corefWikiFeatures = "basic+lastnames";

	  trainCorefStructuralTask();
  }
  
  def trainCorefStructuralTask() {
    
    val trainDataPath = "../coref/berkfiles/data/ace05/train";
    val devDataPath = "../coref/berkfiles/data/ace05/dev";
    val testDataPath = "../coref/berkfiles/data/ace05/test";
    //val testDataPath = devDataPath;

    //val trainDataPath = "../coref/berkfiles/data/ace05/train_1";
    //val testDataPath = "../coref/berkfiles/data/ace05/test_1";
    
    val wikiPath = "../coref/berkfiles/data/ace05/ace05-all-conll-wiki"
    val wikiDBPath = "../coref/berkfiles/models/wiki-db-ace.ser.gz"
    
    val berkeleyCorefPrunerDumpPath = "models:../coref/berkfiles/corefpruner-ace.ser.gz:-5";
    //val berkeleyCorefPrunerDumpPath = "build:./corefpruner-ace.ser.gz:-5:5";
    

    val featIndexer = new Indexer[String]();
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
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
    
    // test examples
    val testDocGraphs = SingleTaskStructTesting.testGetdocgraphs(testDataPath, -1, mentionPropertyComputer);
    val testDocs = testDocGraphs.map { dg => dg.corefDoc };
    berkPruner.pruneAll(testDocGraphs);
    
    val testIndepExs = SingleTaskStructTesting.extractAllTaskExamples(testDocGraphs, 
                                           nerFeaturizer,
                                           goldWikification, wikiDB, false, jqdcomputer,
                                           basicFeaturizer);
    
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
////// Perform Domain Pruning Here ///////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

    val prunWghtFile = new File("pruner_develop_ace05.weight_full");

    val pwght = if (!prunWghtFile.exists()) {
      val pw = LinearDomainPruner.runIndependentTaskTraining(trainIndepExs, testIndepExs, featIndexer);
      StaticDomainPruner.savePrunerWeight(pw, prunWghtFile.getAbsolutePath);
      pw;  
    } else {
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
    val nonePruner = new NonePruner(new Array[Double](featIndexer.size));
    nonePruner.pruneDomainIndepBatch(trainIndepExs, -1, -1, -1);
    nonePruner.pruneDomainIndepBatch(testIndepExs, -1, -1, -1);
    
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // extract joint structural examples!
    val trainStructs = JointTaskStructTesting.extractJointExamples(trainDocGraphs, 
                                            nerFeaturizer,
                                            goldWikification, wikiDB, true, jqdcomputer,
                                            basicFeaturizer,
                                            jointFeaturier);
    // test examples
    val testStructs = JointTaskStructTesting.extractJointExamples(testDocGraphs, 
                                           nerFeaturizer,
                                           goldWikification, wikiDB, false, jqdcomputer,
                                           basicFeaturizer,
                                           jointFeaturier);

    println("FeatureCnt = " + featIndexer.size);
    
    val extendedPrunerWeight = SingleTaskStructTesting.extendWeight(pwght, featIndexer.size);
    val unaryPruner = new LinearDomainPruner(extendedPrunerWeight, 5, 7, 12);

    
    // Learning!
    val beamSize = 1;//40;
    val restart = 10;
    val IterNum = 10;//100;
    val zbKeys = new ZobristKeys(3000, 1000);
    val greedySearcher = new SearchBasedLearner(zbKeys);
    
    //val greedySearcher = new NerWikiOnlyLearner();
    //val weight = greedySearcher.runLearningDelayUpdate(trainStructs, featIndexer, testStructs, beamSize, restart, unaryPruner, IterNum);
    //val weight = greedySearcher.CSearchLearning(trainStructs, featIndexer, testStructs, unaryPruner, IterNum);

    val weight = StructuralSVMLearner.uiucStructLearning(trainStructs, featIndexer, testStructs, beamSize, restart, greedySearcher, unaryPruner);

    // test as oracle?
    //testPrunedOracle(testStructs, featIndexer.size());
    
    // test!
    //testHillClimb(testStructs, weight, greedySearcher, unaryPruner)
    testBeamSearch(testStructs, beamSize, restart, weight, greedySearcher, unaryPruner)
    //testGroundTruthBeamSearch(testStructs, beamSize, restart, weight, greedySearcher, unaryPruner);

    // evaluate
    evaluateAceStructsJoint(testStructs, goldWikification);
  }

  def getRandomWeight(len: Int): Array[Double] = {
    val w = new Array[Double](len);
    val rnd = new Random();
    for (i <- 0 until w.length) {
      w(i) = rnd.nextDouble(); 
    }
    w;
  }
  
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

  
}