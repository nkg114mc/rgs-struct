package berkeleyentity.prunedomain

import scala.collection.JavaConverters._
import berkeleyentity.joint.JointDocACE
import berkeleyentity.ner.MCNerFeaturizer
import berkeleyentity.wiki.Query
import berkeleyentity.wiki.WikipediaInterface
import scala.collection.mutable.ArrayBuffer
import berkeleyentity.sem.BrownClusterInterface
import berkeleyentity.oregonstate.tmpExample
import berkeleyentity.coref.MentionPropertyComputer
import berkeleyentity.coref.NumberGenderComputer
import java.io.PrintWriter
import berkeleyentity.lang.Language
import berkeleyentity.ConllDocReader
import berkeleyentity.coref.CorefDocAssembler
import edu.berkeley.nlp.util.Logger
import berkeleyentity.ranking.UMassRankLib
import berkeleyentity.Driver
import berkeleyentity.coref.CorefDoc
import berkeleyentity.oregonstate.NerTesting
import edu.berkeley.nlp.futile.fig.basic.Indexer
import berkeleyentity.ner.MCNerExample
//import berkeleyentity.oregonstate.NerDecision
import scala.collection.mutable.HashMap
import ciir.umass.edu.learning.SparseDataPoint
import ciir.umass.edu.learning.DataPoint
import util.control.Breaks._
import berkeleyentity.wiki.QueryChoiceComputer
import berkeleyentity.sem.QueryCountsBundle
import berkeleyentity.sem.BasicWordNetSemClasser
import berkeleyentity.coref.FeatureSetSpecification
import berkeleyentity.joint.JointFeaturizerShared
import berkeleyentity.sem.SemClasser
import berkeleyentity.coref.LexicalCountsBundle
import berkeleyentity.coref.PairwiseIndexingFeaturizerJoint

// for synthetic pruner only
//class WeightedIndex(val index: Int,
//                    val weight: Double) {
//  
//}

class FactoryGraphPrunerACE(val doc: JointDocACE,
                            val wikiDB: Option[WikipediaInterface],
                            val model: GraphPrunerModelACE,
                            val gold: Boolean,
                            val training: Boolean) {

  val docGraph = doc.docGraph;
  //val prunerQueryComputer = new QueryChoiceComputer(wikiDB.get, featurizer.indexer);
 
  // indicator of decision pruning (1 means pruning)
  val corefPruneIndicator = new Array[Int](docGraph.size());
  val nerPruneIndicator = new Array[Int](docGraph.size());
  val wikiPruneIndicator = new Array[Int](docGraph.size());
    
  val corefDomain = new Array[Array[Int]](docGraph.size());
  val nerDomain = new Array[Array[String]](docGraph.size());
  val wikiDomain = new Array[Array[String]](docGraph.size()); //val queryDomain = new Array[Array[Query]](docGraph.size());
  
  
  def initIndicators() {
    for (i <- 0 until docGraph.size) {
      // coref
      corefPruneIndicator(i) = 0;
      // ner
      nerPruneIndicator(i) = 0;
      // wiki
      wikiPruneIndicator(i) = 0;
    }
  }
  
  // init indicators...
  initIndicators();
  
  def noDomainPruning() {
    for (i <- 0 until docGraph.size) {
      // coref
      val cDomnArr =  docGraph.getPrunedDomain(i, gold);
      corefDomain(i) = cDomnArr;
      // ner
      val nDomnArr = MCNerFeaturizer.StdLabelIndexer.getObjects.asScala.toArray;
      nerDomain(i) = nDomnArr;
      // wiki
      val queries = Query.extractQueriesBest(docGraph.getMention(i), true);
      val queryDisambigs = queries.map(wikiDB.get.disambiguateBestGetAllOptions(_)); //val queryFeatures = qcComputer.featurizeQueries(queries, addToIndexer);
      val rawQueryDenotations = Query.extractDenotationSetWithNil(queries, queryDisambigs, Driver.maxNumWikificationOptions);//prunerQueryComputer.extractDenotationSetWithNil(queries, queryDisambigs, Driver.maxNumWikificationOptions);
      wikiDomain(i) = rawQueryDenotations.toArray;
    }
    println("Doesn't run domain pruning at all ...");
  }

  
  ///////////////////////////////////////
  //////// Run actual pruning ///////////
  ///////////////////////////////////////
  def doDomainPruning() {
    
    
    
    // coref
    //doCorefDomainPruning();
    // ner
    //doNerDomainPruning(); 
    // wiki
    
    
    
    
    
    
    
    println("Runing really domain pruning ...");
  }
  

/*
  ///// Coref
  def doCorefDomainPruning() {
    // no pruning
    for (i <- 0 until docGraph.size) {
      // coref
      val cDomnArr =  docGraph.getPrunedDomain(i, gold);
      corefDomain(i) = cDomnArr;
    }
  }
  
  ///// Wiki
*/
  
  //////
 
  
  //// getters
  
  def getCorefDomain(mentionIdx: Int, useGold: Boolean) = {
    corefDomain(mentionIdx);
  }
  
  def getNerDomain(mentionIdx: Int, useGold: Boolean) = {
    nerDomain(mentionIdx);
  }
  
  def getWikiDomain(mentionIdx: Int, useGold: Boolean) = {
    wikiDomain(mentionIdx);
  }

}


// include the featurizer
class GraphPrunerModelACE(pfeaturizer: PrunerFeaturizer) {
  

	def featurizeCoref() {

	}

	def featurizeNer() {

	}

	def featurizeWiki() {

	}
  
}

object GraphPrunerModelACE {
  


 
  
}




















/////////////////////////////////////////
/////////////////////////////////////////
/////////////////////////////////////////

class GraphPrunerTrainer {
  
}

object GraphPrunerTrainer {

  
    def trainPruner(numberGenderComputer: NumberGenderComputer,
                    mentionPropertyComputer: MentionPropertyComputer,
                    maybeBrownClusters: Option[Map[String, String]],
                    trainPath: String,
                    testPath: String) = {
      
    	val assembler = CorefDocAssembler(Language.ENGLISH, true); //use gold mentions
    	val trainDocs = ConllDocReader.loadRawConllDocsWithSuffix(trainPath, -1, "", Language.ENGLISH);
    	val trainCorefDocs = trainDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));

    	val testDocs = ConllDocReader.loadRawConllDocsWithSuffix(testPath, -1, "", Language.ENGLISH);
    	val testCorefDocs = testDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));

      val md = trainPrunerGivenDocs(numberGenderComputer, mentionPropertyComputer, maybeBrownClusters,
                                    trainCorefDocs, testCorefDocs);
      md;
    }
  
	def trainPrunerGivenDocs(numberGenderComputer: NumberGenderComputer,
			mentionPropertyComputer: MentionPropertyComputer,
			maybeBrownClusters: Option[Map[String, String]],
			trainCorefDocs: Seq[CorefDoc],
			testCorefDocs: Seq[CorefDoc]) = {
			//devCorefDocs: Seq[CorefDoc]) = {

    // ner initializer training
    //val nerMdl = trainNerPruner(numberGenderComputer, mentionPropertyComputer, maybeBrownClusters,
    //                            trainCorefDocs, testCorefDocs);//trainCorefDocs, testCorefDocs, devCorefDocs);
    
	  val prunFeaturizer = PrunerFeaturizer.constructFeaturizer(trainCorefDocs);
    
	   
    //val totalMdl = new GraphPrunerModelACE();
    //totalMdl;
	  null;
	}
/*
  def trainNerPruner(numberGenderComputer: NumberGenderComputer,
                      mentionPropertyComputer: MentionPropertyComputer,
                      maybeBrownClusters: Option[Map[String, String]],
                      trainCorefDocs: Seq[CorefDoc],
                      testCorefDocs: Seq[CorefDoc]) = {
                      
                      //devCorefDocs: Seq[CorefDoc]) = {

    val trainDocs = trainCorefDocs.map { cdoc => cdoc.rawDoc };
    val testDocs = testCorefDocs.map { cdoc => cdoc.rawDoc };
    //val devDocs = devCorefDocs.map { cdoc => cdoc.rawDoc };

    val trainExs = NerTesting.extractExamples(trainCorefDocs);
    println("ACE NER chunks: " + trainExs.size);
    
    // Extract features
    val featIndexer = new Indexer[String];
    val nerFeaturizer = MCNerFeaturizer(Driver.nerFeatureSet.split("\\+").toSet, featIndexer, MCNerFeaturizer.StdLabelIndexer, trainDocs.flatMap(_.words), None, maybeBrownClusters);

    
    var allTrains = constructTmpExmp(trainExs, nerFeaturizer, true);
    println("Ner feature size = " + nerFeaturizer.featureIndexer.size());
    
    /////////////// for testing & validating ///////////////
    val testExs = NerTesting.extractExamples(testCorefDocs);
    val testEmpExs = constructTmpExmp(testExs, nerFeaturizer, false);
      
    //val devExs = NerTesting.extractExamples(devCorefDocs);
    //val devEmpExs = constructTmpExmp(devExs, nerFeaturizer, false);


    
    // learn!
    val wght = NerTesting.structurePerceptrion(allTrains, featIndexer, testEmpExs, 100);
    //val wght = multiClassSVM(allTrains, featIndexer, testEmpExs);
    
    // test error ranking
    //val errRankModel = "errfinder/ner_error_ace05_lambdamart.txt";
    //val errRankModel = "errfinder/ner_error_ace05_lambdamart_dev.txt";
    //val errRankModel = "errfinder/ner_error_ace05_lambdamart_2.txt";
    val errRankModel = "errfinder/ner_error_ace05_lambdamart_w0_global.txt";
    val ranker = new UMassRankLib();
    ranker.loadModelFile(errRankModel);
    //val trainRelearnExs = NerErrorFinder.computeUpperBoundWithErrFinder(allTrains, wght, ranker, false);
    //val testRelearnExs = NerErrorFinder.computeUpperBoundWithErrFinder(testEmpExs, wght, ranker, false);
    //NerErrorFinder.computeUpperBoundGlobalFinder(allTrains, wght, ranker, false);
    //NerErrorFinder.computeUpperBoundGlobalFinder(testEmpExs, wght, ranker, false);


    Logger.logss("Ner Pruner Training Done !");

    val nerModel = new NerPrunerModelACE(nerFeaturizer,  wght, ranker);
    nerModel;
  }
  
  def constructTmpExmp(mcExmps: ArrayBuffer[MCNerExample], 
                       nerFeaturizer: MCNerFeaturizer,
                       addToIdxer: Boolean) = {
    var tmpExs = new ArrayBuffer[tmpExample]();
    for (ex <- mcExmps) {
      val featEachLabel = nerFeaturizer.featurize(ex, addToIdxer);
      val thisTmp = new tmpExample(ex, featEachLabel);
      tmpExs += (thisTmp); // add to list
    }
    tmpExs;
  }
*/

}
