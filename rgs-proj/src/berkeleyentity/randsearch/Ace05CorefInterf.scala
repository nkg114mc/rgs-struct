package berkeleyentity.randsearch

import collection.JavaConversions._
import berkeleyentity.coref.MentionPropertyComputer
import berkeleyentity.coref.NumberGenderComputer
import berkeleyentity.coref.CorefDocAssembler
import edu.berkeley.nlp.futile.fig.basic.Indexer
import berkeleyentity.lang.Language
import berkeleyentity.Driver
import berkeleyentity.oregonstate.AceSingleTaskStructExample
import berkeleyentity.coref.DocumentGraph
import berkeleyentity.coref.CorefDoc
import berkeleyentity.coref.PairwiseIndexingFeaturizer
import scala.collection.mutable.ArrayBuffer
import berkeleyentity.oregonstate.VarValue
import berkeleyentity.oregonstate.IndepVariable
import berkeleyentity.oregonstate.CorefStructUtils
import berkeleyentity.oregonstate.SingleTaskStructTesting
import berkeleyentity.sem.BasicWordNetSemClasser
import berkeleyentity.coref.FeatureSetSpecification
import berkeleyentity.ConllDocReader
import berkeleyentity.coref.LexicalCountsBundle
import berkeleyentity.coref.PairwiseIndexingFeaturizerJoint
import berkeleyentity.coref.CorefPruner
import java.util.ArrayList
import berkeleyentity.coref.OrderedClustering
import berkeleyentity.coref.CorefEvaluator

class Ace05CorefInterf {
 
    // set some configs
	  Driver.numberGenderDataPath = "../coref/berkfiles/data/gender.data";
	  Driver.brownPath = "../coref/berkfiles/data/bllip-clusters";
	  Driver.useGoldMentions = true;
	  Driver.doConllPostprocessing = false;

    Driver.lossFcn = "customLoss-1-1-1";

	  Driver.corefNerFeatures = "indicators+currlex+antlex";
	  Driver.wikiNerFeatures = "categories+infoboxes+appositives";
	  Driver.corefWikiFeatures = "basic+lastnames";
    
    val trainDataPath = "../coref/berkfiles/data/ace05/train";
    val devDataPath = "../coref/berkfiles/data/ace05/dev";
    val testDataPath = "../coref/berkfiles/data/ace05/test";
    
    //val trainDataPath = "../coref/berkfiles/data/ace05/train_1";
    //val testDataPath = "../coref/berkfiles/data/ace05/test_1";
    
    //Driver.pruningStrategy = "build:../coref/berkfiles/corefpruner-ace.ser.gz:-5:5";
    val berkeleyCorefPrunerDumpPath = "models:../coref/berkfiles/corefpruner-ace.ser.gz:-5";

    //val featIndexer = new Indexer[String]();
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    val assembler = CorefDocAssembler(Language.ENGLISH, true); //use gold mentions
    val trainDocs = ConllDocReader.loadRawConllDocsWithSuffix(trainDataPath, -1, "", Language.ENGLISH);
    val trainCorefDocs = trainDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    
    val lexicalCounts = LexicalCountsBundle.countLexicalItems(trainCorefDocs, Driver.lexicalFeatCutoff);
    val featureIndexer = new Indexer[String]();
    featureIndexer.getIndex(PairwiseIndexingFeaturizerJoint.UnkFeatName);
    val featureSetSpec = FeatureSetSpecification(Driver.pairwiseFeats, Driver.conjScheme, Driver.conjFeats, Driver.conjMentionTypes, Driver.conjTemplates);
    val basicFeaturizer = new PairwiseIndexingFeaturizerJoint(featureIndexer, featureSetSpec, lexicalCounts, None, Some(new BasicWordNetSemClasser));
    
        // load pruner
    val berkPruner = CorefPruner.buildPrunerArguments(berkeleyCorefPrunerDumpPath, trainDataPath, -1);
  
    def getTrainingDocGraphs(): ArrayList[DocumentGraph] = {
    		val trainDocGraphs = trainCorefDocs.map(new DocumentGraph(_, true));
    		CorefStructUtils.preprocessDocsCacheResources(trainDocGraphs);
    		berkPruner.pruneAll(trainDocGraphs);    		// run berkeley pruning
    		val jlist = new ArrayList[DocumentGraph]();
    		for (d <- trainDocGraphs) {
    			jlist.add(d);
    		}

    		return jlist;
    }

    def getTestingDocGraphs(): ArrayList[DocumentGraph] = {
    		val testCorefDocs = CorefStructUtils.loadCorefDocs(testDataPath, -1, Driver.docSuffix, mentionPropertyComputer);
    		val testDocGraphs = SingleTaskStructTesting.testGetdocgraphs(testDataPath, -1, mentionPropertyComputer);
    		berkPruner.pruneAll(testDocGraphs);
    		val jlist = new ArrayList[DocumentGraph]();
    		for (d <- testDocGraphs) {
    			jlist.add(d);
    		}
    		return jlist;
    }

}

object Ace05CorefInterf {
  
////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
  
  def extractOneCorefStructExample(docGraph: DocumentGraph, pairwiseIndexingFeaturizer: PairwiseIndexingFeaturizer) = {
    
    val corefDoc: CorefDoc = docGraph.corefDoc;
    val rawDoc = corefDoc.rawDoc;
    val docName = rawDoc.docID;
    val addToIdxer: Boolean = docGraph.addToFeaturizer;
    
    
    ////// Coref Coref Coref Coref Coref Coref Coref Coref Coref
    val docCorefVars = new ArrayBuffer[IndepVariable[Int]];
    
    // featurizing!
    docGraph.featurizeIndexNonPrunedUseCache(pairwiseIndexingFeaturizer); 
    
    for (i <- 0 until docGraph.size) {
      val ment = docGraph.getMention(i);
      val corefValArr = new ArrayBuffer[VarValue[Int]]();
      val corefGoldArr = new ArrayBuffer[VarValue[Int]]();

      var valueCnt = 0;
      
      val prunedDomain = docGraph.getPrunedDomain(i, false);
      val goldPrunedAntecedents = docGraph.getGoldAntecedentsUnderCurrentPruning(i);
      for (j <- prunedDomain) {
        val correct = goldPrunedAntecedents.contains(j);
        val anteValue = new VarValue[Int](valueCnt, j, docGraph.cachedFeats(i)(j), correct);
        valueCnt += 1;
          
        corefValArr += anteValue;
        if (correct) {
          corefGoldArr += anteValue;
        }
      }
      docCorefVars += (new IndepVariable[Int](corefValArr.toArray, corefGoldArr.toArray, corefValArr(0)));
    }
    
    val structCorefExmp = new AceSingleTaskStructExample[Int](docCorefVars.toArray);
    structCorefExmp;
  }
  
  
  def scoringCorefInstances(tstInsts: java.util.List[AceCorefInstance]) {
    val allPredBackptrs = new Array[Array[Int]](tstInsts.size);
    val docseq = new ArrayBuffer[DocumentGraph]();
    
	  for (i <- 0 until tstInsts.size) {
      val ex = tstInsts.get(i);
		  //// Coref
      allPredBackptrs(i) = computeCorefBackPointer(ex, ex.predictOutput.output);
      docseq += ex.docGraph;
	  }
    
    println("------- COREF:");
    val allPredClusteringsSeq = (0 until allPredBackptrs.length).map(i => OrderedClustering.createFromBackpointers(allPredBackptrs(i)));
    val allPredClusteringsArr = allPredClusteringsSeq.toArray;
    val scoreOutput = CorefEvaluator.evaluateAndRender(docseq.toSeq, allPredBackptrs, allPredClusteringsArr, Driver.conllEvalScriptPath, "DEV: ", Driver.analysesToPrint);
    println(scoreOutput);
    
  }
  
  def computeCorefBackPointer(struct: AceCorefInstance, coutput: Array[Int]): Array[Int] = {
    val corefBackPointer = new Array[Int](coutput.size);
    //val corefStruct = struct.corefOutput;
    for (i <- 0 until coutput.size) {
      val j = coutput(i);
      corefBackPointer(i) = struct.getAnteIndex(i, j);//corefStruct.variables(i).values(j).value;
    }
    corefBackPointer
  }

  
  def docGraphgetGoldAntecedentsUnderCurrentPruning(dg: DocumentGraph, idx: Int) = {
    val seq = dg.getGoldAntecedentsUnderCurrentPruning(idx);
    val ilist = new ArrayList[Integer]();
    for (i <- seq) {
      ilist.add(i);
    }
    ilist;
  }
  
  def convertVarValueListtoArr(list: java.util.List[VarValue[Integer]]) = {
    val seq = list.toSeq;
    seq.toArray;
  }
  
}