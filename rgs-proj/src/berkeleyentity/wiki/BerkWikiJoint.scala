package berkeleyentity.wiki

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import scala.Array.canBuildFrom
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
import berkeleyentity.bp.Domain
import berkeleyentity.bp.UnaryFactorGeneral
import berkeleyentity.ner.NerExample
import berkeleyentity.bp.BinaryFactorGeneral
import berkeleyentity.Driver
import berkeleyentity.bp.ConstantBinaryFactor
import berkeleyentity.bp.SimpleFactorGraph
import berkeleyentity.ner.MCNerExample
import berkeleyentity.bp.ConstantUnaryFactor
import berkeleyentity.ner.MCNerFeaturizer
import berkeleyentity.wiki._
import edu.berkeley.nlp.futile.util.Counter
import java.io.PrintWriter
import berkeleyentity.joint.JointFeaturizerShared
import berkeleyentity.joint.JointDocACE
import berkeleyentity.sem.BrownClusterInterface
import berkeleyentity.oregonstate.CorefStructUtils
import berkeleyentity.coref.MentionPropertyComputer
import berkeleyentity.sem.QueryCountsBundle
import berkeleyentity.joint.GeneralTrainer
import berkeleyentity.joint.JointComputerShared
import berkeleyentity.coref.NumberGenderComputer
import berkeleyentity.sem.BasicWordNetSemClasser
import berkeleyentity.coref.FeatureSetSpecification
import berkeleyentity.EntitySystem
import berkeleyentity.joint.FactorGraphFactoryACE
import berkeleyentity.lang.Language
import berkeleyentity.sem.SemClasser
import berkeleyentity.ConllDocReader
import berkeleyentity.coref.CorefDocAssembler
import berkeleyentity.coref.LexicalCountsBundle
import berkeleyentity.coref.CorefFeaturizerTrainer
import berkeleyentity.joint.JointPredictorACE
import berkeleyentity.coref.CorefPruner
import berkeleyentity.joint.JointLossFcns
import berkeleyentity.coref.PairwiseLossFunctions
import berkeleyentity.coref.UID
import berkeleyentity.joint.LikelihoodAndGradientComputer
import berkeleyentity.joint.JointDocFactorGraph

class WikiFactorGraphACE(val doc: JointDocACE,
                         val featurizer: JointFeaturizerShared[MCNerFeaturizer],
                         val wikiDB: Option[WikipediaInterface],
                         val gold: Boolean,
                         val addToIndexer: Boolean,
                         val corefLossFcn: (CorefDoc, Int, Int) => Float,
                         val nerLossFcn: (String, String) => Float,
                         val wikiLossFcn: (Seq[String], String) => Float) {//  extends JointDocFactorGraph {
  val name = doc.rawDoc.fileName;
  val docGraph = doc.docGraph;
  val nerLabelIndexer = featurizer.nerFeaturizer.labelIndexer;
  
  println("Instantiating factor graph for " + doc.rawDoc.printableDocName + " with " + doc.rawDoc.words.size + " sentences and " + docGraph.getMentions.size + " mentions");

  val wikiNodes = new Array[Node[String]](docGraph.size);
  val queryNodes = new Array[Node[Query]](docGraph.size);

  val allNodes = new ArrayBuffer[Node[_]]();
  
  val wikiUnaryFactors = new Array[ConstantUnaryFactor[String]](docGraph.size);
  val queryUnaryFactors = new Array[UnaryFactorGeneral](docGraph.size);
  val queryWikiBinaryFactors = new Array[BinaryFactorGeneral](docGraph.size);

  
  val allFactors = new ArrayBuffer[Factor]();
//  val allFactorsEveryIter = new ArrayBuffer[Factor]();
  
  private def addAndReturnNode[T](node: Node[T], isEveryItr: Boolean): Node[T] = {
    allNodes += node;
//    if (isEveryItr) allNodesEveryIter += node;
    node;
  }
  
  private def addAndReturnFactor[T <: Factor](factor: T, isEveryItr: Boolean): T = {
    allFactors += factor;
//    if (isEveryItr) allFactorsEveryIter += factor; 
    factor;
  }
  
  ///////////////////////////////
  // BUILDING THE FACTOR GRAPH //
  ///////////////////////////////
  val featsChart = docGraph.featurizeIndexNonPrunedUseCache(featurizer.corefFeaturizer);
  
  val qcComputer = new QueryChoiceComputer(wikiDB.get, featurizer.indexer);
  // ALL NODES AND UNARY FACTORS
  for (i <- 0 until docGraph.size) {
    val mentIsInGold = docGraph.corefDoc.isInGold(docGraph.getMention(i));
    // WIKI
    val goldWikAnnots = doc.getGoldWikLabels(docGraph.getMention(i));
    // LATENT QUERY WIKIFICATION
    if (docGraph.getMention(i).mentionType.isClosedClass) {
    	val nilQueryDomainArr = Array(Query.makeNilQuery(docGraph.getMention(i)));
    	queryNodes(i) = addAndReturnNode(new Node[Query](new Domain(nilQueryDomainArr)), false);
    	queryUnaryFactors(i) = addAndReturnFactor(new UnaryFactorGeneral(queryNodes(i), Array(Array())), false);
    	wikiNodes(i) = addAndReturnNode(new Node[String](new Domain(Array(ExcludeToken))), true);
    	queryWikiBinaryFactors(i) = addAndReturnFactor(new BinaryFactorGeneral(queryNodes(i), wikiNodes(i), Array(Array(Array()))), false);
    } else {
    	val queries = Query.extractQueriesBest(docGraph.getMention(i), true);
    	val queryDisambigs = queries.map(wikiDB.get.disambiguateBestGetAllOptions(_));
    	// Build unary factors for queries
    	val queryDomain = new Domain(queries.toArray)
    	queryNodes(i) = addAndReturnNode(new Node[Query](queryDomain), false);
    	val queryFeatures = qcComputer.featurizeQueries(queries, addToIndexer);
    	queryUnaryFactors(i) = addAndReturnFactor(new UnaryFactorGeneral(queryNodes(i), queryFeatures), false);
    	val rawQueryDenotations = qcComputer.extractDenotationSetWithNil(queries, queryDisambigs, Driver.maxNumWikificationOptions);
    	val denotations = if (gold && !Driver.leaveWikificationLatent && mentIsInGold) {
    		val goldDenotations= rawQueryDenotations.filter(den => isCorrect(goldWikAnnots, den));
    		if (goldDenotations.isEmpty) rawQueryDenotations else goldDenotations;
    	} else {
    		rawQueryDenotations;
    	}
    	wikiNodes(i) = addAndReturnNode(new Node[String](new Domain(denotations.toArray)), true);
    	queryWikiBinaryFactors(i) = addAndReturnFactor(new BinaryFactorGeneral(queryNodes(i), wikiNodes(i), qcComputer.featurizeQueriesAndDenotations(queries, denotations, addToIndexer)), false);
    }
  }

  
  // Initialize received messages at nodes
  allNodes.foreach(_.initializeReceivedMessagesUniform());

  var nerNanos = 0L;
  var agreeNanos = 0L;
  
  Logger.logss("Document factor graph instantiated: " + docGraph.size + " mentions, " + allNodes.size + " nodes, " +
               allFactors.size + " factors");
  
  
  
  def setWeights(weights: Array[Float]) {
    // Update weights of the factors
    for (factor <- allFactors) {
      factor.setWeights(weights);
    }
    // Scrub values of potentials. Can't just reset all to zero because they're
    // still linked to the received messages from the previous iteration, so the
    // arrays themselves need to be reinitialized.
    allNodes.foreach(_.initializeReceivedMessagesUniform());
    for (node <- allNodes) {
      node.sendMessages;
    }
  }
  
  /////////////////////
  // MESSAGE PASSING //
  /////////////////////
  
  def computeAndStoreMarginals(weights: Array[Float],
                               exponentiateMessages: Boolean,
                               numBpIters: Int) {
    setWeights(weights);
    passMessagesFancy(numBpIters, exponentiateMessages);
  }
  
  def computeLogNormalizerApprox: Double = {
    SimpleFactorGraph.computeLogNormalizerApprox(allNodes, allFactors)
  }
  
  def scrubMessages() {
    allNodes.foreach(_.initializeReceivedMessagesUniform);
    allNodes.foreach(_.clearSentMessages);
    allFactors.foreach(_.clearAllMessages);
  }
  
  def passMessagesFancy(numItrs: Int, exponentiateMessages: Boolean) {
	  if (exponentiateMessages) {
		  throw new RuntimeException("Exponentiation of messages not implemented");
	  }
	  // LATENT QUERY WIKIFICATION
	  // Send messages from unary factors first; these only need to be sent once
	  queryUnaryFactors.foreach(_.sendMessages());
	  passNodeMessagesNonnull(queryNodes, 1.0);
	  passNodeMessagesNonnull(wikiNodes, 1.0);
	  queryWikiBinaryFactors.foreach(_.sendMessages());
	  /*
      for (i <- 0 until numItrs) {
        passNodeMessagesNonnull(corefNodes, 1.0);
        passNodeMessagesNonnull(nerNodes, 1.0);
        passNodeMessagesNonnull(wikiNodes, 1.0);
        val time = System.nanoTime();
        for (i <- 0 until agreementFactors.size) {
          for (agreementFactor <- agreementFactors(i)) {
            if (agreementFactor != null) {
              agreementFactor.sendMessages;
            }
          }
          for (corefWikiFactor <- corefWikiFactors(i)) {
            if (corefWikiFactor != null) {
              corefWikiFactor.sendMessages;
            }
          }
        }
        for (i <- 0 until wikiNerFactors.size) {
          if (wikiNerFactors(i) != null) {
            wikiNerFactors(i).sendMessages;
          }
        }
        agreeNanos += System.nanoTime() - time;
      }
	   */
	  // Send stuff back to unary factors
	  //passNodeMessagesNonnull(corefNodes, 1.0);
	  //passNodeMessagesNonnull(nerNodes, 1.0);
	  passNodeMessagesNonnull(wikiNodes, 1.0);
	  queryWikiBinaryFactors.foreach(_.sendMessages());
	  passNodeMessagesNonnull(queryNodes, 1.0);
  }
  
  def passNodeMessagesNonnull(nodes: Array[_ <: Node[_]], messageMultiplier: Double) {
    for (node <- nodes) {
      if (node != null) node.sendMessages(messageMultiplier);
    }
  }
  
  def passFactorMessagesNonnull(factors: Array[_ <: Factor]) {
    for (factor <- factors) {
      if (factor != null) factor.sendMessages
    }
  }

  def addExpectedFeatureCountsToGradient(scale: Float, gradient: Array[Float]) {
    allFactors.foreach(_.addExpectedFeatureCounts(scale, gradient))
  }
  
  //////////////
  // DECODING //
  //////////////

  def decodeWikificationProduceChunks = chunkifyMentionAnnots(wikiNodes.map(node => node.domain.entries(GUtil.argMaxIdx(node.getMarginals))))
  
  private def chunkifyMentionAnnots(mentAnnots: Seq[String]) = {
    val chunksPerSentence = (0 until docGraph.corefDoc.rawDoc.numSents).map(i => new ArrayBuffer[Chunk[String]]);
    for (i <- 0 until docGraph.getMentions.size) {
      val ment = docGraph.getMention(i);
      chunksPerSentence(ment.sentIdx) += new Chunk[String](ment.startIdx, ment.endIdx, mentAnnots(i));
    }
    chunksPerSentence;
  }
  
/*
  def getRepresentativeFeatures = {
    val featsByTemplate = new HashMap[String,String];
    val allUnaryFactors: Array[UnaryFactorGeneral] = corefUnaryFactors ++ nerUnaryFactors
    for (factor <- allUnaryFactors) {
      for (featArr <- factor.indexedFeatures; feat <- featArr) {
        val featStr = featurizer.indexer.getObject(feat);
        val template = PairwiseIndexingFeaturizer.getTemplate(featStr);
        if (!featsByTemplate.contains(template)) featsByTemplate.put(template, featStr);
      }
    }
    for (agreementFactorArr <- agreementFactors; agreementFactor <- agreementFactorArr) {
      if (agreementFactor != null) {
        for (featSeqArr <- agreementFactor.indexedFeatureMatrix; featSeq <- featSeqArr; feat <- featSeq) {
          val featStr = featurizer.indexer.getObject(feat);
          val template = PairwiseIndexingFeaturizer.getTemplate(featStr);
          if (!featsByTemplate.contains(template)) featsByTemplate.put(template, featStr);
        }
      }
    }
    featsByTemplate
  }
*/
  
}




/////////////////////////////
/////////////////////////////
/////////////////////////////

class WikiComputer(factorGraphFactory: WikiGraphFactoryACE)  extends LikelihoodAndGradientComputer[JointDocACE] {
  
  var egCounter = 0;
  val NumBpIters = Driver.numBPItrs; // 15;
  def getInitialWeightVector(featureIndexer: Indexer[String]): Array[Float] = Array.fill(featureIndexer.size())(0.0F);
  
  def computeLogLikelihood(doc: JointDocACE,
                           weights: Array[Float]): Float = {
    val goldGraph = factorGraphFactory.getDocFactorGraphHard(doc, true);
    val predGraph = factorGraphFactory.getDocFactorGraphHard(doc, false);
    computeLogLikelihood(goldGraph, predGraph, weights).toFloat;
  }
  
  private def computeLogLikelihood(goldDocFactorGraph: WikiFactorGraphACE,
                                   predDocFactorGraph: WikiFactorGraphACE,
                                   weights: Array[Float]): Double = {
    goldDocFactorGraph.computeAndStoreMarginals(weights, false, NumBpIters);
    val goldNormalizer = goldDocFactorGraph.computeLogNormalizerApprox;
    goldDocFactorGraph.scrubMessages();
    predDocFactorGraph.computeAndStoreMarginals(weights, false, NumBpIters);
    val predNormalizer = predDocFactorGraph.computeLogNormalizerApprox;
    predDocFactorGraph.scrubMessages();
    goldNormalizer - predNormalizer;
  }
  
  def addUnregularizedStochasticGradient(doc: JointDocACE,
                                         weights: Array[Float],
                                         gradient: Array[Float]) = {
    val doTwiddleWeightsAndDoEmpiricalGradient = false;
    
    val predDocFactorGraph = factorGraphFactory.getDocFactorGraphHard(doc, false);
    predDocFactorGraph.computeAndStoreMarginals(weights, false, NumBpIters);
    predDocFactorGraph.addExpectedFeatureCountsToGradient(-1.0F, gradient);
    val goldDocFactorGraph = factorGraphFactory.getDocFactorGraphHard(doc, true);
    goldDocFactorGraph.computeAndStoreMarginals(weights, false, NumBpIters);
    goldDocFactorGraph.addExpectedFeatureCountsToGradient(1.0F, gradient);
//    Logger.logss("Gradient of ExactHeadMatch=True: " + gradient(featurizer.indexer.indexOf("ExactHeadMatch=true")));
//    Logger.logss("Value of ExactHeadMatch=true: " + weights(featurizer.indexer.indexOf("ExactHeadMatch=true")));
/*   
    // EMPIRICAL GRADIENT CHECK
//    if (doTwiddleWeightsAndDoEmpiricalGradient && goldDocFactorGraph.allFactorsEveryIter.size > 0) {
    if (doTwiddleWeightsAndDoEmpiricalGradient && egCounter > 40) {
      Logger.logss("DocumentInferencerLoopy: empirical gradient check");
      val freshGradient = Array.fill(weights.size)(0.0F);
      predDocFactorGraph.addExpectedFeatureCountsToGradient(-1.0F, freshGradient);
      goldDocFactorGraph.addExpectedFeatureCountsToGradient(1.0F, freshGradient);
      val llBefore = computeLogLikelihood(doc, weights);
//      Logger.logss("Base likelihood correct: " + llBefore);
//      // Pick one feat from each template and check the EG on that
      val featsByTemplate = predDocFactorGraph.getRepresentativeFeatures;
      Logger.logss("Checking empirical gradient on " + featsByTemplate.size + " representatives of templates");
      for (template <- featsByTemplate.keySet) {
        val feat = featsByTemplate(template);
        val i = factorGraphFactory.getIndexer.getIndex(feat);
        Logger.logss(i + " " + feat);
        // If delta is too small, the fact that the likelihood comes back as a float is a liability because
        // most of the bits disappear and then it gets scaled up
        val delta = 1e-2F;
        weights(i) += delta;
        val llAfter = computeLogLikelihood(doc, weights);
        weights(i) -= delta;
        val diff = Math.abs(freshGradient(i) - (llAfter - llBefore)/delta);
        if (diff > 1e-2) {
          Logger.logss((if (diff > 0.5) "BAD " else "") + "Bump test problem: " + template + " - " + i + ": gradient = " + freshGradient(i) + ", change = " + ((llAfter - llBefore)/delta));
        }
      }
      System.exit(0);
    }
*/
    predDocFactorGraph.scrubMessages();
    goldDocFactorGraph.scrubMessages();
    egCounter += 1;
  }
  
  def viterbiDecodeProduceAnnotations(doc: JointDocACE, weights: Array[Float]) = {
    val factorGraph = factorGraphFactory.getDocFactorGraph(doc, false, false, false, PairwiseLossFunctions.noLoss, JointLossFcns.noNerLossFcn, JointLossFcns.noWikiLossFcn);
    // We exponentiate messages here, but don't need to exponentiate them below because that doesn't
    // change the max.
    factorGraph.computeAndStoreMarginals(weights, false, NumBpIters);
//    computeAndStoreMarginals(factorGraph, weights, lossAugmented = false, exponentiateMessages = true);
    // MBR decoding on coref
    //val predBackptrs = factorGraph.decodeCorefProduceBackpointers;
    // MBR decoding on NER as well
    //val chunks = factorGraph.decodeNERProduceChunks;
    val wikiChunks = factorGraph.decodeWikificationProduceChunks;
    factorGraph.scrubMessages();
    //factorGraph.printNodeDomains();
    ( wikiChunks);
  }
    
}

class WikiGraphFactoryACE(val featurizer: JointFeaturizerShared[MCNerFeaturizer],
                          val wikiDB: Option[WikipediaInterface]) { // extends FactorGraphFactory[JointDocACE,JointDocFactorGraphACE] {
  val goldFactorGraphCache = new HashMap[UID, WikiFactorGraphACE]();
  val guessFactorGraphCache = new HashMap[UID, WikiFactorGraphACE]();
  
  private def fetchGraphCache(gold: Boolean) = {
    if (gold) {
      goldFactorGraphCache
    } else {
      guessFactorGraphCache;
    }
  }
  
  def getIndexer() = {
    featurizer.indexer;
  }
  
  def getDocFactorGraph(doc: JointDocACE,
                        gold: Boolean,
                        addToIndexer: Boolean,
                        useCache: Boolean,
                        corefLossFcn: (CorefDoc, Int, Int) => Float,
                        nerLossFcn: (String, String) => Float,
                        wikiLossFcn: (Seq[String], String) => Float): WikiFactorGraphACE = {
    if (useCache) {
      val cache = fetchGraphCache(gold);
      if (!cache.contains(doc.rawDoc.uid)) {
        cache.put(doc.rawDoc.uid, new WikiFactorGraphACE(doc, featurizer, wikiDB, gold, addToIndexer, corefLossFcn, nerLossFcn, wikiLossFcn));
      }
      cache(doc.rawDoc.uid);
    } else {
      if (corefLossFcn == null) {
        throw new RuntimeException("You called getDocFactorGraphHard but it wasn't in the cache...")
      }
      new WikiFactorGraphACE(doc, featurizer, wikiDB, gold, addToIndexer, corefLossFcn, nerLossFcn, wikiLossFcn)
    }
  }
  
  def getDocFactorGraphHard(obj: JointDocACE, isGold: Boolean) = {
    getDocFactorGraph(obj, isGold, false, true, null, null, null);
  }
}


class WikiPredictorACE(val jointFeaturizer: JointFeaturizerShared[MCNerFeaturizer],
                       val weights: Array[Float]) extends Serializable {
  
	def decodeWriteOutputEvaluate(jointTestDocs: Seq[JointDocACE], maybeWikipediaInterface: Option[WikipediaInterface], doConllPostprocessing: Boolean, wikiLabelsInTrain: Set[String] = Set[String]()) {

		val allPredWikiChunks = new ArrayBuffer[Seq[Chunk[String]]];
		val allPredWikiTitles = new ArrayBuffer[Set[String]];
		Logger.startTrack("Decoding");
		for (i <- (0 until jointTestDocs.size)) {
			Logger.logss("Decoding " + i);
			val jointDevDoc = jointTestDocs(i);
			val wikiChunks = decode(jointDevDoc, maybeWikipediaInterface);
			allPredWikiChunks ++= wikiChunks;
			allPredWikiTitles += WikificationEvaluator.convertChunksToBagOfTitles(wikiChunks);
		}
		Logger.endTrack();
		Logger.logss("MENTION DETECTION")
		CorefDoc.displayMentionPRF1(jointTestDocs.map(_.docGraph.corefDoc));
		Logger.logss("WIKIFICATION");
		WikificationEvaluator.evaluateWikiChunksBySent(jointTestDocs.flatMap(_.goldWikiChunks), allPredWikiChunks);
		WikificationEvaluator.evaluateBOTF1(jointTestDocs.map(doc => WikificationEvaluator.convertSeqChunksToBagOfTitles(doc.goldWikiChunks)), allPredWikiTitles);
		WikificationEvaluator.evaluateFahrniMetrics(jointTestDocs.flatMap(_.goldWikiChunks), allPredWikiChunks, wikiLabelsInTrain)
	}
  
  def decode(jointTestDoc: JointDocACE, maybeWikipediaInterface: Option[WikipediaInterface]) = {
    val fgf = new WikiGraphFactoryACE(jointFeaturizer, maybeWikipediaInterface);
    val computer = new WikiComputer(fgf);
    computer.viterbiDecodeProduceAnnotations(jointTestDoc, weights);
  }
  /*
  def getOracleResult(jointTestDoc: JointDocACE, maybeWikipediaInterface: Option[WikipediaInterface]) = {
    val fgfOnto = new FactorGraphFactoryACE(jointFeaturizer, maybeWikipediaInterface);
    val myOracle = new JointOracle(fgfOnto);
    myOracle.runILPinference(jointTestDoc, weights);
  }
  */

  def pack: WikiPredictorACE = {
    if (jointFeaturizer.canReplaceIndexer) {
      val (newIndexer, newWeights) = GUtil.packFeaturesAndWeights(jointFeaturizer.indexer, weights);
      new WikiPredictorACE(jointFeaturizer.replaceIndexer(newIndexer), newWeights);
    } else {
      this;
    }
  }
}

object BerkWikiJoint {

	def main(args: Array[String]) {

		// set some configs
		Driver.numberGenderDataPath = "data/gender.data";
		Driver.brownPath = "data/bllip-clusters";
		Driver.wordNetPath = "data/dict";
		Driver.useGoldMentions = true;
		Driver.doConllPostprocessing = false;
		//Driver.pruningStrategy = "build:./corefpruner-ace.ser.gz:-5:5";
		Driver.lossFcn = "customLoss-1-1-1";

		Driver.corefNerFeatures = "indicators+currlex+antlex";
		Driver.wikiNerFeatures = "categories+infoboxes+appositives";
		Driver.corefWikiFeatures = "basic+lastnames";

		runBerkeleyWiki();
	}

	def runBerkeleyWiki() {

		val trainDataPath = "data/ace05/train";
		val devDataPath = "data/ace05/dev";
		val testDataPath = "data/ace05/test";
		val wikiPath = "data/ace05/ace05-all-conll-wiki"
		val wikiDBPath = "models/wiki-db-ace.ser.gz"

		// independent model only
		Driver.corefNerFeatures = "";
		Driver.wikiNerFeatures = "";
		Driver.corefWikiFeatures = "";


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
		//featurizerTrainer.featurizeBasic(trainDocGraphs, basicFeaturizer);

		// joint featurizer
		val jointFeaturizer = new JointFeaturizerShared[MCNerFeaturizer](basicFeaturizer, nerFeaturizer, maybeBrownClusters, Driver.corefNerFeatures, Driver.corefWikiFeatures, Driver.wikiNerFeatures, featIndexer);

    
		val jointDocs = EntitySystem.preprocessACEDocsForTrainEval(trainDataPath, -1, mentionPropertyComputer, berkPruner, wikiPath, true);

		///////////////////////
		// Cache features
		val fgfAce = new WikiGraphFactoryACE(jointFeaturizer, Some(wikiDB));
		val computer = new WikiComputer(fgfAce);
		jointDocs.foreach(jointDoc => {
			fgfAce.getDocFactorGraph(jointDoc, true, true, true, PairwiseLossFunctions(Driver.lossFcn), JointLossFcns.nerLossFcn, JointLossFcns.wikiLossFcn);
			fgfAce.getDocFactorGraph(jointDoc, false, true, true, PairwiseLossFunctions(Driver.lossFcn), JointLossFcns.nerLossFcn, JointLossFcns.wikiLossFcn);
		});
		PairwiseIndexingFeaturizer.printFeatureTemplateCounts(featIndexer)
		Logger.logss(featIndexer.size + " total features");

		val finalWeights = new GeneralTrainer[JointDocACE].trainAdagrad(jointDocs, computer, featIndexer.size, Driver.eta.toFloat, Driver.reg.toFloat, Driver.batchSize, Driver.numItrs);

		val model = new WikiPredictorACE(jointFeaturizer, finalWeights).pack;
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