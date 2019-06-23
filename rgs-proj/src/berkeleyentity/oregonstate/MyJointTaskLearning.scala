package berkeleyentity.oregonstate

import berkeleyentity.sem._
import berkeleyentity.sem.SemClass._
import berkeleyentity.wiki.WikificationEvaluator
import berkeleyentity.coref.MentionPropertyComputer
import berkeleyentity.joint.JointDoc
import berkeleyentity.joint.GeneralTrainer
import berkeleyentity.joint.JointDocACE
import berkeleyentity.joint.JointComputerShared
import berkeleyentity.coref.NumberGenderComputer
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
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
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
import berkeleyentity.ilp.MyModel
import berkeleyentity.Driver
import berkeleyentity.GUtil
import berkeleyentity.ConllDocReader
import berkeleyentity.Chunk
import berkeleyentity.coref.Mention
import berkeleyentity.coref.MentionType
import berkeleyentity.coref.Number
import berkeleyentity.coref.Gender
import berkeleyentity.coref.MentClusterMapping

import scala.collection.JavaConverters._
import berkeleyentity.GUtil
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import scala.util.Random

import edu.berkeley.nlp.futile.util.Counter
import edu.berkeley.nlp.futile.util.Iterators
import edu.berkeley.nlp.futile.util.Logger
import edu.mit.jwi.item.Pointer
import berkeleyentity.sem.SemClass
import berkeleyentity.WordNetInterfacer


import java.util.ArrayList;
import java.nio.file.Files;
import java.nio.file.CopyOption._;
import java.io.File;
import scala.util.control.Breaks._



class JointStructureOutputACE(val doc: JointTaskDocument,
		val featurizer: JointFeaturizerShared[MCNerFeaturizer],
		val wikiDB: Option[WikipediaInterface],
		val gold: Boolean,
		val addToIndexer: Boolean,
		val corefLossFcn: (CorefDoc, Int, Int) => Float,
		val nerLossFcn: (String, String) => Float,
		val wikiLossFcn: (Seq[String], String) => Float) {

	val name = doc.corefDoc.rawDoc.fileName;
	val nerLabelIndexer = featurizer.nerFeaturizer.labelIndexer;
  val neLabelIdx = MCNerFeaturizer.StdLabelIndexer;
/*	
  Logger.logss("Instantiating factor graph for " + doc.rawDoc.printableDocName + " with " + doc.rawDoc.words.size + " sentences and " + docGraph.getMentions.size + " mentions");

	val corefNodes = new Array[Node[Int]](docGraph.size);
	val nerNodes = new Array[Node[String]](docGraph.size);
	val wikiNodes = new Array[Node[String]](docGraph.size);
	//val queryNodes = new Array[Node[Query]](docGraph.size);

	val allNodes = new ArrayBuffer[Node[_]]();

	val corefUnaryFactors = new Array[UnaryFactorGeneral](docGraph.size);
	val nerUnaryFactors = new Array[UnaryFactorGeneral](docGraph.size);
	val wikiUnaryFactors = new Array[ConstantUnaryFactor[String]](docGraph.size);

	val queryUnaryFactors = new Array[UnaryFactorGeneral](docGraph.size);
	val queryWikiBinaryFactors = new Array[BinaryFactorGeneral](docGraph.size);

	// Agreement = joint coref + NER factors
	val agreementFactors = Array.tabulate(docGraph.size)(i => new Array[BetterPropertyFactor[String]](i));
	val wikiNerFactors = new Array[BinaryFactorGeneral](docGraph.size);

	val corefWikiFactors = Array.tabulate(docGraph.size)(i => new Array[BetterPropertyFactor[String]](i));

	val allFactors = new ArrayBuffer[Factor]();

	private def addAndReturnNode[T](node: Node[T], isEveryItr: Boolean): Node[T] = {
			allNodes += node;
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
		// COREF
		val domainArr = docGraph.getPrunedDomain(i, gold);
		corefNodes(i) = addAndReturnNode(new Node[Int](new Domain(domainArr)), true);
		val featsEachDecision = domainArr.map(antIdx => featsChart(i)(antIdx));
		corefUnaryFactors(i) = addAndReturnFactor(new UnaryFactorGeneral(corefNodes(i), featsEachDecision), false);
		corefUnaryFactors(i).setConstantOffset(Array.tabulate(domainArr.size)(entryIdx => corefLossFcn(docGraph.corefDoc, i, domainArr(entryIdx))));
		// NER
		val nerDomainArr = if (gold && !Driver.leaveNERLatent && mentIsInGold) {
			new Domain[String](Array(doc.getGoldLabel(docGraph.getMention(i)))) 
		} else {
			new Domain[String](MCNerFeaturizer.StdLabelIndexer.getObjects.asScala.toArray);
		}
		nerNodes(i) = addAndReturnNode(new Node[String](nerDomainArr), true);
		val allNerFeats = featurizer.nerFeaturizer.featurize(MCNerExample(docGraph, i), addToIndexer);
		val nerFeatsEachDecision = nerDomainArr.entries.map(neLabel => allNerFeats(neLabelIdx.indexOf(neLabel)));
		nerUnaryFactors(i) = addAndReturnFactor(new UnaryFactorGeneral(nerNodes(i), nerFeatsEachDecision), false);
		nerUnaryFactors(i).setConstantOffset(Array.tabulate(nerDomainArr.size)(entryIdx => {
			//        Logger.logss(docGraph.getMention(i).nerString + " " + nerDomain.entries(entryIdx) + " " + nerLossFcn(docGraph.getMention(i).nerString, nerDomain.entries(entryIdx)));
			nerLossFcn(doc.getGoldLabel(docGraph.getMention(i)), nerDomainArr.entries(entryIdx))
		}));
		// WIKI
		val goldWikAnnots = doc.getGoldWikLabels(docGraph.getMention(i));
		// LATENT QUERY WIKIFICATION
		if (doc.getMention(i).mentionType.isClosedClass) {
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

	// NER+COREF FACTORS
	if (featurizer.corefNerFeatures != "") {
		for (i <- 0 until doc.size) {
			val domain = corefNodes(i).domain;
			val currNerNode = nerNodes(i);
			for (j <- domain.entries) {
				if (j != i) {
					val antNerNode = nerNodes(j);
					val featsIndexed: Array[Array[Array[Int]]] = Array.tabulate(currNerNode.domain.size, antNerNode.domain.size)((currNerValIdx, antNerValIdx) => {
						featurizer.getCorefNerFeatures(docGraph, i, j, currNerNode.domain.entries(currNerValIdx), antNerNode.domain.entries(antNerValIdx), addToIndexer);
					});
					agreementFactors(i)(j) = addAndReturnFactor(new BetterPropertyFactor[String](j, currNerNode, corefNodes(i), antNerNode, featsIndexed), true);
				}
			}
		}
	}
	// NER+WIKIFICATION FACTORS
	if (featurizer.wikiNerFeatures != "") {
		for (i <- 0 until doc.size) {
			val wikiNode = wikiNodes(i);
			val nerNode = nerNodes(i);
			val featsIndexed: Array[Array[Array[Int]]] = Array.tabulate(wikiNode.domain.size, nerNode.domain.size)((wikiValIdx, nerValIdx) => {
				featurizer.getWikiNerFeatures(docGraph, i, wikiNode.domain.entries(wikiValIdx), nerNode.domain.entries(nerValIdx), wikiDB, addToIndexer);
			});
			wikiNerFactors(i) = addAndReturnFactor(new BinaryFactorGeneral(wikiNode, nerNode, featsIndexed), true);
		}
	}
	// COREF+WIKIFICATION FACTORS
	if (featurizer.corefWikiFeatures != "") {
		for (i <- 0 until doc.size) {
			val domain = corefNodes(i).domain;
			val currWikiNode = wikiNodes(i);
			for (j <- domain.entries) {
				if (j != i) {
					val antWikiNode = wikiNodes(j);
					val featsIndexed: Array[Array[Array[Int]]] = Array.tabulate(currWikiNode.domain.size, antWikiNode.domain.size)((currWikiValIdx, antWikiValIdx) => {
						featurizer.getCorefWikiFeatures(docGraph, i, j, currWikiNode.domain.entries(currWikiValIdx), antWikiNode.domain.entries(antWikiValIdx), wikiDB, addToIndexer);
					});
					corefWikiFactors(i)(j) = addAndReturnFactor(new BetterPropertyFactor[String](j, currWikiNode, corefNodes(i), antWikiNode, featsIndexed), true);
				}
			}
		}
	}
  */
  
  
}



// contains all useful informations about each task
class JointTaskDocument(val corefDoc: CorefDoc) {

  var goldWikiChunks: Seq[Seq[Chunk[Seq[String]]]] = null;
  def setGoldWikiChunks(gwchunks: Seq[Seq[Chunk[Seq[String]]]]) {
    goldWikiChunks = gwchunks;
  }
  
  val goldChunks = (0 until corefDoc.rawDoc.numSents).map(sentIdx => {
    // Only take the part that's actually the type from each one, "GPE", "LOC", etc.
    corefDoc.goldMentions.filter(_.sentIdx == sentIdx).map(ment => new Chunk[String](ment.startIdx, ment.endIdx, getGoldLabel(ment)));
  });
  
  def getGoldLabel(ment: Mention) = {
    if (ment.nerString.size >= 3) {
      ment.nerString.substring(0, 3)
    } else {
      "O"  // SHouldn't happen during training
    };
  }
  
  def getGoldWikLabels(ment: Mention): Seq[String] = {
    val matchingChunk = goldWikiChunks(ment.sentIdx).filter(chunk => chunk.start == ment.startIdx && chunk.end == ment.endIdx);
    if (matchingChunk.size > 0) matchingChunk.head.label else Seq(ExcludeToken);
  }
  
  /////////////////////////////////////////////////////////
  
  // addToFeaturizer should be true for train documents (if a feature is unseen on
  // these, we add it to the featurizer) and false for dev/test documents
  // By convention: a feature vector is empty if it has been pruned
  val emptyIntArray = Array[Int]();
  var cachedFeats = new Array[Array[Array[Int]]](corefDoc.numPredMents);
  for (i <- 0 until corefDoc.numPredMents) {
    cachedFeats(i) = Array.fill(i+1)(emptyIntArray);
  }
  // These are just here so we don't have to reinstantiate them; they should
  // be overwritten every time the weights change (which is all the time)
  val cachedScoreMatrix = new Array[Array[Float]](corefDoc.numPredMents); 
  val cachedMarginalMatrix = new Array[Array[Float]](corefDoc.numPredMents);
  for (i <- 0 until corefDoc.numPredMents) {
    cachedScoreMatrix(i) = Array.fill(i+1)(0.0F);
    cachedMarginalMatrix(i) = Array.fill(i+1)(0.0F);
  }
  // Only used for DocumentInferencerRahman
  val cachedMentClusterMapping = new MentClusterMapping(corefDoc.numPredMents);
  
  var cachedFeaturizer: PairwiseIndexingFeaturizer = null;
  var cacheEmpty = true;
  // If an edge is pruned, it will never be featurized
  var prunedEdges = new Array[Array[Boolean]](corefDoc.numPredMents);
  for (i <- 0 until prunedEdges.size) {
    prunedEdges(i) = Array.fill(i+1)(false);
  }
  
  // Cached information for feature computation
  val storedClusterPosteriors = new ArrayBuffer[Array[Array[Float]]]();
  val storedDistributedLabels = new ArrayBuffer[Array[Array[Int]]]();
  val storedSemClass: Array[Option[SemClass]] = Array.tabulate(this.size)(i => None);
  val storedRelsBetter = new Array[HashMap[Seq[Pointer],Set[String]]](this.size);
  val storedRelsBetterCumulative = new Array[HashMap[Seq[Pointer],Set[String]]](this.size);
  val cachedMentionHeadMatchStatus: Array[Option[Boolean]] = Array.tabulate(this.size)(i => None);
  
  // WordNetInterfacer so the featurizer can find it if it needs to
  var cachedWni: WordNetInterfacer = null;
  
  def size() = corefDoc.numPredMents
  
  def getMention(idx: Int) = corefDoc.predMentions(idx);
  
  def getMentions() = corefDoc.predMentions;
  
  def getOraclePredClustering() = corefDoc.getOraclePredClustering;
  
  def getMentionStrAndContext(idx: Int): String = {
    val ment = getMention(idx);
    val mentionStart = ment.startIdx;
    val mentionEnd = ment.endIdx;
    val sentence = corefDoc.rawDoc.words(ment.sentIdx);
    val contextStart = Math.max(0, mentionStart - 3);
    val contextEnd = Math.min(mentionEnd + 3, sentence.size);
    (sentence.slice(contextStart, mentionStart).foldLeft("")(_ + " " + _) + " [" + sentence.slice(mentionStart, mentionEnd).foldLeft("")(_ + " " + _) +
      "] " + sentence.slice(mentionEnd, contextEnd).foldLeft("")(_ + " " + _)).trim(); 
  }
  
  def isGoldNoPruning(currIdx: Int, antecedentIdx: Int) = getGoldAntecedentsNoPruning(currIdx).contains(antecedentIdx);
  
  def isGoldCurrentPruning(currIdx: Int, antecedentIdx: Int) = getGoldAntecedentsUnderCurrentPruning(currIdx).contains(antecedentIdx);
  
  def isPruned(currIdx: Int, antecedentIdx: Int): Boolean = prunedEdges(currIdx)(antecedentIdx);
  
  def getPrunedDomain(idx: Int, gold: Boolean): Array[Int] = {
    val currAntecedents = getGoldAntecedentsUnderCurrentPruning(idx);
    val domainSeq = new ArrayBuffer[Int]();
    for (j <- 0 to idx) {
      if (!isPruned(idx, j) && (!gold || currAntecedents.contains(j))) {
        domainSeq += j;
      }
    }
    domainSeq.toArray;
  }

  
  
  
  
  //////////////////////////////////////////////////////////
   
  def pruneEdgesMentDistanceSentDistance(maxBackptrMentDistance: Int, maxPronounSentDistance: Int) {
    for (i <- 0 until prunedEdges.size) {
      val iSentIdx = getMention(i).sentIdx;
      for (j <- 0 to i) {
        val jSentIdx = getMention(j).sentIdx;
        if (j < i - maxBackptrMentDistance || (getMention(i).mentionType == MentionType.PRONOMINAL && iSentIdx - jSentIdx > maxPronounSentDistance)) {
          prunedEdges(i)(j) = true;
          cachedFeats(i)(j) = emptyIntArray;
        }
      }
    }
  }
  
  def pruneEdgesModel(model: PairwiseScorer, logPruningThreshold: Double) {
    /*
    for (i <- 0 until prunedEdges.size) {
      val scores = (0 to i).map(j => model.score(this, i, j, false));
      val bestIdx = GUtil.argMaxIdxFloat(scores);
      for (j <- 0 to i) {
        if (scores(j) < scores(bestIdx) + logPruningThreshold) {
          prunedEdges(i)(j) = true;
          cachedFeats(i)(j) = emptyIntArray;
        }
      }
    }*/
  }
  
  def getGoldClustersNoPruning(): Seq[Seq[Mention]] = {
    val allClusters = new ArrayBuffer[Seq[Mention]]();
    val oracleClustering = corefDoc.getOraclePredClustering
    for (cluster <- oracleClustering.clusters) {
      allClusters += cluster.map(getMention(_));
    }
    allClusters;
  }
  
  def getAllAntecedentsCurrentPruning(idx: Int): Seq[Int] = {
    val antecedents = new ArrayBuffer[Int];
    for (i <- 0 to idx) {
      if (!prunedEdges(idx)(i)) {
        antecedents += i;
      }
    }
    antecedents;
  }
  
  def getGoldAntecedentsNoPruning(): Array[Seq[Int]] = {
    (0 until this.size).map(getGoldAntecedentsNoPruning(_)).toArray;
  }
  
  def getGoldAntecedentsNoPruning(idx: Int): Seq[Int] = {
    val oracleClustering = corefDoc.getOraclePredClustering
    val antecedents = oracleClustering.getAllAntecedents(idx);
    if (antecedents.isEmpty) Seq(idx) else antecedents;
  }
  
  // This and the following return the set of allowed antecedents if all gold
  // antecedents have been pruned; effectively this ignores examples where
  // there is no gold. Always returns nonempty.
  def getGoldAntecedentsUnderCurrentPruning(): Array[Seq[Int]] = {
    (0 until this.size).map(getGoldAntecedentsUnderCurrentPruning(_)).toArray;
  }
  
  def getGoldAntecedentsUnderCurrentPruning(idx: Int): Seq[Int] = {
    val oracleClustering = corefDoc.getOraclePredClustering
    val antecedentsRaw = oracleClustering.getAllAntecedents(idx);
    val antecedents = if (antecedentsRaw.isEmpty) Seq(idx) else antecedentsRaw;
    val unprunedAntecedents = antecedents.filter(j => !prunedEdges(idx)(j))
    if (unprunedAntecedents.isEmpty) {
      // This is a little inefficient but this code isn't called that much (extremely rare in coarse pass
      // and generally not called for nonanaphoric guys, and most things are nonanaphoric)
      val allUnprunedBackptrs = prunedEdges(idx).zipWithIndex.filter((prunedAndIdx) => !prunedAndIdx._1).map(_._2).toSeq;
      allUnprunedBackptrs
    } else {
      unprunedAntecedents;
    }
  }
  
  // This and the following return the set of unpruned antecedents, possibly empty
  def getGoldAntecedentsUnderCurrentPruningOrEmptySet(): Array[Seq[Int]] = {
    (0 until this.size).map(getGoldAntecedentsUnderCurrentPruningOrEmptySet(_)).toArray;
  }
  
  def getGoldAntecedentsUnderCurrentPruningOrEmptySet(idx: Int): Seq[Int] = {
    val oracleClustering = corefDoc.getOraclePredClustering
    val antecedentsRaw = oracleClustering.getAllAntecedents(idx);
    val antecedents = if (antecedentsRaw.isEmpty) Seq(idx) else antecedentsRaw;
    val unprunedAntecedents = antecedents.filter(j => !prunedEdges(idx)(j))
    unprunedAntecedents;
  }

  // N.B. The matrices returned by this method are volatile. The feats one hangs around
  // unless you refeaturize, but the other one gets mutated every time you call this
  // method (though obviously it's only different if you prune or if the weights have changed).
  def featurizeIndexAndScoreNonPrunedUseCache(scorer: PairwiseScorer): (Array[Array[Array[Int]]], Array[Array[Float]]) = {
    val featsChart = featurizeIndexNonPrunedUseCache(scorer.featurizer);
    val scoreChart = cachedScoreMatrix;
    for (i <- 0 until corefDoc.numPredMents) {
      for (j <- 0 to i) {
        if (!prunedEdges(i)(j)) {
          require(featsChart(i)(j).size > 0);
          scoreChart(i)(j) = GUtil.scoreIndexedFeats(featsChart(i)(j), scorer.weights);
        } else {
          scoreChart(i)(j) = Float.NegativeInfinity;
        }
      }
    }
    (featsChart, scoreChart)
  }
  
  // How does this know whether or not to add features? The private variable addToFeatures...
  // a bit of a hack...
  def featurizeIndexNonPrunedUseCache(featurizer: PairwiseIndexingFeaturizer): Array[Array[Array[Int]]] = {
    if (cacheEmpty || featurizer != cachedFeaturizer) {
      cachedFeats = featurizeIndexNonPruned(featurizer);
      cachedFeaturizer = featurizer;
      cacheEmpty = false;
    }
    cachedFeats;
  }

  private def featurizeIndexNonPruned(featurizer: PairwiseIndexingFeaturizer): Array[Array[Array[Int]]] = {
    /*
    val featsChart = new Array[Array[Array[Int]]](corefDoc.numPredMents);
    for (i <- 0 until corefDoc.numPredMents) {
      featsChart(i) = new Array[Array[Int]](i+1);
      for (j <- 0 to i) {
        if (!prunedEdges(i)(j)) {
          featsChart(i)(j) = featurizer.featurizeIndex(this, i, j, addToFeaturizer);
        }
      }
    }
    featsChart;*/
    ???
  }
  
  def scoreNonPrunedUseCache(weights: Array[Float]): Array[Array[Float]] = {
    val featsChart = cachedFeats;
    val scoreChart = cachedScoreMatrix;
    for (i <- 0 until corefDoc.numPredMents) {
      for (j <- 0 to i) {
        if (!prunedEdges(i)(j)) {
          require(featsChart(i)(j).size > 0);
          scoreChart(i)(j) = GUtil.scoreIndexedFeats(featsChart(i)(j), weights);
        } else {
          scoreChart(i)(j) = Float.NegativeInfinity;
        }
      }
    }
    scoreChart
  }
  
  def setPrunedEdges(prunedEdges: Array[Array[Boolean]]) {
    this.prunedEdges = prunedEdges;
    for (i <- 0 until prunedEdges.size) {
      for (j <- 0 until prunedEdges(i).size) {
        if (prunedEdges(i)(j)) {
          cachedFeats(i)(j) = emptyIntArray;
        }
      }
    }
  }
  
  def clearFeatureCache() {
    for (i <- 0 until cachedFeats.size) {
      for (j <- 0 until cachedFeats(i).size) {
        cachedFeats(i)(j) = emptyIntArray;
      }
    }
  }
  
  def printAverageFeatureCountInfo() {
    var numerAnaphoric = 0;
    var denomAnaphoric = 0;
    var numerNonanaphoric = 0;
    var denomNonanaphoric = 0;
    for (i <- 0 until cachedFeats.size) {
      for (j <-0 until cachedFeats(i).size) {
        if (!prunedEdges(i)(j)) {
          if (i != j) {
            numerAnaphoric += cachedFeats(i)(j).size;
            denomAnaphoric += 1;
          } else {
            numerNonanaphoric += cachedFeats(i)(j).size;
            denomNonanaphoric += 1;
          }
        }
      }
    }
    Logger.logss("Avg feature counts anaphoric: " + numerAnaphoric.toDouble/denomAnaphoric.toDouble);
    Logger.logss("Avg feature counts nonanaphoric: " + numerNonanaphoric.toDouble/denomNonanaphoric.toDouble);
  }
  
  def numClusterers = storedClusterPosteriors.size;
  
  def numClusters(clustererIdx: Int) = storedClusterPosteriors(clustererIdx)(0).size;
  
  def getClusterPosteriors(clustererIdx: Int, mentIdx: Int): Array[Float] = {
    storedClusterPosteriors(clustererIdx)(mentIdx);
  }
  
  def getBestCluster(clustererIdx: Int, mentIdx: Int): Int = {
    var bestScore = Float.NegativeInfinity;
    var bestIdx = -1;
    for (i <- 0 until storedClusterPosteriors(clustererIdx)(mentIdx).length) {
      if (storedClusterPosteriors(clustererIdx)(mentIdx)(i) > bestScore) {
        bestScore = storedClusterPosteriors(clustererIdx)(mentIdx)(i);
        bestIdx = i;
      }
    }
    bestIdx;
  }
  
  def computeAndStorePhiPosteriors(useNumber: Boolean, useGender: Boolean, useNert: Boolean) {
    if (useNumber) {
      computeAndStorePhiPosterior((ment: Mention) => ment.number.ordinal(), Number.values().size - 1, Number.UNKNOWN.ordinal())
    }
    if (useGender) {
      computeAndStorePhiPosterior((ment: Mention) => ment.gender.ordinal(), Gender.values().size - 1, Gender.UNKNOWN.ordinal())
    }
  }
  
  def computeAndStorePhiPosterior(fcn: (Mention => Int), domainSize: Int, unknown: Int) {
    val EstimatorConfidence = 0.75F;
    val posteriors = new Array[Array[Float]](this.size);
    for (i <- 0 until size) {
      val idx = fcn(getMention(i));
      if (idx == unknown || idx == -1) {
        posteriors(i) = Array.tabulate(domainSize)(j => 1.0F/domainSize);
      } else if (idx >= domainSize) {
        throw new RuntimeException("Bad idx: " + idx + " for domain size " + domainSize + " " + getMention(i).nerString);
      } else {
        posteriors(i) = Array.tabulate(domainSize)(j => (1.0F - EstimatorConfidence)/domainSize);
        posteriors(i)(idx) += EstimatorConfidence;
      }
    }
    this.storedClusterPosteriors += posteriors;
  }
  
  def getSemClassUseCache(wordNetInterfacer: WordNetInterfacer, idx: Int) = {
    if (!storedSemClass(idx).isDefined) {
      storedSemClass(idx) = Some(SemClass.getSemClass(getMention(idx).headStringLc, getMention(idx).nerString, wordNetInterfacer))
    }
    storedSemClass(idx).getOrElse(SemClass.Other);
  }
  
  def getWordNetRelsBetterUseCache(wordNetInterfacer: WordNetInterfacer, rels: Seq[Pointer], idx: Int) = {
    if (storedRelsBetter(idx) == null) {
      storedRelsBetter(idx) = new HashMap[Seq[Pointer],Set[String]];
    }
    if (!storedRelsBetter(idx).contains(rels)) {
      storedRelsBetter(idx).put(rels, wordNetInterfacer.getWordsOnSynsetRelation(getMention(idx), rels).toSet);
    }
    storedRelsBetter(idx)(rels);
  }
  
  def getWordNetRelsBetterCumulativeUseCache(wordNetInterfacer: WordNetInterfacer, rels: Seq[Pointer], idx: Int) = {
    if (storedRelsBetterCumulative(idx) == null) {
      storedRelsBetterCumulative(idx) = new HashMap[Seq[Pointer],Set[String]];
    }
    if (!storedRelsBetterCumulative(idx).contains(rels)) {
      storedRelsBetterCumulative(idx).put(rels, wordNetInterfacer.getWordsUpToSynsetRelation(getMention(idx), rels).toSet);
    }
    storedRelsBetterCumulative(idx)(rels);
  }
  
  def getHeadMatchStatus(idx: Int) = {
    if (!cachedMentionHeadMatchStatus(idx).isDefined) {
      cachedMentionHeadMatchStatus(idx) = Some((0 until idx).map(i => (getMention(i).headStringLc == getMention(idx).headStringLc)).foldLeft(false)(_ || _));
    }
    cachedMentionHeadMatchStatus(idx).getOrElse(false);
  }
  
  def cacheWordNetInterfacer(wni: WordNetInterfacer) = {
    this.cachedWni = wni;
  }
  
}


object MyJointTaskLearning {

	def runTrainAceStruct(trainPath: String, trainSize: Int, testPath: String, testSize: Int) {
/*    
		println("My own joint training ...");

		val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
		val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));

		// Load coref models
		val corefPruner = CorefPruner.buildPruner(Driver.pruningStrategy)
		val jointDocs = preprocessACEDocsForTrainEval(trainPath, trainSize, mentionPropertyComputer, corefPruner, Driver.wikiGoldPath, false);//true);

		///////////////////////
		// Build the featurizer, which involves building specific featurizers for each task
		//val featureIndexer = new Indexer[String]();
		//val jointFeaturizer = buildFeaturizerShared(jointDocs.map(_.docGraph.corefDoc), featureIndexer, nerFeaturizer, maybeBrownClusters);
		val maybeBrownClusters = if (Driver.brownPath != "") Some(BrownClusterInterface.loadBrownClusters(Driver.brownPath, 0)) else None
		val maybeWikipediaInterface: Option[WikipediaInterface] = if (Driver.wikipediaPath != "") Some(GUtil.load(Driver.wikipediaPath).asInstanceOf[WikipediaInterface]) else None;


		val myJointFeatIndexer = new Indexer[String](); // for ILP Learning only
		val nerFeaturizer = MCNerFeaturizer(Driver.nerFeatureSet.split("\\+").toSet, myJointFeatIndexer, MCNerFeaturizer.StdLabelIndexer, jointDocs.flatMap(_.rawDoc.words), None, maybeBrownClusters)

				val myIlpFeaturizer = constructIlpFeaturizer[MCNerFeaturizer](jointDocs.map(_.docGraph.corefDoc), myJointFeatIndexer, nerFeaturizer, maybeBrownClusters);
		val prbConstructAce = new IlpAceProblemConstructor(myIlpFeaturizer, maybeWikipediaInterface, independentPredictor);
		jointDocs.foreach(jointDoc => { // load problem into memory
			prbConstructAce.getAceIlpProblem(jointDoc, true, true, PairwiseLossFunctions(Driver.lossFcn), JointLossFcns.nerLossFcn, JointLossFcns.wikiLossFcn);
			prbConstructAce.getAceIlpProblem(jointDoc, false, true, PairwiseLossFunctions(Driver.lossFcn), JointLossFcns.nerLossFcn, JointLossFcns.wikiLossFcn);
		});
		PairwiseIndexingFeaturizer.printFeatureTemplateCounts(myJointFeatIndexer);
		println("Done feature and doc loading! Total recorded feature values = " + myJointFeatIndexer.size());

		val structPerceptronLearner = new StructurePerceptronLearner();
		val myModel = structPerceptronLearner.runStructPerceptronTrain(jointDocs, prbConstructAce);
		MyModel.saveMyModel(myModel, Driver.myjModelPath);

		///////////////////////
		// Evaluation of each part of the model
		val jointDevDocs = preprocessACEDocsForTrainEval(testPath, testSize, mentionPropertyComputer, corefPruner, wikiPath, false);
		val wikiLabelsInTrain: Set[String] = jointDocs.flatMap(_.goldWikiChunks.flatMap(_.flatMap(_.label)).toSet).toSet;
		//myModel.decodeWriteOutputEvaluate(jointDevDocs, maybeWikipediaInterface, Driver.doConllPostprocessing, wikiLabelsInTrain)
*/
	}
  
	def jointDocConstructor(corefDocs: Seq[CorefDoc],
			                    wikiPath: String) {
    
    // extract gold wiki chunks
		val goldWikification: CorpusWikiAnnots = if (wikiPath != "") {
			val corpusAnnots = new CorpusWikiAnnots;
			for (entry <- WikiAnnotReaderWriter.readAllStandoffAnnots(wikiPath)) {
				val fileName = entry._1._1;
				val docAnnots = new DocWikiAnnots;
				for (i <- 0 until entry._2.size) {
					docAnnots += i -> (new ArrayBuffer[Chunk[Seq[String]]]() ++ entry._2(i))
				}
				corpusAnnots += fileName -> docAnnots
			}
			corpusAnnots;
		} else {
			Logger.logss("Wikification not loaded");
			new CorpusWikiAnnots;
		}
    
    // construct joint docs
	}
  
  def preprocessCorefDocsACE(path: String,
                             size: Int,
                             mentionPropertyComputer: MentionPropertyComputer,
                             //corefPruner: CorefPruner,
                             train: Boolean) = {

    // Read in raw data
    val rawDocs = ConllDocReader.loadRawConllDocsWithSuffix(path, size, "", Language.ENGLISH);
    val corefDocs = if (Driver.useGoldMentions) {
      val assembler = CorefDocAssembler(Driver.lang, true);
      rawDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    } else {
      val assembler = new CorefDocAssemblerACE(Driver.allAcePath); // ace predict mentions
      rawDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    }
    CorefDocAssembler.checkGoldMentionRecall(corefDocs);
    corefDocs;
/*    
    val docGraphs = corefDocs.map(new DocumentGraph(_, train));
    //preprocessDocsCacheResources(docGraphs);
    // Prune coref now that we have mentions
    //corefPruner.pruneAll(docGraphs);
    
    
    val jointDocsOrigOrder = JointDocACE.assembleJointDocs(docGraphs, goldWikification);
    // TODO: Apply NER pruning
//    JointDoc.applyNerPruning(jointDocsOrigOrder, nerMarginals);
    if (train) {
      // Randomize
      new scala.util.Random(0).shuffle(jointDocsOrigOrder)
    } else {
      jointDocsOrigOrder;
    }*/
  }
  
  /*
  def constructIlpFeaturizer[T](trainDocs: Seq[CorefDoc], myIndexer: Indexer[String], nerFeaturizer: T, maybeBrownClusters: Option[Map[String,String]]) = {

    myIndexer.getIndex(PairwiseIndexingFeaturizerJoint.UnkFeatName);
    val queryCounts: Option[QueryCountsBundle] = None;
    val lexicalCounts = LexicalCountsBundle.countLexicalItems(trainDocs, Driver.lexicalFeatCutoff);
    val semClasser: Option[SemClasser] = Some(new BasicWordNetSemClasser);
    val corefFeatureSetSpec = FeatureSetSpecification(Driver.pairwiseFeats, Driver.conjScheme, Driver.conjFeats, Driver.conjMentionTypes, Driver.conjTemplates);
    val corefFeaturizer = new PairwiseIndexingFeaturizerJoint(myIndexer, corefFeatureSetSpec, lexicalCounts, queryCounts, semClasser);
    
    val myfeater =  new FeaturizerILP[T](corefFeaturizer, 
                                         nerFeaturizer, 
                                         maybeBrownClusters, 
                                         Driver.corefNerFeatures, 
                                         Driver.corefWikiFeatures, 
                                         Driver.wikiNerFeatures, 
                                         myIndexer);
    //(val corefFeaturizer: PairwiseIndexingFeaturizer,
    //                   val nerFeaturizer: T,
    //                   val maybeBrownClusters: Option[Map[String,String]],
    //                   val corefNerFeatures: String,
    //                   val corefWikiFeatures: String,
    //                   val wikiNerFeatures: String,
    //                   val indexer: Indexer[String])
    myfeater;
  }*/
}