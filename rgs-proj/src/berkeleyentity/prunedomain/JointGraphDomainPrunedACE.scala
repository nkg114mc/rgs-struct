package berkeleyentity.prunedomain

import java.io.PrintWriter
import scala.Array.canBuildFrom
import scala.collection.JavaConverters.asScalaBufferConverter
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import berkeleyentity.Chunk
import berkeleyentity.Driver
import berkeleyentity.GUtil
import berkeleyentity.bp.BetterPropertyFactor
import berkeleyentity.bp.BinaryFactorGeneral
import berkeleyentity.bp.ConstantUnaryFactor
import berkeleyentity.bp.Domain
import berkeleyentity.bp.Factor
import berkeleyentity.bp.Node
import berkeleyentity.bp.SimpleFactorGraph
import berkeleyentity.bp.UnaryFactorGeneral
import berkeleyentity.coref.CorefDoc
import berkeleyentity.coref.PairwiseIndexingFeaturizer
import berkeleyentity.ner.MCNerExample
import berkeleyentity.ner.MCNerFeaturizer
import berkeleyentity.wiki.ExcludeToken
import berkeleyentity.wiki.Query
import berkeleyentity.wiki.QueryChoiceComputer
import berkeleyentity.wiki.WikipediaInterface
import berkeleyentity.wiki.isCorrect
import edu.berkeley.nlp.futile.util.Logger
import berkeleyentity.joint.JointDocFactorGraph
import berkeleyentity.joint.JointFeaturizerShared
import berkeleyentity.joint.JointDocACE

class JointGraphDomainPrunedACE(val doc: JointDocACE,
                                val featurizer: JointFeaturizerShared[MCNerFeaturizer],
                                val wikiDB: Option[WikipediaInterface],
                                val gold: Boolean,
                                val addToIndexer: Boolean,
                                val corefLossFcn: (CorefDoc, Int, Int) => Float,
                                val nerLossFcn: (String, String) => Float,
                                val wikiLossFcn: (Seq[String], String) => Float,
                                val domainModel: GraphPrunerModelACE,
                                val isTraining: Boolean) extends JointDocFactorGraph {
  
  val domainPruner = new FactoryGraphPrunerACE(doc, wikiDB, domainModel, gold, isTraining);
  val name = doc.rawDoc.fileName;
  val docGraph = doc.docGraph;
  val nerLabelIndexer = featurizer.nerFeaturizer.labelIndexer;
  Logger.logss("Instantiating factor graph for " + doc.rawDoc.printableDocName + " with " + doc.rawDoc.words.size + " sentences and " + docGraph.getMentions.size + " mentions");
  
  val corefNodes = new Array[Node[Int]](docGraph.size);
  val nerNodes = new Array[Node[String]](docGraph.size);
  val neLabelIdx = MCNerFeaturizer.StdLabelIndexer;
  val wikiNodes = new Array[Node[String]](docGraph.size);
  val queryNodes = new Array[Node[Query]](docGraph.size);

  val allNodes = new ArrayBuffer[Node[_]]();
//  val allNodesEveryIter = new ArrayBuffer[Node[_]]();
  
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
  
  /*
  if (isTraining) {
    //domainPruner.doSyntheticDomainPruning();
    //domainPruner.doDomainPruning();
    domainPruner.noDomainPruning();
  } else {
    //domainPruner.doSyntheticDomainPruning();
    domainPruner.doDomainPruning(); // run domain prunning
    //domainPruner.noDomainPruning();
  }
  //domainPruner.doDomainPruning(); // run domain prunning
  //domainPruner.noDomainPruning();
  */
  ///////////////////////////////
  // DO PRUNNING ////////////////
  ///////////////////////////////
  //domainPruner.doDomainPruning(); // run domain prunning
  domainPruner.noDomainPruning();
  
  // ALL NODES AND UNARY FACTORS
  for (i <- 0 until docGraph.size) {
    val mentIsInGold = docGraph.corefDoc.isInGold(docGraph.getMention(i));
    // COREF
    //val domainArr = docGraph.getPrunedDomain(i, gold);
    val domainArr = domainPruner.getCorefDomain(i, gold);
    corefNodes(i) = addAndReturnNode(new Node[Int](new Domain(domainArr)), true);
    val featsEachDecision = domainArr.map(antIdx => featsChart(i)(antIdx));
    corefUnaryFactors(i) = addAndReturnFactor(new UnaryFactorGeneral(corefNodes(i), featsEachDecision), false);
    corefUnaryFactors(i).setConstantOffset(Array.tabulate(domainArr.size)(entryIdx => corefLossFcn(docGraph.corefDoc, i, domainArr(entryIdx))));
    // NER
    val nerDomainArr = if (gold && !Driver.leaveNERLatent && mentIsInGold) {
      new Domain[String](Array(doc.getGoldLabel(docGraph.getMention(i)))) 
    } else {
      new Domain[String](domainPruner.getNerDomain(i, gold));
      //new Domain[String](MCNerFeaturizer.StdLabelIndexer.getObjects.asScala.toArray);
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
    if (false) {
      throw new RuntimeException("No longer implemented!");
    } else { // LATENT QUERY WIKIFICATION
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
          val goldDenotations = rawQueryDenotations.filter(den => isCorrect(goldWikAnnots, den));
          if (goldDenotations.isEmpty) rawQueryDenotations else goldDenotations;
        } else {
          val prunedDenotations = domainPruner.getWikiDomain(i, gold);//, rawQueryDenotations);
          prunedDenotations.toSeq;
          //rawQueryDenotations;
        }
        wikiNodes(i) = addAndReturnNode(new Node[String](new Domain(denotations.toArray)), true);
        queryWikiBinaryFactors(i) = addAndReturnFactor(new BinaryFactorGeneral(queryNodes(i), wikiNodes(i), qcComputer.featurizeQueriesAndDenotations(queries, denotations, addToIndexer)), false);
      }
    }
  }
  // NER+COREF FACTORS
  if (featurizer.corefNerFeatures != "") {
    for (i <- 0 until docGraph.size) {
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
    for (i <- 0 until docGraph.size) {
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
    for (i <- 0 until docGraph.size) {
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
  
  // Initialize received messages at nodes
  allNodes.foreach(_.initializeReceivedMessagesUniform());

  var nerNanos = 0L;
  var agreeNanos = 0L;
  
  Logger.logss("Document factor graph instantiated: " + docGraph.size + " mentions, " + allNodes.size + " nodes, " +
               allFactors.size + " factors, " + corefUnaryFactors.size + " coref unary factors, " +
               nerUnaryFactors.size + " NER unary factors");
  
  
  
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
    if (false) {
      throw new RuntimeException("No longer implemented!");
    } else { // LATENT QUERY WIKIFICATION
      // Send messages from unary factors first; these only need to be sent once
      corefUnaryFactors.foreach(_.sendMessages());
      nerUnaryFactors.foreach(_.sendMessages());
      queryUnaryFactors.foreach(_.sendMessages());
      passNodeMessagesNonnull(queryNodes, 1.0);
      passNodeMessagesNonnull(wikiNodes, 1.0);
      queryWikiBinaryFactors.foreach(_.sendMessages());
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
      // Send stuff back to unary factors
      passNodeMessagesNonnull(corefNodes, 1.0);
      passNodeMessagesNonnull(nerNodes, 1.0);
      passNodeMessagesNonnull(wikiNodes, 1.0);
      queryWikiBinaryFactors.foreach(_.sendMessages());
      passNodeMessagesNonnull(queryNodes, 1.0);
    }
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
  
  def decodeCorefProduceBackpointers = {
    (0 until docGraph.size).map(i => corefNodes(i).domain.entries(GUtil.argMaxIdx(corefNodes(i).getMarginals))).toArray;
  }
  
  def decodeNERProduceChunks = chunkifyMentionAnnots(nerNodes.map(node => node.domain.entries(GUtil.argMaxIdx(node.getMarginals))))
  
  def decodeWikificationProduceChunks = chunkifyMentionAnnots(wikiNodes.map(node => node.domain.entries(GUtil.argMaxIdx(node.getMarginals))))
  
  private def chunkifyMentionAnnots(mentAnnots: Seq[String]) = {
    val chunksPerSentence = (0 until docGraph.corefDoc.rawDoc.numSents).map(i => new ArrayBuffer[Chunk[String]]);
    for (i <- 0 until docGraph.getMentions.size) {
      val ment = docGraph.getMention(i);
      chunksPerSentence(ment.sentIdx) += new Chunk[String](ment.startIdx, ment.endIdx, mentAnnots(i));
    }
    chunksPerSentence;
  }
  
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
  
  def printNodeDomains() {
    // do nothing ...    
  }
  
  def pickMaxVal(arr: Array[Double]) : Double = {
    val maxVal : Double = arr.reduceLeft(math.max);
    maxVal;
  }
}
