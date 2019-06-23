package berkeleyentity.mentions

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap

import berkeleyentity.Driver
import berkeleyentity.EntitySystem
import berkeleyentity.GUtil
import berkeleyentity.MyTimeCounter
import berkeleyentity.coref.CorefEvaluator
import berkeleyentity.coref.CorefFeaturizerTrainer
import berkeleyentity.coref.CorefPruner
import berkeleyentity.coref.CorefSystem
import berkeleyentity.coref.DocumentGraph
import berkeleyentity.coref.DocumentInferencerBasic
import berkeleyentity.coref.FeatureSetSpecification
import berkeleyentity.coref.LexicalCountsBundle
import berkeleyentity.coref.Mention
import berkeleyentity.coref.NumberGenderComputer
import berkeleyentity.coref.PairwiseIndexingFeaturizer
import berkeleyentity.coref.PairwiseIndexingFeaturizerJoint
import berkeleyentity.coref.PairwiseScorer
import berkeleyentity.sem.BasicWordNetSemClasser
import berkeleyentity.sem.QueryCountsBundle
import berkeleyentity.sem.SemClasser
import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.futile.util.Counter
import edu.berkeley.nlp.futile.util.Logger
import edu.illinois.cs.cogcomp.sl.core.AbstractFeatureGenerator
import edu.illinois.cs.cogcomp.sl.core.IInstance
import edu.illinois.cs.cogcomp.sl.core.IStructure
import edu.illinois.cs.cogcomp.sl.core.SLModel
import edu.illinois.cs.cogcomp.sl.core.SLParameters
import edu.illinois.cs.cogcomp.sl.core.SLProblem
import edu.illinois.cs.cogcomp.sl.latentsvm.AbstractLatentInferenceSolver
import edu.illinois.cs.cogcomp.sl.learner.Learner
import edu.illinois.cs.cogcomp.sl.learner.LearnerFactory
import edu.illinois.cs.cogcomp.sl.util.FeatureVectorBuffer
import edu.illinois.cs.cogcomp.sl.util.IFeatureVector
import edu.illinois.cs.cogcomp.sl.util.WeightVector
import edu.illinois.cs.cogcomp.sl.core.AbstractInferenceSolver
import berkeleyentity.coref.OrderedClustering
import java.util.Arrays
import java.io.PrintWriter
import berkeleyentity.coref.PairwiseIndexingFeaturizerChao
import berkeleyentity.coref.CorefDoc
import berkeleyentity.coref.MentionPropertyComputer
import berkeleyentity.coref.CorefDocAssemblerFile
import berkeleyentity.ConllDocReader
import berkeleyentity.coref.CorefDocAssembler

//////////////////////////////////////
// for UIUC Structural Learning Lib //
//////////////////////////////////////

class BerkSvmCorefOutput(val output: Int) extends IStructure {
	override def equals(o1: Any): Boolean = {
		val vec = o1.asInstanceOf[BerkSvmCorefOutput];
		return (this.output == vec.output);
	}
}

class BerkSvmCorefFeatGener extends AbstractFeatureGenerator {
	override def getFeatureVector(xi: IInstance, yhati: IStructure): IFeatureVector = {
			val x = xi.asInstanceOf[SingletonMentionInstance];
			val yhat = yhati.asInstanceOf[BerkSvmCorefOutput];
			val fb = new FeatureVectorBuffer();
			
			val bestEdge = x.edges(yhat.output)
			val edgeFeature = bestEdge.features
			
			for (idx <- edgeFeature) {
				fb.addFeature(idx, 1.0);
			}
			fb.toFeatureVector();
	}
	override def getFeatureVectorDiff(x: IInstance, y1: IStructure, y2: IStructure): IFeatureVector = {
			val f1 = getFeatureVector(x, y1);
			val f2 = getFeatureVector(x, y2);		
			return f1.difference(f2);
	}
}

class BerkSvmCorefInferencer() extends AbstractLatentInferenceSolver {

	val callCount = new Counter[Int](); // for efficiency statistic

	def getBestStructure(wi: WeightVector, xi: IInstance): IStructure = {
			callCount.incrementCount(1, 1);
			getLossAugmentedBestStructure(wi, xi, null);
	}

	def getLoss(xi: IInstance, ystari: IStructure, yhati: IStructure): Float = {
			val x = xi.asInstanceOf[SingletonMentionInstance];
			val ystar = ystari.asInstanceOf[BerkSvmCorefOutput];
			val yhat = yhati.asInstanceOf[BerkSvmCorefOutput];
			val loss = getWeightedZeroOneError(yhat, ystar);
			loss.toFloat; // non-normalized loss
	}
	
	def getWeightedZeroOneError(pred: BerkSvmCorefOutput, gold: BerkSvmCorefOutput): Float = {
	  if (pred.output == gold.output) {
	    return 0f;
	  } else {
	    return 1.0f;
	  }
	}

	def getLossAugmentedBestStructure(wi: WeightVector, xi: IInstance, ystari: IStructure): IStructure = {

			callCount.incrementCount(0, 1);
			val example = xi.asInstanceOf[SingletonMentionInstance];
			val ystar = ystari.asInstanceOf[BerkSvmCorefOutput];

			val wgtArr = wi.getDoubleArray();
			
			var bestSc: Float = -Float.MaxValue;
			var besteIdx: Int = -1;
			for (i <- 0 until example.edges.size) {
			  val e = example.edges(i)
			  val addOnLoss = if (ystar != null) {
				  if (i == ystar.output) {
				    0f
				  } else {
				    1f
				  }
			  } else {
				  0f
			  }

			  val score = SingletonDetection.doDotProduct(wgtArr, e.features)
			  val lossAugScore = score + addOnLoss
			  if (lossAugScore > bestSc) {
			    bestSc = lossAugScore
			    besteIdx = i
			  }
			}

			(new BerkSvmCorefOutput(besteIdx));
	}
	


	override def getBestLatentStructure(weight: WeightVector, ins: IInstance, gold: IStructure): IStructure = {
			callCount.incrementCount(2, 1);
			val example = ins.asInstanceOf[SingletonMentionInstance];
			val ystar = gold.asInstanceOf[BerkSvmCorefOutput];

			val wgtArr = weight.getDoubleArray();
			
			var bestSc: Float = -Float.MaxValue;
			var besteIdx: Int = -1;
			for (i <- 0 until example.size()) {
			  val e = example.edges(i)
			  if (e.goldLb > 0) {
				  val lossAugScore = SingletonDetection.doDotProduct(wgtArr, e.features)
					if (lossAugScore > bestSc) {
						bestSc = lossAugScore
						besteIdx = i
					}
			  }
			}

			(new BerkSvmCorefOutput(besteIdx));
	}

	def getWeightArray(wv: WeightVector, sz: Int): Array[Double] = {
			val farr = wv.getWeightArray; // (0) is bias
			var darr = new Array[Double](sz);
			for (i <- 0 until darr.length) {
				darr(i) = farr(i).toDouble;
			}
			darr;
	}

	def printCount() {
		println("LossAugmntCall: " + callCount.getCount(0).toInt);
		println(" InferenceCall: " + callCount.getCount(1).toInt);
		println("LatentInfrCall: " + callCount.getCount(2).toInt);
	}

}

@SerialVersionUID(1L)
class CorefPrunerTopK(val model: PairwiseScorer,
                      val topk: Int) extends CorefPruner with Serializable {
  
  def prune(doc: DocumentGraph) {
    pruneWithtopk(doc, topk);
  }
  
  def pruneWithtopk(doc: DocumentGraph, topk: Int) {
    doc.pruneEdgesModelTopK(model, topk);
  }
}

@SerialVersionUID(1L)
class CorefPrunerTopAlpha(val model: PairwiseScorer,
                          val topAlpha: Double) extends CorefPruner with Serializable {
  
  def prune(doc: DocumentGraph) {
    val n = (doc.size().toDouble * topAlpha).toInt
    val topk = if (n < 3) 3 else n 
    doc.pruneEdgesModelTopK(model, topk);
  }
}


object BerkSvmCoref {
  
	def doDotProduct(weights: Array[Double], feature: Array[Int]): Float = {
		var sum: Float = 0
		for (idx <- feature) {
			sum += (weights(idx).toFloat)
		}
	  return sum
	}
  val suffix1 = "v4_auto_conll"
  val suffix2 = "v9_auto_conll"
  val trainSuffix = suffix1
  val testSuffix = suffix2
  
  val prunerPath = "model/berkEdgeLinear.pruner";
  val prunerScrPath = "model/berkEdgeLinearPrunerScorer.model";
  
  var globalMentIndex: Int = 0;
  
  def main(args: Array[String]) {
    runBerkOntoCoref()
  }
  
  def runBerkOntoCoref() {
    val trainPath = "/home/mc/workplace/rand_search/coref/berkfiles/data/ontonotes5/train";
    val testPath = "/home/mc/workplace/rand_search/coref/berkfiles/data/ontonotes5/test";
    
    //Driver.useGoldMentions = true
    
    //val (pscorer, pruner) = trainLinearPruner(trainPath, -1, trainSuffix, testPath, -1, testSuffix)
    //GUtil.save(pscorer, prunerScrPath);
    //GUtil.save(pruner, prunerPath);
    
    //val modelPath = "model/berkOnto.model";
    runBerkSvmTrain(trainPath, -1, trainSuffix, testPath, -1, testSuffix)
  }
  
  def runEvaluateGivenModel(devPath: String, devSize: Int, sufix: String, modelPath: String) {
    val scorer = GUtil.load(modelPath).asInstanceOf[PairwiseScorer]
    runEvaluate(devPath, devSize, sufix, scorer)
  }
  
  def runEvaluate(devPath: String, devSize: Int, sufix: String, scorer: PairwiseScorer) {
    val conllEvalScriptPath = "/home/mc/workplace/rand_search/coref/scorer/v7/scorer.pl"
	  val devDocGraphs = prepareTestDocs(devPath, devSize, sufix, false, null);
	  new CorefFeaturizerTrainer().featurizeBasic(devDocGraphs, scorer.featurizer);  // dev docs already know they are dev docs so they don't add features
	  Logger.startTrack("Decoding dev");
	  val basicInferencer = new DocumentInferencerBasic();
	  val (allPredBackptrs, allPredClusterings) = basicInferencer.viterbiDecodeAllFormClusterings(devDocGraphs, scorer);
	  Logger.logss(CorefEvaluator.evaluateAndRender(devDocGraphs, allPredBackptrs, allPredClusterings, Driver.conllEvalScriptPath, "DEV: ", Driver.analysesToPrint));
	  Logger.endTrack();
  }
  
  def prepareTestDocs(devPath: String, devSize: Int, sufix: String, doPrune: Boolean, pruner: CorefPruner): Seq[DocumentGraph] = {
		  val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
		  val devDocs = loadCorefDocsFileMentions(devPath, devSize, sufix, Some(numberGenderComputer));
		  val devDocGraphs = devDocs.map(new DocumentGraph(_, false));
		  EntitySystem.preprocessDocsCacheResources(devDocGraphs);
		  if (doPrune) {
		    val prnr = if (pruner == null) {
		      CorefPruner.buildPruner(Driver.pruningStrategy)
		    } else {
		      pruner
		    }
		    prnr.pruneAll(devDocGraphs);
		  }
		  devDocGraphs;
  }
  
  def loadCorefDocsFileMentions(path: String, size: Int, suffix: String, maybeNumberGenderComputer: Option[NumberGenderComputer]): Seq[CorefDoc] = {
    val docs = ConllDocReader.loadRawConllDocsWithSuffix(path, size, suffix);
    val assembler = new CorefDocAssemblerFile(CorefDocAssembler.getLangPack(Driver.lang), Driver.useGoldMentions);
    val mentionPropertyComputer = new MentionPropertyComputer(maybeNumberGenderComputer);
    val corefDocs = docs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    corefDocs;
  }

  def runBerkSvmTrain(trainPath: String, trainSize: Int, sufix: String,
                        devPath: String, devSize: Int, tstSufix: String) {
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val queryCounts: Option[QueryCountsBundle] = None;
    val trainDocs = loadCorefDocsFileMentions(trainPath, trainSize, sufix, Some(numberGenderComputer));
    // Randomize
    val trainDocsReordered = new scala.util.Random(0).shuffle(trainDocs);
    val lexCutoff = 20 // Driver.lexicalFeatCutoff
    val lexicalCounts = LexicalCountsBundle.countLexicalItems(trainDocs, lexCutoff);
    val semClasser: Option[SemClasser] = Driver.semClasserType match {
      case "basic" => Some(new BasicWordNetSemClasser);
      case e => throw new RuntimeException("Other semclassers not implemented");
    }
    val trainDocGraphs = trainDocsReordered.map(new DocumentGraph(_, true));
    EntitySystem.preprocessDocsCacheResources(trainDocGraphs);
    
    /*
    val prunerScorer = GUtil.load(prunerScrPath).asInstanceOf[PairwiseScorer];
    val pruner = new CorefPrunerTopK(prunerScorer, 30)
    //val pruner = new CorefPrunerTopAlpha(prunerScorer, 0.3)
    pruner.pruneAll(trainDocGraphs);*/
    
    val featureIndexer = new Indexer[String]();
    featureIndexer.getIndex(PairwiseIndexingFeaturizerChao.UnkFeatName);
    val featureSetSpec = FeatureSetSpecification(Driver.pairwiseFeats, Driver.conjScheme, Driver.conjFeats, Driver.conjMentionTypes, Driver.conjTemplates);
    val basicFeaturizer = new PairwiseIndexingFeaturizerChao(featureIndexer, featureSetSpec, lexicalCounts, queryCounts, semClasser);
    val featurizerTrainer = new CorefFeaturizerTrainer();
    featurizerTrainer.featurizeBasic(trainDocGraphs, basicFeaturizer);
    
    PairwiseIndexingFeaturizer.printFeatureTemplateCounts(featureIndexer);

    val trnExs = extractBerkInstances(trainDocGraphs, true);
    
    /////////// test documents
    val devDocGraphs = prepareTestDocs(devPath, devSize, tstSufix,false, null);//true, pruner);//
    featurizerTrainer.featurizeBasic(devDocGraphs, basicFeaturizer);
    val tstExs = extractBerkInstances(devDocGraphs, false);
    
    //val wght = runSvmLearning(trnExs, featureIndexer, tstExs)
    val wght = BerkCorefPerceptron.structurePerceptrion(trnExs, featureIndexer, tstExs)


    runSvmEvaluate(devDocGraphs, wght)
    
    //testPruningRecall(trainDocGraphs, pruner)
    //testPruningRecall(devDocGraphs, pruner)
    
    //doPruning(trnExs, wght, 25)
    //doPruning(tstExs, wght, 25)
    //BerkCorefXgboost.dumpSvmRankFile(trnExs, "ArffDump/onto-berk-featdump-train.txt", true)
    //BerkCorefXgboost.dumpSvmRankFile(tstExs, "ArffDump/onto-berk-featdump-test.txt", true)
    
    /*
    val totalExs = (new ArrayBuffer[SingletonMentionInstance]()) ++ trnExs ++ tstExs;
    val mentIndexing = constructMentionIndexing(totalExs)
    dumpMentionPredictionIgnoreLabel(totalExs, mentIndexing, "mentDumpHot0.2.txt")
    dumpPairScores()
    */
    //val totalExs = (new ArrayBuffer[SingletonMentionInstance]()) ++ trnExs ++ tstExs;
    //dumpMentionPredictionIgnoreLabel(totalExs, "mentDumpHot0.2.txt")
    
    val allDocs = (new ArrayBuffer[DocumentGraph]()) ++ trainDocGraphs ++ devDocGraphs;
    dumpPairScores(allDocs, wght, "mentDumpHot0.2.txt", "mentPairScores1.txt")
  }
  
  def dumpPairScores(docs: Seq[DocumentGraph],
                     weights: Array[Double],
                     pathMent: String,
                     pathPair: String) {
    /*
    val writer = new PrintWriter(path)
    for (mi <- mentions) {
      val ment = mi.graph.getMention(mi.idx)
      val edges = mi.edges
      val docID = ment.rawDoc.getDocNameWithPart()
      //writer.println(docID + "\t" + ment.sentIdx + "\t" + ment.startIdx + "\t" + ment.endIdx + "\t" + ment.headIdx + "\t" + mi.idx)
    }
    writer.close()
    */
    val writerMent = new PrintWriter(pathMent)
    val writerPair = new PrintWriter(pathPair)
    
    var mGlobalIdx = 0;
    for (doc <- docs) {
      val docID = doc.corefDoc.rawDoc.getDocNameWithPart()
      val gidArr = new Array[Int](doc.size);
      for (i <- 0 until doc.size) {
        mGlobalIdx += 1;
        gidArr(i) = mGlobalIdx
      }
      for (i <- 0 until doc.size) {
        // mention dump
        val ment = doc.getMention(i)
        writerMent.println(docID + "\t" + ment.sentIdx + "\t" + ment.startIdx + "\t" + ment.endIdx + "\t" + ment.headIdx + "\t" + gidArr(i))
        // pairs
        val antes = doc.getAllAntecedentsCurrentPruning(i);
        for (j <- antes) {
          val feature = doc.cachedFeats(i)(j)
          val score = SingletonDetection.doDotProduct(weights, feature)
          //val goldEdgeLabel = decideEdgeGoldLabel(doc, i, j)
          //val edgeConsist = if (goldEdgeLabel > 0) true else false
          //val edgeij = new MentionEdge(i, j, feature, goldEdgeLabel, edgeConsist)
          //indexToEdges(i) += edgeij;
          val anteGlbIdx = if (j >= 0) gidArr(j) else -1 
          writerPair.println(gidArr(i) + " " + anteGlbIdx + " " + score)
        }
      }
    }

    writerMent.close()
    writerPair.close()
  }
  
  
  def constructMentionIndexing(mentions: Seq[SingletonMentionInstance]) = {
    val mentIdxing = new HashMap[SingletonMentionInstance, Int]();
    var midx = 0;
    for (mi <- mentions) {
      midx += 1
      mentIdxing += (mi -> midx)
      mi.globalIndex = midx
    }
    mentIdxing
  }

  def dumpMentionPredictionIgnoreLabel(mentions: Seq[SingletonMentionInstance], path: String) {
    val writer = new PrintWriter(path)
    for (mi <- mentions) {
      val ment = mi.graph.getMention(mi.idx)
      val docID = ment.rawDoc.getDocNameWithPart()
      writer.println(docID + "\t" + ment.sentIdx + "\t" + ment.startIdx + "\t" + ment.endIdx + "\t" + ment.headIdx + "\t" + mi.idx)
    }
    writer.close()
  }
  
  def doPruning(exs: Seq[SingletonMentionInstance], weights: Array[Double], topk: Int) {
		  for (ex <- exs) {
			  val sortedList = ex.getScoredAndSortedEdges(weights)
		    val topN = if (sortedList.size > topk) topk else sortedList.size
		    sortedList.map { m => m.isPruned = true }
		    for (k <- 0 until topN) {
					sortedList(k).isPruned = false
				}
		  }
  }

  def doubleArrToFloatArr(weights: Array[Double]): Array[Float] = {
    val fwght = weights.map { d => d.toFloat }// new Array[Float](weights.length)
    fwght
  }
  
  def trainLinearPruner(trainPath: String, trainSize: Int, sufix: String,
                        devPath: String, devSize: Int, tstSufix: String) = {
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val queryCounts: Option[QueryCountsBundle] = None;
    val trainDocs = CorefSystem.loadCorefDocs(trainPath, trainSize, sufix, Some(numberGenderComputer));
    // Randomize
    val trainDocsReordered = new scala.util.Random(0).shuffle(trainDocs);
    val lexicalCounts = LexicalCountsBundle.countLexicalItems(trainDocs, Driver.lexicalFeatCutoff);
    val semClasser: Option[SemClasser] = Driver.semClasserType match {
      case "basic" => Some(new BasicWordNetSemClasser);
      case e => throw new RuntimeException("Other semclassers not implemented");
    }
    val trainDocGraphs = trainDocsReordered.map(new DocumentGraph(_, true));
    EntitySystem.preprocessDocsCacheResources(trainDocGraphs);
    
    val featureIndexer = new Indexer[String]();
    featureIndexer.getIndex(PairwiseIndexingFeaturizerChao.UnkFeatName);
    val featureSetSpec = FeatureSetSpecification(Driver.pairwiseFeats, Driver.conjScheme, Driver.conjFeats, Driver.conjMentionTypes, Driver.conjTemplates);
    val basicFeaturizer = new PairwiseIndexingFeaturizerJoint(featureIndexer, featureSetSpec, lexicalCounts, queryCounts, semClasser);
    val featurizerTrainer = new CorefFeaturizerTrainer();
    featurizerTrainer.featurizeBasic(trainDocGraphs, basicFeaturizer);

    val trnExs = extractBerkInstances(trainDocGraphs, true);
    
    /////////// test documents
    val devDocGraphs = prepareTestDocs(devPath, devSize, tstSufix, false, null);
    featurizerTrainer.featurizeBasic(devDocGraphs, basicFeaturizer);
    val tstExs = extractBerkInstances(devDocGraphs, false);
    
    val wght =  BerkCorefPerceptron.structurePerceptrion(trnExs, featureIndexer, tstExs)
    val fwght = doubleArrToFloatArr(wght)
    val pruneScoer = new PairwiseScorer(basicFeaturizer, fwght).pack;
    
    //val topAlphapruner = new CorefPrunerTopAlpha(pruneScoer, 0.3)
    val topAlphapruner = new CorefPrunerTopK(pruneScoer, 30)
    testPruningRecall(trainDocGraphs, topAlphapruner)
    testPruningRecall(devDocGraphs, topAlphapruner)
    (pruneScoer, topAlphapruner)
  }
  
  def testPruningRecall(docs: Seq[DocumentGraph], pruner: CorefPruner) {
    // pruning on documents
    pruner.pruneAll(docs)
    
    var ok = 0;
    var total = 0;
    for (doc <- docs) {
      for (i <- 0 until doc.size()) {
        val unprunedGold = doc.getGoldAntecedentsUnderCurrentPruningOrEmptySet(i)
        if (!unprunedGold.isEmpty) {
          ok += 1
        }
        total += 1
      }
    }
    
    val zeroOne = ok.toDouble / total.toDouble
    println("pruning recall: " + ok + " / " + total + " = " + zeroOne);
  }
  
  def runSvmEvaluate(devDocGraphs: Seq[DocumentGraph], weights: Array[Double]) {
    val conllEvalScriptPath = "/home/mc/workplace/rand_search/coref/scorer/v7/scorer.pl"
	  Logger.startTrack("Decoding dev");

    val allPredBackptrs = new Array[Array[Int]](devDocGraphs.size);
	  for (i <- 0 until devDocGraphs.size) {
      val dgraph = devDocGraphs(i);
      allPredBackptrs(i) = computeCorefBackPointer(dgraph, weights);
      //allPredBackptrs(i) = computeCorefBackPointerTopkGold(dgraph, weights, 50);
	  }
	  
	  val allPredClusteringsSeq = (0 until allPredBackptrs.length).map(i => OrderedClustering.createFromBackpointers(allPredBackptrs(i)));
	  val allPredClusteringsArr = allPredClusteringsSeq.toArray;

	  Logger.logss(CorefEvaluator.evaluateAndRender(devDocGraphs, allPredBackptrs, allPredClusteringsArr, Driver.conllEvalScriptPath, "DEV: ", Driver.analysesToPrint));
	  Logger.endTrack();
  }
  
  def computeCorefBackPointer(docGraph: DocumentGraph, weights: Array[Double]): Array[Int] = {
		  val corefBackPointer = new Array[Int](docGraph.size);
		  val docInsts = extractSingletonInstancesOneDoc(docGraph, false);
		  for (i <- 0 until docGraph.size) {
		    val bestEdgeIdx = docInsts(i).getBestEdgeIndex(weights)
		    //val bestEdgeIdx = docInsts(i).getBestGoldEdgeIndex(weights)
			  val bestEdge = docInsts(i).edges(bestEdgeIdx)
			  corefBackPointer(i) = bestEdge.anteIdx;
		  }
		  corefBackPointer
  }
  
  def computeCorefBackPointerTopkGold(docGraph: DocumentGraph, weights: Array[Double], topk: Int): Array[Int] = {
		  val corefBackPointer = new Array[Int](docGraph.size);
		  val docInsts = extractSingletonInstancesOneDoc(docGraph, false);
		  for (i <- 0 until docGraph.size) {
		    val ex = docInsts(i)
			  val sortedList = ex.getScoredAndSortedEdges(weights)
		    val topN = if (sortedList.size > topk) topk else sortedList.size
				var bestPredEdge: MentionEdge = sortedList(0);
		    var bestGoldEdge: MentionEdge = null;
		    for (j <- 0 until topN) {
					if (sortedList(j).goldLb > 0) {
					  bestGoldEdge = sortedList(j)
					}
				}
			  val bestEdge = if (bestGoldEdge == null) {
			    bestPredEdge
			  } else {
			    bestGoldEdge
			  }
			  corefBackPointer(i) = bestEdge.anteIdx;
		  }
		  corefBackPointer
  }
  
  //////////////////////////////////////////////////////////
  //// Singleton Instances
  //////////////////////////////////////////////////////////
  
  //// SVM
  
  def constructBerkProblem(exs: Seq[SingletonMentionInstance]): SLProblem = {
    val sp: SLProblem = new SLProblem();
    for (ex <- exs) {
      val goldOutput = new BerkSvmCorefOutput(-1); // a randome output ...
      sp.addExample(ex, goldOutput);
    }
    sp;
  }
  
  def runSvmLearning(trainExs: Seq[SingletonMentionInstance], 
                     featIndexer: Indexer[String],
                     testExs: Seq[SingletonMentionInstance]): Array[Double] = {
    
		val trainTimer = new MyTimeCounter("Training time");
	  trainTimer.start();
    
    
    val model = new SingletonSvmModel();
    
    val spTrain = constructBerkProblem(trainExs);

    val featGenr = new BerkSvmCorefFeatGener();
    val latentInfr = new BerkSvmCorefInferencer();
    
    val slcfgPath = "/home/mc/workplace/rand_search/sl-config/ontonotes-berk-DCD.config"; 
    val para = new SLParameters();
    para.loadConfigFile(slcfgPath);
    para.TOTAL_NUMBER_FEATURE = featIndexer.size();

    val baseLearner: Learner = LearnerFactory.getLearner(latentInfr, featGenr, para);
    
    //// about latent settings
    val latentPara = new SLParameters();
    latentPara.TOTAL_NUMBER_FEATURE = featIndexer.size();
    latentPara.MAX_NUM_ITER = 1;
    
    val latentLearner = new GeneralLatentSvmLearner(baseLearner, featGenr, latentPara, latentInfr);
    model.infSolver = latentInfr;
    model.wv = latentLearner.train(spTrain, new WeightVector(latentPara.TOTAL_NUMBER_FEATURE), testExs, runHammingEvaluation)
    WeightVector.printSparsity(model.wv);
    
    latentInfr.printCount();
    println("C = " + baseLearner.getParameters.C_FOR_STRUCTURE);
    println("Iterations = " + latentPara.MAX_NUM_ITER);
    
    // training time count~
    trainTimer.end();
    trainTimer.printSecond("Training time");

    // save the model
    model.convertDoubleWeightArray()
  }


  def runHammingEvaluation(testExs: Seq[IInstance], wv: WeightVector) {
    val weights: Array[Double] = wv.getDoubleArray
    var tpos = 0
    for (ex <- testExs) {
      val pm = ex.asInstanceOf[SingletonMentionInstance]
    	val predEdge = pm.getBestEdgeIndex(weights)
    	if (pm.edges(predEdge).goldLb > 0) {
    		tpos += 1
    	}
    }
    
    val zeroOne = tpos.toDouble / (testExs.size).toDouble
    println("0-1Accuracy: " + tpos + " / " + (testExs.size) + " = " + zeroOne);
  }

  
  //////////////////////
  //// Learning
  //////////////////////
  
  def bool2int(b:Boolean) = if (b) 1 else 0
  
  def extractBerkInstances(docs: Seq[DocumentGraph], addFeat: Boolean) = {
    val singletonInsts = new ArrayBuffer[SingletonMentionInstance]();
    for (d <- docs) {
      val docInsts = extractSingletonInstancesOneDoc(d, addFeat);
      singletonInsts ++= docInsts;
    }

    singletonInsts.toSeq
  }
  
  def extractSingletonInstancesOneDoc(doc: DocumentGraph, addFeat: Boolean): ArrayBuffer[SingletonMentionInstance] = {
    val ies = new ArrayBuffer[ArrayBuffer[MentionEdge]](doc.size());
    for (i <- 0 until doc.size()) {
      ies += (new ArrayBuffer[MentionEdge]());
    }
    val indexToEdges = ies.toArray
    
    for (i <- 0 until doc.size) {
      val antes = doc.getAllAntecedentsCurrentPruning(i);
      //val antes = doc.getAllAntecedentsNoPruning(i);
      for (j <- antes) {
          val feature = doc.cachedFeats(i)(j)
          val goldEdgeLabel = decideEdgeGoldLabel(doc, i, j)
          val edgeConsist = if (goldEdgeLabel > 0) true else false
          val edgeij = new MentionEdge(i, j, feature, goldEdgeLabel, edgeConsist)
          indexToEdges(i) += edgeij;
      }
    }
    
    //// features
    val insts = new ArrayBuffer[SingletonMentionInstance]()
    for (i <- 0 until doc.size()) {
      val edges = indexToEdges(i)
      val goldEdges = indexToEdges(i).filter { e => (e.goldLb > 0) }
      //if (goldEdges.size > 0) {
    	  val mentInst = new SingletonMentionInstance(i, doc, indexToEdges(i))
    		insts += (mentInst);
      //}
    }
    
    insts
  }
  
  def decideEdgeGoldLabel(doc: DocumentGraph, curIdx: Int, anteIdx: Int): Int = {
    
    val mention = doc.getMention(curIdx)
    val matched = hasMatchedGoldMention(doc, mention)
    
    /*
    // span is matched
    val label = if (matched) {
      if (anteIdx == curIdx) {
        0
      } else {
        bool2int(doc.isGoldNoPruning(curIdx, anteIdx))
      }
    // span not matched at all
    } else {
      0
    }
    */
    val label = bool2int(doc.isGoldCurrentPruning(curIdx, anteIdx))
    label
  }
  
  def hasMatchedGoldMention(doc: DocumentGraph, pm: Mention): Boolean = {
    val goldMents = doc.corefDoc.goldMentions;
    val matched = goldMents.filter { m => ((m.sentIdx == pm.sentIdx) && (m.startIdx == pm.startIdx) && (m.endIdx == pm.endIdx)) }
    if (matched.size > 0) {
      return true
    } else {
      return false
    }
  }


}

object BerkCorefXgboost {
  
  def main(args: Array[String]) {
    val prdr = new StateActionXgboostPredictor();
		prdr.trainRanker("policy_scoring_xgboost_beam1_ontonotes5.txt",
                      "ArffDump/onto-berk-featdump-train.txt", 
                      "ArffDump/onto-berk-featdump-test.txt", 1);
  }
  
  def dumpSvmRankFile(exs: Seq[SingletonMentionInstance], fn: String, doPrune: Boolean) {
    val featDumper = new PrintWriter(fn)
    var qid = 0
    for (ex <- exs) {
      qid += 1
      val edges = ex.edges
      for (edge <- edges) {
    	  if (doPrune) {
    		  if (!edge.isPruned) {
    			  val svmrankLine = printForSvmRank(edge.goldLb, qid, edge.features);
    			  featDumper.println(svmrankLine);
    		  }
    	  } else {
    		  val svmrankLine = printForSvmRank(edge.goldLb, qid, edge.features);
    		  featDumper.println(svmrankLine);
    	  }
      }
    }
    featDumper.close()
  }
  
  def printForSvmRank(rank: Int, qid: Int, featIdx: Array[Int]): String = {
    val sb = new StringBuilder();
		// rank
		sb.append(rank);
		// qid
		sb.append(" qid:"+qid);	
		// feature vector
		for (k <- 0 until featIdx.length) {
			sb.append(" "+(featIdx(k)+1)+":"+1);
		}
		sb.toString()
  }
}

object BerkCorefPerceptron {
  
  // structural learning
	def structurePerceptrion(allTrains: Seq[SingletonMentionInstance], 
			                     featIndexer: Indexer[String],
			                     testExs: Seq[SingletonMentionInstance]): Array[Double] = {

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

				//println("Iter " + iter);
				var exId = 0;
				for (example <- allTrains) {

					exId += 1;

          val predBestOutput = example.getBestEdgeIndex(weight); // my prediction
					val goldBestOutput = example.getBestGoldEdgeIndex(weight);  // gold best
          
					// update?
					if (!example.edgeIndexIsCorrect(predBestOutput)) {
						updateCnt += 1;
						if (updateCnt % 10000 == 0) println("Update " + updateCnt);
            
            val featGold = example.getFeaturize(goldBestOutput);
            val featPred = example.getFeaturize(predBestOutput);
            
						updateWeightStruct(weight, 
								               featGold,
								               featPred,
								               learnRate,
								               lambda);
						sumWeight(weightSum, weight);
					}
				}

				///////////////////////////////////////////////////
				// have a test after each iteration (for learning curve)
				val tmpAvg = new Array[Double](weightSum.size)
				Array.copy(weightSum, 0, tmpAvg, 0, weightSum.size);
				divdeNumber(tmpAvg, updateCnt.toDouble);

				//quickTest(allTrains, tmpAvg, wikiDB);
        quickTest(allTrains, tmpAvg);
				quickTest(testExs, tmpAvg);
				quickTestTopK(allTrains, tmpAvg, 100)
				quickTestTopK(testExs, tmpAvg, 100)
				println("Iter " + iter + " Update Cnt = " + (updateCnt - lastUpdtCnt));
			}

			divdeNumber(weightSum, updateCnt.toDouble);
			weightSum;
	}
	
	def quickTest(exs: Seq[SingletonMentionInstance], weights: Array[Double]) {
		var sumTotal : Double = 0;
	  var sumErr: Double = 0
		for (ex <- exs) {
			val predBestOutput = ex.getBestEdgeIndex(weights); // my prediction
			val err = if (ex.edgeIndexIsCorrect(predBestOutput)) 0 else 1
			val total = 1;
			sumErr += err.toDouble;
			sumTotal += total.toDouble;
		}

	  val crct = sumTotal - sumErr;
	  val acc = crct / sumTotal;
	  println("quick test: 01-Acc = " + crct + "/" + sumTotal + " = " + acc);
	}
	
  def quickTestTopK(exs: Seq[SingletonMentionInstance], weights: Array[Double], topk: Int) {
		var sumTotal : Double = 0;
		val accAtK = new Array[Int](5000)
		val accAtAlpha = new Array[Int](11)
		Arrays.fill(accAtK, 0)
		Arrays.fill(accAtAlpha, 0)
	  var sumErr: Double = 0
		for (ex <- exs) {
		  val graphSize = ex.graph.size()
		  val sortedList = ex.getScoredAndSortedEdges(weights)
		  var acc: Int = 0
		  for (j <- 0 until sortedList.size) {
		    if (sortedList(j).goldLb > 0) acc = 1
		    accAtK(j) += acc
		  }
		  for (j <- sortedList.size until 5000) {
		    accAtK(j) += acc
		  }
			sumTotal += 1;
			//// accAtAlpha
			for (j <- 1 to 10) {
			  val tpk = (graphSize.toDouble * j.toDouble * 0.1).toInt
			  accAtAlpha(j) = accAtK(tpk)
			}
		}

		val recallAtK = new Array[Double](5001)
		for (j <- 0 until 5000) {
		  recallAtK(j) = accAtK(j).toDouble / sumTotal.toDouble
		}
		println("========")
		for (j <- 0 until 100) {
		  println("quick test: Recall@" + (j+1) + ": " + accAtK(j) + "/" + sumTotal + " = " + recallAtK(j));
		}
		println("========")
		var j2: Double = 0;
		for (j <- 1 to 10) {
		  j2 += 0.1
		  println("quick test: Recall@" + "Alpha" + j2 + ": " + accAtAlpha(j) + "/" + sumTotal + " = " + accAtAlpha(j).toDouble / sumTotal.toDouble);
		}
	}

	def updateWeightStruct(currentWeight: Array[Double], 
			            featGold: HashMap[Int,Double],//Array[Int],
                  featPred: HashMap[Int,Double],//Array[Int],
                  eta: Double,
                  lambda: Double) {
		var gradient = Array.fill[Double](currentWeight.length)(0);//new Array[Double](currentWeight.length);
		for ((i, vgold) <- featGold) {
			gradient(i) += (vgold);
		}
		for ((j, vpred) <- featPred) {
			 gradient(j) -= (vpred);
		}

		// do L2 Regularization
		//var l1norm = getL1Norm(currentWeight);
		for (i2 <- 0 until currentWeight.length) {

//			var reg: Double = 1.0 - (eta * lambda)
//					var curWeightVal = currentWeight(i2) * reg;
//		currentWeight(i2) = curWeightVal + (gradient(i2) * eta);
		  currentWeight(i2) += (gradient(i2) * eta);
		}
	}

	def computeScore(wght: Array[Double], feat: Array[Int]) : Double = {
			var result : Double = 0;
	for (idx <- feat) {
		if (idx >= 0) {
			result += (wght(idx));
		}
	}
	result;
	}

	def computeScoreSparse(wght: Array[Double], featSparse: HashMap[Int, Double]) : Double = {
			var result : Double = 0;
	  for ((idx,value) <- featSparse) {
		  //if (idx >= 0) {
			result += (wght(idx) * value);
		  //}
	  }
	result;
	}

	def sumWeight(sum: Array[Double], w: Array[Double]) {
		for (i <- 0 until w.length) {
			sum(i) += w(i);
		}
	}

	def divdeNumber(w: Array[Double], deno: Double) {
		for (i <- 0 until w.length) {
			w(i) = (w(i) / deno);
		}
	}

	def getL1Norm(w: Array[Double]): Double = {
			var norm: Double = 0;
	for (v <- w) {
		norm += Math.abs(v);
	}
	return norm;
	}
	
	  
  def extendWeight(w: Array[Double], newLen: Int): Array[Double] = {
    val newW = new Array[Double](newLen);
    Arrays.fill(newW, 0);
    for (i <- 0 until w.length) {
      newW(i) = w(i);
    }
    newW;
  }
  
  
}