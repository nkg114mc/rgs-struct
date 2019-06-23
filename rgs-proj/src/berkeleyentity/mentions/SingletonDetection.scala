package berkeleyentity.mentions

import java.util.ArrayList
import java.util.{HashMap => JHashMap}
import java.util.HashSet
import java.util.List
import java.util.Random

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.util.control.Breaks._

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
import berkeleyentity.randsearch.AceCorefInferencer
import berkeleyentity.sem.BasicWordNetSemClasser
import berkeleyentity.sem.QueryCountsBundle
import berkeleyentity.sem.SemClasser
import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.futile.util.Counter
import edu.berkeley.nlp.futile.util.Logger
import edu.illinois.cs.cogcomp.sl.core.AbstractFeatureGenerator
import edu.illinois.cs.cogcomp.sl.core.AbstractInferenceSolver
import edu.illinois.cs.cogcomp.sl.core.IInstance
import edu.illinois.cs.cogcomp.sl.core.IStructure
import edu.illinois.cs.cogcomp.sl.core.SLModel
import edu.illinois.cs.cogcomp.sl.core.SLParameters
import edu.illinois.cs.cogcomp.sl.core.SLProblem
import edu.illinois.cs.cogcomp.sl.learner.Learner
import edu.illinois.cs.cogcomp.sl.learner.LearnerFactory
import edu.illinois.cs.cogcomp.sl.util.FeatureVectorBuffer
import edu.illinois.cs.cogcomp.sl.util.IFeatureVector
import edu.illinois.cs.cogcomp.sl.util.WeightVector
import elearning.ElearningArg
import experiment.CostFuncCacherAndLoader
import experiment.RndLocalSearchExperiment.InitType
import experiment.RndLocalSearchExperiment.MulLbLossType
import general.AbstractFactorGraph
import general.AbstractFeaturizer
import general.AbstractInstance
import general.AbstractOutput
import general.FactorGraphBuilder.FactorGraphType
import init.RandomStateGenerator
import init.UniformRndGenerator
import search.GreedySearcher
import search.SeachActionGenerator
import search.SearchAction
import search.SearchState
import search.ZobristKeys
import sequence.hw.HwInstance
import sequence.hw.HwOutput
import sequence.hw.HwSegment
import scala.collection.mutable.HashMap
import search.loss.SearchLossExmpAcc
import search.loss.SearchLossExmpF1
import search.loss.SearchLossHamming
import general.AbstractLossFunction

class MentionEdge(val currIdx: Int, 
                  val anteIdx: Int, 
                  val features: Array[Int],
                  val goldLb: Int,
                  val consistWithGold: Boolean) {
  
  var predictScore: Double = 0
  var isPruned: Boolean = false
  
}

//////////////////////////////////////
// for UIUC Structural Learning Lib //
//////////////////////////////////////

class SingletonMentionInstance(val idx: Int,
		                    val graph: DocumentGraph,
		                    val edges: ArrayBuffer[MentionEdge])  extends IInstance {//,
		                    //val goldLabel: Int) extends IInstance {
  
  var globalIndex: Int = -1;

  def size() = {
    edges.size
  }
  def checkEdgesHasGold(): Boolean = {
    val consists = edges.filter { e => e.consistWithGold }
    val hasGold = (consists.size > 0)
    assert(hasGold)
    hasGold
  }
  
  def getDomain(useGold: Boolean): Array[Int] = {
    val domains = new ArrayBuffer[Int]();
    for (i <- 0 until edges.size) {
    	if (useGold) {
    		if (edges(i).consistWithGold) {
    			domains += i
    		}
    	} else {
    		domains += i
    	}
    }
    domains.toArray
  }
  
  def getBestEdgeIndex(weights: Array[Double]): Int = {
	  var bestSc: Float = -Float.MaxValue;
    var besteIdx: Int = -1;
    for (i <- 0 until edges.size) {
	    val e = edges(i)
			 val score = SingletonDetection.doDotProduct(weights, e.features)
			 if (score > bestSc) {
				bestSc = score
				besteIdx = i
			 }
    }
    (besteIdx);
  }
  
  def getBestGoldEdgeIndex(weights: Array[Double]): Int = {
	  var bestSc: Float = -Float.MaxValue;
    var besteIdx: Int = 0;
    for (i <- 0 until edges.size) {
	    val e = edges(i)
	    if (e.goldLb > 0) {
	    	val score = SingletonDetection.doDotProduct(weights, e.features)
	    	if (score > bestSc) {
	    		bestSc = score
	    		besteIdx = i
	    	}
	    }
    }
    (besteIdx);
  }
  
  def getScoredAndSortedEdges(weights: Array[Double]) = {
    val edgeList = new ArrayBuffer[MentionEdge]()
    for (i <- 0 until edges.size) {
	    val e = edges(i)
	    e.predictScore = SingletonDetection.doDotProduct(weights, e.features)
	    edgeList += e
    }
    val sorted = edgeList.toList.sortWith(_.predictScore > _.predictScore)
    sorted
  }
  
  def getSortedEdgesGivenScore() = {
    val edgeList = new ArrayBuffer[MentionEdge]()
    for (i <- 0 until edges.size) {
	    edgeList += edges(i)
    }
    val sorted = edgeList.toList.sortWith(_.predictScore > _.predictScore)
    sorted
  }
  
  def edgeIndexIsCorrect(edgeIdx: Int): Boolean = {
	  val e = edges(edgeIdx)
	  if (e.goldLb > 0) {
	    return true
    } else {
      return false
    }
  }
  
  def getFeaturize(edgeIdx: Int) = {
		  val valueMap = new HashMap[Int,Double]();
		  val feats = edges(edgeIdx).features;
		  for (idx <- feats) {
			  addValueToVector(valueMap, idx, 1.0);
		  }
		  valueMap; // return a sparse feature vector
  }
  
  def addValueToVector(myMap: HashMap[Int,Double], index: Int, value: Double) {
		val maybeVal = myMap.get(index);
		if (maybeVal == None) {
			myMap += (index -> 1.0);
		} else {
			val newV = maybeVal.get + 1.0;
			myMap += (index -> newV);
		}
	}
  
  def predictLabel(weights: Array[Double]): Int = {
    	var bestSc: Float = -Float.MaxValue;
			var besteIdx: Int = -1;
			for (i <- 0 until edges.size) {
			  val e = edges(i)
			  val score = SingletonDetection.doDotProduct(weights, e.features)
			  if (score > bestSc) {
			    bestSc = score
			    besteIdx = i
			  }
			}
			
			val predLab = if (edges(besteIdx).currIdx == edges(besteIdx).anteIdx) {
			  0
			} else {
			  1
			}

			(predLab);
  }

}
//(val ments: Array[SingletonMentionInstance]) 
class SingletonInstance(segs: List[HwSegment], albt: Array[String]) extends HwInstance(segs, albt) {
  
  var mentions: Array[SingletonMentionInstance] = null;
  
  def this(ments: Array[SingletonMentionInstance]) {
    this(null,null)
    mentions = ments;
  }
  
  override def size() = {
    mentions.length
  }
  
  def getDomain(idx: Int, useGold: Boolean): Array[Int] = {
    val men = mentions(idx)
    men.getDomain(useGold);
  }
}

/*
class SingletonOutput(var output: Array[Int]) extends HwOutput {
    
	def setOutput(newOutput: Array[Int]) {
		if (newOutput.length != output.length) {
			throw new RuntimeException("Wrong length for output: " + newOutput.length);
		}
		output = newOutput;
	}

	override def equals(o1: Any): Boolean = {
			val vec = o1.asInstanceOf[SingletonOutput];
			if (vec.output.length != output.length) {
				throw new RuntimeException("Wrong length for output: " + vec.output.length);
			}
			output.corresponds(vec.output){_ == _};
	}
}
*/

class SingletonSvmModel extends SLModel {
  
  
  def svmLoadModel(fn: String): Array[Double] =  {
    SLModel.loadModel(fn);
    convertDoubleWeightArray();
  }
  
  def convertDoubleWeightArray(): Array[Double] = {
		val farr = wv.getWeightArray; // (0) is bias
    var darr = new Array[Double](farr.length);
	  for (i <- 0 until darr.length) {
		  darr(i) = farr(i).toDouble;
	  }
	  darr;
  }
  
}


class SingletonFeatGener extends AbstractFeaturizer {
  
	override def getFeatureVector(xi: IInstance, yhati: IStructure): IFeatureVector = {
			val x = xi.asInstanceOf[SingletonInstance]; 
			val y = yhati.asInstanceOf[HwOutput];
			val sparseValues = featurize(x,y);
			val fb = new FeatureVectorBuffer();
			for (idx <- sparseValues.keySet().asScala) {
				fb.addFeature(idx, sparseValues.get(idx));			
			}
			fb.toFeatureVector();
	}
	
	def addValueToVector(myMap: JHashMap[Integer,java.lang.Double], index: Int, value: Double) {
		if (myMap.containsKey(index)) {
			val newV = myMap.get(index).doubleValue() + value;
			myMap.put(index, newV);
		} else {
			myMap.put(index, value);
		}
	}

  override def featurize(x1: AbstractInstance, x2: AbstractOutput): java.util.HashMap[Integer,java.lang.Double] = {
    val x = x1.asInstanceOf[SingletonInstance]; 
			val y = x2.asInstanceOf[HwOutput];


			val sparseValues = new JHashMap[Integer, java.lang.Double]();

			for (i <- 0 until x.size()) {
				val bestEdge = x.mentions(i).edges(y.output(i))
						val edgeFeature = bestEdge.features
						for (idx <- edgeFeature) {
							addValueToVector(sparseValues, idx, 1.0);
						}
			}
sparseValues;
  }
  override def getFeatLen(): Int = 0
	
	override def getFeatureVectorDiff(x: IInstance, y1: IStructure, y2: IStructure): IFeatureVector = {
		val f1 = getFeatureVector(x, y1);
		val f2 = getFeatureVector(x, y2);		
		return f1.difference(f2);
	}
}

class SingletonActionGenerator(val useGold: Boolean) extends SeachActionGenerator {

	override def genAllAction(instace: AbstractInstance, currState: SearchState) = {
	
		val hotInst = instace.asInstanceOf[SingletonInstance];
	
		val actions = new ArrayList[SearchAction]();
		assert (currState.structOutput.size() == instace.size());
		
		for (i <- 0 until currState.structOutput.size()) {
			
			val domains =  hotInst.getDomain(i, useGold); // gold actions only
			
			for (j <- 0 until domains.length) {
				val oldv = currState.structOutput.getOutput(i);
				val newv = domains(j);
				if (oldv != newv) {
					val act = new SearchAction(i, newv, oldv);
					actions.add(act);
				}
			}
		}

		actions;
	}
}

class SingletonUniformRndGenerator(val rnd: Random, 
                                   val useGold: Boolean) extends UniformRndGenerator(rnd) {

	override def generateRandomInitState(inst: AbstractInstance, stateNum: Int) = {
		
		val x = inst.asInstanceOf[SingletonInstance];
		val rndSet = new HashSet[SearchState]();
		
		// repeat
		var cnt = 0;
		breakable {
			while (true) {

				// get a uniform state
				val output = new HwOutput(inst.size(), x.alphabet);
				output.output(0) = 0;
				for (j <- 0 until output.size()) {
					val domain = x.getDomain(j, useGold)
					val vidx = UniformRndGenerator.getValueIndexUniformly(domain.length, random);// purely uniform
					output.output(j) = domain(vidx);
				}

				val s = new SearchState(output);
				if (!rndSet.contains(s)) {
					rndSet.add(s);
					cnt += 1;
				}

				if (cnt >= stateNum) {
					break;
				}
			}
		}
		rndSet;
	}
}

class SingletonFactorGraph(insti: AbstractInstance, ftzri: AbstractFeaturizer) extends AbstractFactorGraph {
	
	val instance = insti.asInstanceOf[SingletonInstance];
	val featurizer = ftzri.asInstanceOf[SingletonFeatGener];
	
	val cachedUnaryScores = Array.ofDim[Double](instance.size() + 1, instance.size() + 1);
	var cachedScore = 0;

	override def updateScoreTable(weights: Array[Double]) {
		for (i <- 0 until instance.size()) {
			val domains = instance.getDomain(i, false);
			for (j <- 0 until domains.length) {
				val jvIdx = domains(j);
				val unaryFeat = instance.mentions(i).edges(jvIdx).features  //instance.getMentPairFeature(i, jvIdx);
				val sc = SingletonDetection.doDotProduct(weights, unaryFeat);
				cachedUnaryScores(i)(jvIdx) = sc
			}
		}
	}

	override def computeScoreWithTable(weights: Array[Double], output: HwOutput): Double = {
		var score: Double = 0;
		for (i <- 0 until instance.size()) {
			score += cachedUnaryScores(i)(output.getOutput(i));
		}
		return score;
	}

	override def computeScoreDiffWithTable(weights: Array[Double], action: SearchAction, output: HwOutput): Double = {
		
		var scoreDiff: Double = 0;
		
		val vIdx = action.getSlotIdx();
		val oldv = action.getOldVal();
		val newv = action.getNewVal();

		// store origin value
		val originValue = output.getOutput(vIdx);
		assert (newv == originValue);
		
		// unary
		scoreDiff -= cachedUnaryScores(vIdx)(oldv);
		scoreDiff += cachedUnaryScores(vIdx)(newv);
		
		return scoreDiff;
	}

	override def computeScore(weights: Array[Double], output: HwOutput): Double = {
		throw (new RuntimeException("Not implemented!"));
	}

	override def getCachedScore(): Double = {
		throw (new RuntimeException("Not implemented!"));
	}

}

class SingletonInferencer() {// extends AbstractLatentInferenceSolver {

	val callCount = new Counter[Int](); // for efficiency statistic

/*
	def getBestStructure(wi: WeightVector, xi: IInstance): IStructure = {
			callCount.incrementCount(1, 1);
			getLossAugmentedBestStructure(wi, xi, null);
	}

	def getLoss(xi: IInstance, ystari: IStructure, yhati: IStructure): Float = {
			val x = xi.asInstanceOf[SingletonInstance];
			val ystar = ystari.asInstanceOf[SingletonOutput];
			val yhat = yhati.asInstanceOf[SingletonOutput];
			val loss = getWeightedZeroOneError(yhat, ystar);
			loss.toFloat; // non-normalized loss
	}
	
	def getWeightedZeroOneError(pred: SingletonOutput, gold: SingletonOutput): Float = {
	  if (pred.output == gold.output) {
	    return 0f;
	  } else {
	    return 1.0f;
	  }
	}

	def getLossAugmentedBestStructure(wi: WeightVector, xi: IInstance, ystari: IStructure): IStructure = {

			callCount.incrementCount(0, 1);
			val example = xi.asInstanceOf[SingletonInstance];
			val ystar = ystari.asInstanceOf[SingletonOutput];

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

			(new SingletonOutput(besteIdx));
	}
	


	override def getBestLatentStructure(weight: WeightVector, ins: IInstance, gold: IStructure): IStructure = {
			callCount.incrementCount(2, 1);
			val example = ins.asInstanceOf[SingletonInstance];
			val ystar = gold.asInstanceOf[SingletonOutput];

			val wgtArr = weight.getDoubleArray();
			
			var bestSc: Float = -Float.MaxValue;
			var besteIdx: Int = -1;
			for (i <- 0 until example.edges.size) {
			  val e = example.edges(i)
			  if (e.consistWithGold) {
				  val lossAugScore = SingletonDetection.doDotProduct(wgtArr, e.features)
					if (lossAugScore > bestSc) {
						bestSc = lossAugScore
						besteIdx = i
					}
			  }
			}

			(new SingletonOutput(besteIdx));
	}
*/

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


object SingletonDetection {
  
	def doDotProduct(weights: Array[Double], feature: Array[Int]): Float = {
		var sum: Float = 0
		for (idx <- feature) {
			sum += (weights(idx).toFloat)
		}
	  return sum
	}
  
  val trainSuffix = "v9_auto_conll"
  val testSuffix = "v9_auto_conll"
  
  def main(args: Array[String]) {
    runBerkOntoCoref()
  }
  
  def runBerkOntoCoref() {
    val trainPath = "/home/mc/workplace/rand_search/coref/berkfiles/data/ontonotes5/test";//train";
    val suffix1 = "v4_auto_conll"
    val testPath = "/home/mc/workplace/rand_search/coref/berkfiles/data/ontonotes5/test";
    val suffix2 = "v9_auto_conll"
    
    //Driver.useGoldMentions = true
    val modelPath = "model/berkOnto.model";
    //runTrainEvaluate(trainPath, -1, testPath, -1, modelPath)
    //runEvaluateGivenModel(testPath, -1, testSuffix, modelPath)
    //runSingletonTrain(testPath, -1, testSuffix)
    runSingletonTrain(trainPath, -1, trainSuffix, testPath, -1, testSuffix)
  }
  
  def runEvaluateGivenModel(devPath: String, devSize: Int, sufix: String, modelPath: String) {
    val scorer = GUtil.load(modelPath).asInstanceOf[PairwiseScorer]
    runEvaluate(devPath, devSize, sufix, scorer)
  }
  
  def runEvaluate(devPath: String, devSize: Int, sufix: String, scorer: PairwiseScorer) {
    val conllEvalScriptPath = "/home/mc/workplace/rand_search/coref/scorer/v7/scorer.pl"
	  val devDocGraphs = prepareTestDocs(devPath, devSize, sufix);
	  new CorefFeaturizerTrainer().featurizeBasic(devDocGraphs, scorer.featurizer);  // dev docs already know they are dev docs so they don't add features
	  Logger.startTrack("Decoding dev");
	  val basicInferencer = new DocumentInferencerBasic();
	  val (allPredBackptrs, allPredClusterings) = basicInferencer.viterbiDecodeAllFormClusterings(devDocGraphs, scorer);
	  Logger.logss(CorefEvaluator.evaluateAndRender(devDocGraphs, allPredBackptrs, allPredClusterings, Driver.conllEvalScriptPath, "DEV: ", Driver.analysesToPrint));
	  Logger.endTrack();
  }
  
  def prepareTestDocs(devPath: String, devSize: Int, sufix: String): Seq[DocumentGraph] = {
		  val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
		  val devDocs = CorefSystem.loadCorefDocs(devPath, devSize, sufix, Some(numberGenderComputer));
		  val devDocGraphs = devDocs.map(new DocumentGraph(_, false));
		  EntitySystem.preprocessDocsCacheResources(devDocGraphs);
		  CorefPruner.buildPruner(Driver.pruningStrategy).pruneAll(devDocGraphs);
		  devDocGraphs;
  }

  def runSingletonTrain(trainPath: String, trainSize: Int, sufix: String,
                        devPath: String, devSize: Int, tstSufix: String) {
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
    CorefPruner.buildPruner(Driver.pruningStrategy).pruneAll(trainDocGraphs);
    
    val featureIndexer = new Indexer[String]();
    featureIndexer.getIndex(PairwiseIndexingFeaturizerJoint.UnkFeatName);
    val featureSetSpec = FeatureSetSpecification(Driver.pairwiseFeats, Driver.conjScheme, Driver.conjFeats, Driver.conjMentionTypes, Driver.conjTemplates);
    val basicFeaturizer = new PairwiseIndexingFeaturizerJoint(featureIndexer, featureSetSpec, lexicalCounts, queryCounts, semClasser);
    val featurizerTrainer = new CorefFeaturizerTrainer();
    featurizerTrainer.featurizeBasic(trainDocGraphs, basicFeaturizer);
    PairwiseIndexingFeaturizer.printFeatureTemplateCounts(featureIndexer)

    val trnExs = extractSingletonInstances(trainDocGraphs, true);
    
    /////////// test documents
    val devDocGraphs = prepareTestDocs(devPath, devSize, tstSufix);
    featurizerTrainer.featurizeBasic(devDocGraphs, basicFeaturizer);
    val tstExs = extractSingletonInstances(devDocGraphs, false);
    
    
    //val wght = uiucStructLearning(trnExs, featureIndexer, tstExs)
    
    val eLnArg = new ElearningArg();
		val costchr = new CostFuncCacherAndLoader(CostFuncCacherAndLoader.defaultFolder);

		val wght = runStructLearning(trnExs, tstExs, featureIndexer,
                                 InitType.UNIFORM_INIT, MulLbLossType.EXMPF1_LOSS, 1,
			                           eLnArg, costchr)
  }
  
  
  
  //////////////////////////////////////////////////////////
  //// Singleton Instances
  //////////////////////////////////////////////////////////
  
  //// SVM
  
  def constructProblem(exs: Seq[SingletonInstance]): SLProblem = {
    val sp: SLProblem = new SLProblem();
    for (ex <- exs) {
      val goldOutput = new HwOutput(ex.size(), null); // a randome output ...
      sp.addExample(ex, goldOutput);
    }
    sp;
  }
  
/*
  def uiucStructLearning(trainExs: Seq[SingletonInstance], 
                         featIndexer: Indexer[String],
                         testExs: Seq[SingletonInstance]): Array[Double] = {
    
		val trainTimer = new MyTimeCounter("Training time");
	  trainTimer.start();
    
    
    val model = new SingletonSvmModel();
    
    val spTrain = constructProblem(trainExs);

    val featGenr = new SingletonFeatGener();
    val latentInfr = new SingletonInferencer();
    
    val slcfgPath = "/home/mc/workplace/rand_search/sl-config/ontonotes-singleton-DCD.config"; 
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
    model.wv = latentLearner.train(spTrain, new WeightVector(latentPara.TOTAL_NUMBER_FEATURE), testExs);
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
*/
  def runStructLearning(trnInsts: Seq[SingletonInstance], tstInsts: Seq[SingletonInstance], 
      featIndexer: Indexer[String],
                        initType: InitType, lossTyp: MulLbLossType, randomNum: Int,
			                  evalLearnArg: ElearningArg, costCacher: CostFuncCacherAndLoader) = {

		val nrnd = randomNum;
		val dsName = "Singleton";
		val configPath = "/home/mc/workplace/rand_search/sl-config/ontonotes-singleton-DCD.config"; 
		
		// load data
		println("Train docs = " + trnInsts.size);
		
		
		var initStateGener: RandomStateGenerator = null;
		var initStateGenerGold: RandomStateGenerator = null;
		println("InitType = " + initType.toString());
		if (initType == InitType.UNIFORM_INIT) {
			initStateGener = new SingletonUniformRndGenerator(new Random(), false);
			initStateGenerGold = new SingletonUniformRndGenerator(new Random(), true);
		} else if (initType == InitType.LOGISTIC_INIT) {
			//initStateGener = MultiLabelSamplingRndGenerator.loadGenrIfExist(modelLogisticFn, dsName, trtstInsts.get(0), trtstInsts.get(1), Label.MULTI_LABEL_DOMAIN, false);
		}
		println("=======");
		
		val actGener = new SingletonActionGenerator(false);
		val goldGener = new SingletonActionGenerator(true);
		val searchLossFunc = buildLossFunction(lossTyp);

		//////////////////////////////////////////////////////////////////////
		// train
		val model = new SingletonSvmModel();
		val spTrain = constructProblem(trnInsts)//.asScala.toSeq);//Ace05DataSet.ExampleListToSLProblem(trnInsts);///slproblems.get(0);

		// initialize the inference solver
		val abkeys = new ZobristKeys(1000, 1000);
		val fg = new SingletonFeatGener();
		
		val searcher = new GreedySearcher(FactorGraphType.SingletonGraph, fg, nrnd, actGener, initStateGener, searchLossFunc, abkeys);
		val gdScher = new GreedySearcher(FactorGraphType.SingletonGraph, fg, nrnd, goldGener, initStateGenerGold, searchLossFunc, abkeys);
		val corefInfr = new AceCorefInferencer(searcher, gdScher);
		model.infSolver = corefInfr;
		model.featureGenerator = fg;

		val para = new SLParameters();
		para.loadConfigFile(configPath);
		para.TOTAL_NUMBER_FEATURE = featIndexer.size();//fg.getFeatLen();
		val baseLearner = LearnerFactory.getLearner(model.infSolver, fg, para);


		//////// latent_learner
		val latentParam = new SLParameters();
		latentParam.loadConfigFile(configPath);
		latentParam.TOTAL_NUMBER_FEATURE = fg.getFeatLen();
		latentParam.MAX_NUM_ITER = 10;
		//val latentLearner = new MyLatentLearner(baseLearner, fg, latentParam, corefInfr);
    val latentLearner = new GeneralLatentSvmLearner(baseLearner, fg, latentParam, corefInfr);

		val initwv = new WeightVector(para.TOTAL_NUMBER_FEATURE);
		model.wv = initwv;
		System.err.println("weightLength1 = " + initwv.getLength() + " " + para.TOTAL_NUMBER_FEATURE);

		val trnTimer = new MyTimeCounter("singleton-train");
		trnTimer.start();
		
		if (CostFuncCacherAndLoader.cacheCostWeight) {
			val loadedWv = costCacher.loadCachedWeight(dsName, initType, randomNum, CostFuncCacherAndLoader.getFeatDim(true, true, true), lossTyp, para.C_FOR_STRUCTURE, -1);
			if (loadedWv != null) { // load failure...
				model.wv = loadedWv;
			} else {
				model.wv = latentLearner.train(spTrain, initwv);
				costCacher.saveCachedWeight(model.wv, dsName, initType, randomNum, CostFuncCacherAndLoader.getFeatDim(true, true, true), lossTyp, para.C_FOR_STRUCTURE, -1); // save
			}
		} else {
			//model.wv = latentLearner.train(spTrain, initwv);
			model.wv = latentLearner.train(spTrain, initwv, tstInsts, runSingletonEvaluation);
			WeightVector.printSparsity(model.wv);
		}
		
		model.config = new JHashMap[String, String]();
		System.out.println("Done training...");
		trnTimer.end();
		System.out.println("time (sec): " + trnTimer.getSeconds());
		
		// test
		//////////////////////////////////
		model.convertDoubleWeightArray()
  }
  
  def buildLossFunction(lossTyp : MulLbLossType): AbstractLossFunction = {
		if (lossTyp == MulLbLossType.HAMMING_LOSS) {
			return (new SearchLossHamming());
		} else if (lossTyp == MulLbLossType.EXMPF1_LOSS) {
			return (new SearchLossExmpF1());
		} else if (lossTyp == MulLbLossType.EXMPACC_LOSS) {
			return (new SearchLossExmpAcc());
		}
		return null;
	}
  
  def runSingletonEvaluation(testExs: Seq[IInstance], wv: WeightVector) {
    val weights = wv.getDoubleArray
    var tpos = 0
    var tneg = 0
    var fpos = 0
    var fneg = 0
    for (ex1 <- testExs) {
      val ex = ex1.asInstanceOf[SingletonInstance]
      val pms = ex.mentions
      for (i <- 0 until pms.length) {
        val pm = pms(i)
        val predLb = pm.predictLabel(weights)//predictLabel(pm, model)
      	//val lbl = pm.goldLabel
        /*
        if (predLb > 0 && lbl > 0) {
        	tpos += 1
        } else if (predLb == 0 && lbl > 0) {
        	fneg += 1
        } else if (predLb > 0 && lbl == 0) {
        	fpos += 1
        } else if (predLb == 0 && lbl == 0) {
        	tneg += 1
        }*/
      }

    }
    
    val zeroOne = tpos.toDouble / (testExs.size).toDouble
    println("0-1Accuracy: " + tpos + " / " + (testExs.size) + " = " + zeroOne);
    val precision = tpos.toDouble / (tpos + fpos).toDouble
    println("Precision: " + tpos + " / " + (tpos + fpos) + " = " + precision);
    val recall = tpos.toDouble / (tpos + fneg).toDouble
    println("Recall: " + tpos + " / " + (tpos + fneg) + " = " + recall);
    val recall2 = tpos.toDouble / (19764).toDouble
    println("Recall2: " + tpos + " / " + (19764) + " = " + recall2);
  }
  
  
  //////////////////////
  //// Learning
  //////////////////////
  
  def runSingletonDetectSVMLearning() {
    
  }
  
  def bool2int(b:Boolean) = if (b) 1 else 0
  
  def extractSingletonInstances(docs: Seq[DocumentGraph], addFeat: Boolean) = {
    val singletonInsts = new ArrayBuffer[SingletonInstance]();
    for (d <- docs) {
      val docInsts = extractSingletonInstancesOneDoc(d, addFeat);
      singletonInsts ++= docInsts;
    }
    singletonInsts.toSeq
  }
  
  def extractSingletonInstancesOneDoc(doc: DocumentGraph, addFeat: Boolean): ArrayBuffer[SingletonInstance] = {
    val ies = new ArrayBuffer[ArrayBuffer[MentionEdge]](doc.size());
    for (i <- 0 until doc.size()) {
      ies += (new ArrayBuffer[MentionEdge]());
    }
    val indexToEdges = ies.toArray
    
    for (i <- 0 until doc.size) {
      //val antes = doc.getAllAntecedentsCurrentPruning(i);
      val antes = doc.getAllAntecedentsNoPruning(i);
      for (j <- antes) {
        //if (i != j) {
          val feature = doc.cachedFeats(i)(j)
          val goldEdgeLabel = decideEdgeGoldLabel(doc, i, j)
          val edgeConsist = decideConsistency(doc, i, j, goldEdgeLabel)
          val edgeij = new MentionEdge(i, j, feature, goldEdgeLabel, edgeConsist)
          indexToEdges(i) += edgeij;
          indexToEdges(j) += edgeij;
        //}
      }
    }
    
    //// features
    val insts = new ArrayBuffer[SingletonMentionInstance]()
    for (i <- 0 until doc.size()) {
      val lb = pickMentInstLabel(indexToEdges(i))
      val mentInst = new SingletonMentionInstance(i, doc, indexToEdges(i))//, lb);
      insts += (mentInst);
    }
    
    val singleInst = new SingletonInstance(insts.toArray)
    val singleInstList = new ArrayBuffer[SingletonInstance]()
    singleInstList += singleInst
    
    singleInstList
  }
  
  def decideEdgeGoldLabel(doc: DocumentGraph, curIdx: Int, anteIdx: Int): Int = {
    
    val mention = doc.getMention(curIdx)
    val matched = hasMatchedGoldMention(doc, mention)
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
    
    label
  }
  
  def decideConsistency(doc: DocumentGraph, curIdx: Int, anteIdx: Int, goldLabel: Int): Boolean = {
    
    val consistent = if (goldLabel > 0) {
      true
    } else {
      if (anteIdx == curIdx) {
        true
      } else {
        false
      }
    }
    
    consistent
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
  
  def pickMentInstLabel(edges: ArrayBuffer[MentionEdge]) = {
    var maxLb = 0;
    maxLb = edges.map { me => me.goldLb }.max
    maxLb
  }
}