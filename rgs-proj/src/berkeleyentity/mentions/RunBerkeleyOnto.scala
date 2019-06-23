package berkeleyentity.mentions

import berkeleyentity.coref.DocumentInferencerBasic
import berkeleyentity.sem.QueryCountsBundle
import berkeleyentity.coref.NumberGenderComputer
import berkeleyentity.sem.BasicWordNetSemClasser
import berkeleyentity.coref.FeatureSetSpecification
import berkeleyentity.GUtil
import berkeleyentity.sem.SemClasser
import berkeleyentity.coref.LexicalCountsBundle
import berkeleyentity.coref.CorefFeaturizerTrainer
import berkeleyentity.coref.PairwiseLossFunctions
import berkeleyentity.coref.PairwiseIndexingFeaturizerJoint
import berkeleyentity.coref.PairwiseIndexingFeaturizer
import edu.berkeley.nlp.futile.fig.basic.Indexer
import berkeleyentity.coref.PairwiseScorer
import berkeleyentity.Driver
import berkeleyentity.coref.CorefPruner
import berkeleyentity.EntitySystem
import berkeleyentity.coref.CorefSystem
import berkeleyentity.coref.DocumentGraph
import berkeleyentity.coref.CorefEvaluator
import edu.berkeley.nlp.futile.util.Logger


object RunBerkeleyOnto {
  
  val trainSuffix = "v4_auto_conll"
  val testSuffix = "v9_auto_conll"
  
  def main(args: Array[String]) {
    runBerkOntoCoref()
  }
  
  def runBerkOntoCoref() {
    val trainPath = "/home/mc/workplace/rand_search/coref/berkfiles/data/ontonotes5/train";
    val suffix1 = "v4_auto_conll"
    val testPath = "/home/mc/workplace/rand_search/coref/berkfiles/data/ontonotes5/test";
    val suffix2 = "v9_auto_conll"
    
    //val modelPath = "model/berkOnto-lex100.model";
    //runTrainEvaluate(trainPath, -1, testPath, -1, modelPath)
    
    val modelPath = "model/berkOnto.model";
    runEvaluateGivenModel(testPath, -1, testSuffix, modelPath)
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
  
  def runTrainEvaluate(trainPath: String, trainSize: Int, devPath: String, devSize: Int, modelPath: String) {
	  val scorer = runTrain(trainPath, trainSize, trainSuffix);
	  if (!modelPath.isEmpty) {
		  GUtil.save(scorer, modelPath);
	  }
	  if (!devPath.isEmpty) {
		  runEvaluate(devPath, devSize, testSuffix, scorer);
	  }
  }
  
  def runTrain(trainPath: String, trainSize: Int, modelPath: String, sufix: String) {
    val scorer = runTrain(trainPath, trainSize, sufix);
    GUtil.save(scorer, modelPath);
  }
  
  def runTrain(trainPath: String, trainSize: Int, sufix: String): PairwiseScorer = {
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

    val basicInferencer = new DocumentInferencerBasic()
    val lossFcnObjFirstPass = PairwiseLossFunctions(Driver.lossFcn);
    val firstPassWeights = featurizerTrainer.train(trainDocGraphs,
                                                   basicFeaturizer,
                                                   Driver.eta.toFloat,
                                                   Driver.reg.toFloat,
                                                   Driver.batchSize,
                                                   lossFcnObjFirstPass,
                                                   Driver.numItrs,
                                                   basicInferencer);
    new PairwiseScorer(basicFeaturizer, firstPassWeights).pack;
  }
}