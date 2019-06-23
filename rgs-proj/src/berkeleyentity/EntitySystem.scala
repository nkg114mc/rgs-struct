package berkeleyentity

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

import java.util.ArrayList;
import java.nio.file.Files;
import java.nio.file.CopyOption._;
import java.io.File;
import scala.util.control.Breaks._

object EntitySystem {
  
  def preprocessDocsCacheResources(allDocGraphs: Seq[DocumentGraph]) {
    if (Driver.wordNetPath != "") {
      val wni = new WordNetInterfacer(Driver.wordNetPath);
      allDocGraphs.foreach(_.cacheWordNetInterfacer(wni));
    }
  }
  
//  def preprocessDocs(path: String,
//                     size: Int,
//                     mentionPropertyComputer: MentionPropertyComputer,
//                     nerPruner: NerPruner,
//                     corefPruner: CorefPruner,
//                     train: Boolean) = {
//    // Read in raw data
//    val rawDocs = ConllDocReader.loadRawConllDocsWithSuffix(path, size, "auto_conll");
//    val goldConllDocs = ConllDocReader.loadRawConllDocsWithSuffix(path, size, "gold_conll");
//    val goldWikification = new HashMap[String,HashMap[Int,ArrayBuffer[Chunk[String]]]];
//    val assembler = CorefDocAssembler(Driver.lang, Driver.useGoldMentions);
//    val corefDocs = rawDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
//    CorefDocAssembler.checkGoldMentionRecall(corefDocs);
//    val docGraphs = corefDocs.map(new DocumentGraph(_, train));
//    preprocessDocsCacheResources(docGraphs);
//    // Prune coref now that we have mentions
//    corefPruner.pruneAll(docGraphs);
//    
//    val jointDocsOrigOrder = JointDoc.assembleJointDocs(docGraphs, goldConllDocs, goldWikification);
//    // Store NER marginals
//    jointDocsOrigOrder.foreach(_.cacheNerPruner(Some(nerPruner)));
//    if (train) {
//      // Randomize
//      new scala.util.Random(0).shuffle(jointDocsOrigOrder)
//    } else {
//      jointDocsOrigOrder;
//    }
//  }
//  
//  
//  def preprocessDocs(path: String,
//                     size: Int,
//                     suffix: String,
//                     mentionPropertyComputer: MentionPropertyComputer,
//                     corefPruner: CorefPruner,
//                     nerPruner: NerPruner) = {
//    // Read raw documents
//    val rawDocs = ConllDocReader.loadRawConllDocsWithSuffix(path, size, suffix);
//    // Get mentions
//    val assembler = CorefDocAssembler(Language.ENGLISH, useGoldMentions = false);
//    val corefDocs = rawDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
//    val docGraphs = corefDocs.map(new DocumentGraph(_, false));
//    preprocessDocsCacheResources(docGraphs);
//    // Prune coreference
//    corefPruner.pruneAll(docGraphs);
//    // Build joint document and prune NER
//    val jointDocs = JointDoc.assembleJointDocs(docGraphs, new ArrayBuffer[ConllDoc](), new HashMap[String,HashMap[Int,ArrayBuffer[Chunk[String]]]]);
//    jointDocs.foreach(_.cacheNerPruner(Some(nerPruner)));
//    jointDocs;
//  }
  
  def preprocessDocsForTrain(path: String,
                             size: Int,
                             mentionPropertyComputer: MentionPropertyComputer,
                             nerPruner: NerPruner,
                             corefPruner: CorefPruner) = {
    preprocessDocs(path, "auto_conll", path, "gold_conll", size, mentionPropertyComputer, nerPruner, corefPruner, true);
  }
  
  def preprocessDocsForEval(path: String,
                            size: Int,
                            mentionPropertyComputer: MentionPropertyComputer,
                            nerPruner: NerPruner,
                            corefPruner: CorefPruner) = {
    preprocessDocs(path, "auto_conll", path, "gold_conll", size, mentionPropertyComputer, nerPruner, corefPruner, false);
  }
  
  def preprocessDocsForDecode(path: String,
                              size: Int,
                              suffix: String,
                              mentionPropertyComputer: MentionPropertyComputer,
                              nerPruner: NerPruner,
                              corefPruner: CorefPruner) = {
    preprocessDocs(path, suffix, "", "", size, mentionPropertyComputer, nerPruner, corefPruner, false);
  }
  
  def preprocessDocs(path: String,
                     suffix: String,
                     goldPath: String,
                     goldSuffix: String,
                     size: Int,
                     mentionPropertyComputer: MentionPropertyComputer,
                     nerPruner: NerPruner,
                     corefPruner: CorefPruner,
                     train: Boolean) = {
    // Read in raw data
    val (rawDocs, goldConllDocs) = if (goldPath != "") {
      (ConllDocReader.loadRawConllDocsWithSuffix(path, size, suffix),
       ConllDocReader.loadRawConllDocsWithSuffix(goldPath, size, goldSuffix));
    } else {
      (ConllDocReader.loadRawConllDocsWithSuffix(path, size, suffix),
       new ArrayBuffer[ConllDoc]());
    }
    val goldWikification = new HashMap[String,HashMap[Int,ArrayBuffer[Chunk[String]]]];
    val assembler = CorefDocAssembler(Driver.lang, Driver.useGoldMentions);
    val corefDocs = rawDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    if (train) {
      CorefDocAssembler.checkGoldMentionRecall(corefDocs);
    }
    val docGraphs = corefDocs.map(new DocumentGraph(_, train));
    preprocessDocsCacheResources(docGraphs);
    // Prune coref now that we have mentions
    corefPruner.pruneAll(docGraphs);
    
    val jointDocsOrigOrder = JointDoc.assembleJointDocs(docGraphs, goldConllDocs, goldWikification);
    // Store NER marginals
    jointDocsOrigOrder.foreach(_.cacheNerPruner(Some(nerPruner)));
    if (train) {
      // Randomize
      new scala.util.Random(0).shuffle(jointDocsOrigOrder)
    } else {
      jointDocsOrigOrder;
    }
  }
  
  def runOntoPredict(path: String, size: Int, modelPath: String) {
    val jointPredictor = GUtil.load(modelPath).asInstanceOf[JointPredictor];
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    val maybeWikipediaInterface: Option[WikipediaInterface] = if (Driver.wikipediaPath != "") Some(GUtil.load(Driver.wikipediaPath).asInstanceOf[WikipediaInterface]) else None;
    val jointDocs = preprocessDocsForDecode(path, size, Driver.docSuffix, mentionPropertyComputer, jointPredictor.nerPruner, jointPredictor.corefPruner);
    jointPredictor.decodeWriteOutput(jointDocs, maybeWikipediaInterface, Driver.doConllPostprocessing);
  }
  
  def runOntoPredictEvaluate(path: String, size: Int, modelPath: String) {
    val jointPredictor = GUtil.load(modelPath).asInstanceOf[JointPredictor];
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    val maybeWikipediaInterface: Option[WikipediaInterface] = if (Driver.wikipediaPath != "") Some(GUtil.load(Driver.wikipediaPath).asInstanceOf[WikipediaInterface]) else None;
    val jointDocs = preprocessDocsForEval(path, size, mentionPropertyComputer, jointPredictor.nerPruner, jointPredictor.corefPruner);
    jointPredictor.decodeWriteOutputEvaluate(jointDocs, maybeWikipediaInterface, Driver.doConllPostprocessing);
  }
  
  def runTrainEvaluate(trainPath: String, trainSize: Int, testPath: String, testSize: Int) = {
    // Resources needed for document assembly: number/gender computer, NER marginals, coref models and mapping of documents to folds
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    
    // N.B. DIFFERENT DUE TO NER BEING PRESENT
    // Load NER pruning masks and coref models
    val nerPruner = NerPruner.buildPruner(Driver.nerPruningStrategy);
    val corefPruner = CorefPruner.buildPruner(Driver.pruningStrategy)
    val jointDocs = preprocessDocsForTrain(trainPath, trainSize, mentionPropertyComputer, nerPruner, corefPruner);
    
    // N.B. Only difference here is the NER
    ///////////////////////
    // Build the featurizer, which involves building specific featurizers for each task
    val featureIndexer = new Indexer[String];
    val maybeBrownClusters = if (Driver.brownPath != "") Some(BrownClusterInterface.loadBrownClusters(Driver.brownPath, 0)) else None
    val nerFeaturizer = NerFeaturizer(Driver.nerFeatureSet.split("\\+").toSet, featureIndexer, NerSystemLabeled.StdLabelIndexer, jointDocs.flatMap(_.rawDoc.words), None, maybeBrownClusters);
    val jointFeaturizer = buildFeaturizerShared(jointDocs.map(_.docGraph.corefDoc), featureIndexer, nerFeaturizer, maybeBrownClusters);
    val maybeWikipediaInterface: Option[WikipediaInterface] = if (Driver.wikipediaPath != "") Some(GUtil.load(Driver.wikipediaPath).asInstanceOf[WikipediaInterface]) else None;
    
    
    ///////////////////////
    val fgfOnto = new FactorGraphFactoryOnto(jointFeaturizer, maybeWikipediaInterface);
    val computer = new JointComputerShared(fgfOnto);
    jointDocs.foreach(jointDoc => {
      fgfOnto.getDocFactorGraph(jointDoc, true, true, true, PairwiseLossFunctions(Driver.lossFcn), JointLossFcns.nerLossFcn, JointLossFcns.wikiLossFcn);
      fgfOnto.getDocFactorGraph(jointDoc, false, true, true, PairwiseLossFunctions(Driver.lossFcn), JointLossFcns.nerLossFcn, JointLossFcns.wikiLossFcn);
    });
    PairwiseIndexingFeaturizer.printFeatureTemplateCounts(featureIndexer)
    Logger.logss(featureIndexer.size + " total features");
    
    val finalWeights = new GeneralTrainer[JointDoc].trainAdagrad(jointDocs, computer, featureIndexer.size, Driver.eta.toFloat, Driver.reg.toFloat, Driver.batchSize, Driver.numItrs);
    val model = new JointPredictor(jointFeaturizer, finalWeights, corefPruner, nerPruner).pack;
    if (Driver.modelPath != "") GUtil.save(model, Driver.modelPath);
    
    ///////////////////////
    // Evaluation of each part of the model
    // Build dev docs
    val jointDevDocs = preprocessDocsForEval(testPath, testSize, mentionPropertyComputer, nerPruner, corefPruner);
    model.decodeWriteOutputEvaluate(jointDevDocs, maybeWikipediaInterface, Driver.doConllPostprocessing);
  }
  
  
  //////////////////////////////
  //////////// ACE /////////////
  //////////////////////////////
  
  // N.B. Doubles with JointPredictor.preprocessACEDocs
  def preprocessACEDocsForTrainEval(path: String,
                                    size: Int,
                                    mentionPropertyComputer: MentionPropertyComputer,
                                    corefPruner: CorefPruner,
                                    wikiPath: String,
                                    train: Boolean) = {
    // Read in raw data
    val rawDocs = ConllDocReader.loadRawConllDocsWithSuffix(path, size, "", Language.ENGLISH);
    val goldWikification: CorpusWikiAnnots = if (wikiPath != "") {
      val corpusAnnots = new CorpusWikiAnnots;
      for (entry <- WikiAnnotReaderWriter.readAllStandoffAnnots(wikiPath)) {
        val fileName = entry._1._1;
        val docAnnots = new DocWikiAnnots;
        for (i <- 0 until entry._2.size) {
//          if (!entry._2(i).isEmpty) {
          docAnnots += i -> (new ArrayBuffer[Chunk[Seq[String]]]() ++ entry._2(i))
//          }
        }
        corpusAnnots += fileName -> docAnnots
      }
      corpusAnnots;
    } else {
      Logger.logss("Wikification not loaded");
      new CorpusWikiAnnots;
    }
    val corefDocs = if (Driver.useGoldMentions) {
      val assembler = CorefDocAssembler(Driver.lang, true);
      rawDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    } else {
      val assembler = new CorefDocAssemblerACE(Driver.allAcePath);
      rawDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    }
    CorefDocAssembler.checkGoldMentionRecall(corefDocs);
    val docGraphs = corefDocs.map(new DocumentGraph(_, train));
    preprocessDocsCacheResources(docGraphs);
    // Prune coref now that we have mentions
    corefPruner.pruneAll(docGraphs);
    
    
    val jointDocsOrigOrder = JointDocACE.assembleJointDocs(docGraphs, goldWikification);
    // TODO: Apply NER pruning
//    JointDoc.applyNerPruning(jointDocsOrigOrder, nerMarginals);
    if (train) {
      // Randomize
      new scala.util.Random(0).shuffle(jointDocsOrigOrder)
    } else {
      jointDocsOrigOrder;
    }
  }
  
  def preprocessACEDocsForDecode(path: String, size: Int, suffix: String, mentionPropertyComputer: MentionPropertyComputer, corefPruner: CorefPruner) = {
    val rawDocs = ConllDocReader.loadRawConllDocsWithSuffix(path, size, suffix, Language.ENGLISH);
    val assembler = CorefDocAssembler(Driver.lang, Driver.useGoldMentions);
    val corefDocs = rawDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    val docGraphs = corefDocs.map(new DocumentGraph(_, false));
    CorefSystem.preprocessDocsCacheResources(docGraphs);
    // Prune coref now that we have mentions
    corefPruner.pruneAll(docGraphs);
    // No NER pruning
    JointDocACE.assembleJointDocs(docGraphs, new CorpusWikiAnnots);
  }
  
  def runACEPredict(path: String, size: Int, modelPath: String) {
    
    println("MMMMMC start running!");
    
    val jointPredictor = GUtil.load(modelPath).asInstanceOf[JointPredictorACE];
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    val maybeWikipediaInterface: Option[WikipediaInterface] = if (Driver.wikipediaPath != "") Some(GUtil.load(Driver.wikipediaPath).asInstanceOf[WikipediaInterface]) else None;
    val jointDocs = preprocessACEDocsForDecode(path, size, Driver.docSuffix, mentionPropertyComputer, jointPredictor.corefPruner);
    jointPredictor.decodeWriteOutput(jointDocs, maybeWikipediaInterface, Driver.doConllPostprocessing);
  }
  
  def runACEPredictEvaluate(path: String, size: Int, modelPath: String) {
    
    println("MC start running!");
    
    val jointPredictor = GUtil.load(modelPath).asInstanceOf[JointPredictorACE];
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    val maybeWikipediaInterface: Option[WikipediaInterface] = if (Driver.wikipediaPath != "") Some(GUtil.load(Driver.wikipediaPath).asInstanceOf[WikipediaInterface]) else None;
    val jointDocs = preprocessACEDocsForTrainEval(path, size, mentionPropertyComputer, jointPredictor.corefPruner, Driver.wikiGoldPath, false);
    
    //myACEDocShow(jointDocs);
    val ilpPredictor = new JointPredictorACE(jointPredictor.jointFeaturizer,jointPredictor.weights,jointPredictor.corefPruner: CorefPruner);
    
    ///jointPredictor.decodeWriteOutputEvaluate(jointDocs, maybeWikipediaInterface, Driver.doConllPostprocessing);
    ilpPredictor.decodeWriteOutputEvaluate(jointDocs, maybeWikipediaInterface, Driver.doConllPostprocessing);
  }
  
/*
  def runACEInferenceILP(path: String, size: Int, modelPath: String) {
    println("ILP inference start running!");
    val jointPredictor = GUtil.load(modelPath).asInstanceOf[JointPredictorACE];
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    val maybeWikipediaInterface: Option[WikipediaInterface] = if (Driver.wikipediaPath != "") Some(GUtil.load(Driver.wikipediaPath).asInstanceOf[WikipediaInterface]) else None;
    val jointDocs = preprocessACEDocsForTrainEval(path, size, mentionPropertyComputer, jointPredictor.corefPruner, Driver.wikiGoldPath, false);
    
    val myjModel = GUtil.load("./w_joint_model.ser.gz").asInstanceOf[MyModel];
    val myJointFeatIndexer = myjModel.featIndexer;// new Indexer[String](); // for ILP Learning only
    val myFeater = constructIlpFeaturizer[MCNerFeaturizer](jointDocs.map(_.docGraph.corefDoc), myJointFeatIndexer, jointPredictor.jointFeaturizer.nerFeaturizer, jointPredictor.jointFeaturizer.maybeBrownClusters);

    val jointILPinferencer = new JointACEInferenceILP(myFeater, jointPredictor, myjModel);
    jointILPinferencer.inferenceEvaluate(jointDocs, maybeWikipediaInterface, Driver.doConllPostprocessing);
    //jointPredictor.decodeWriteOutputEvaluate(jointDocs, maybeWikipediaInterface, Driver.doConllPostprocessing);
  }
*/

  def myACEDocShow(docs: Seq[JointDocACE]) {
    for (d <- docs) {
       println(d.rawDoc);
    }
  }
  
  def runTrainEvaluateACE(trainPath: String, trainSize: Int, testPath: String, testSize: Int) = {
      runTrainEvaluateACEWikiPath(trainPath, trainSize, testPath, testSize, Driver.wikiGoldPath);
  }
  
  def runTrainEvaluateACEWikiPath(trainPath: String, trainSize: Int, testPath: String, testSize: Int, wikiPath: String) = {
    // Resources needed for document assembly: number/gender computer, NER marginals, coref models and mapping of documents to folds
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    
    // Load coref models
    val corefPruner = CorefPruner.buildPruner(Driver.pruningStrategy)
    val jointDocs = preprocessACEDocsForTrainEval(trainPath, trainSize, mentionPropertyComputer, corefPruner, wikiPath, true);
    // TODO: Are NER models necessary?
    
    ///////////////////////
    // Build the featurizer, which involves building specific featurizers for each task
    val featureIndexer = new Indexer[String]();
    val maybeBrownClusters = if (Driver.brownPath != "") Some(BrownClusterInterface.loadBrownClusters(Driver.brownPath, 0)) else None
    val nerFeaturizer = MCNerFeaturizer(Driver.nerFeatureSet.split("\\+").toSet, featureIndexer, MCNerFeaturizer.StdLabelIndexer, jointDocs.flatMap(_.rawDoc.words), None, maybeBrownClusters)
    val jointFeaturizer = buildFeaturizerShared(jointDocs.map(_.docGraph.corefDoc), featureIndexer, nerFeaturizer, maybeBrownClusters);
    val maybeWikipediaInterface: Option[WikipediaInterface] = if (Driver.wikipediaPath != "") Some(GUtil.load(Driver.wikipediaPath).asInstanceOf[WikipediaInterface]) else None;
    
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
    val model = new JointPredictorACE(jointFeaturizer, finalWeights, corefPruner).pack;
    if (Driver.modelPath != "") GUtil.save(model, Driver.modelPath);
    
    ///////////////////////
    // Evaluation of each part of the model
    // Build dev docs
    val jointDevDocs = preprocessACEDocsForTrainEval(testPath, testSize, mentionPropertyComputer, corefPruner, wikiPath, false);
    val wikiLabelsInTrain: Set[String] = jointDocs.flatMap(_.goldWikiChunks.flatMap(_.flatMap(_.label)).toSet).toSet;
    model.decodeWriteOutputEvaluate(jointDevDocs, maybeWikipediaInterface, Driver.doConllPostprocessing, wikiLabelsInTrain)
  }
  
  def buildFeaturizerShared[T](trainDocs: Seq[CorefDoc], featureIndexer: Indexer[String], nerFeaturizer: T, maybeBrownClusters: Option[Map[String,String]]) = {
    featureIndexer.getIndex(PairwiseIndexingFeaturizerJoint.UnkFeatName);
    val queryCounts: Option[QueryCountsBundle] = None;
    val lexicalCounts = LexicalCountsBundle.countLexicalItems(trainDocs, Driver.lexicalFeatCutoff);
    val semClasser: Option[SemClasser] = Some(new BasicWordNetSemClasser);
    val corefFeatureSetSpec = FeatureSetSpecification(Driver.pairwiseFeats, Driver.conjScheme, Driver.conjFeats, Driver.conjMentionTypes, Driver.conjTemplates);
    val corefFeaturizer = new PairwiseIndexingFeaturizerJoint(featureIndexer, corefFeatureSetSpec, lexicalCounts, queryCounts, semClasser);
    new JointFeaturizerShared[T](corefFeaturizer, nerFeaturizer, maybeBrownClusters, Driver.corefNerFeatures, Driver.corefWikiFeatures, Driver.wikiNerFeatures, featureIndexer)
  }
  
  
  
  ///////////////////////////////////////// By MC ///////////////
/*
  def runSelfTrainACE(trainPath: String, trainSize: Int, testPath: String, testSize: Int) {

    val confidenceFilter = new LabelFilter();
    println("SelfSelfSelfSelfSelfSelfSelfSelf!!!!!!");
    
	  // Resources needed for document assembly: number/gender computer, NER marginals, coref models and mapping of documents to folds
	  val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
	  val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));

	  // Load coref models
	  val corefPruner = CorefPruner.buildPruner(Driver.pruningStrategy)
		val jointDocs = preprocessACEDocsForTrainEval(trainPath, trainSize, mentionPropertyComputer, corefPruner, Driver.wikiGoldPath, true);
	  // TODO: Are NER models necessary?

	  //
	  val startDocsCnt = Driver.selfTrainStartDocNumber;
    val restDocsCnt = jointDocs.size - startDocsCnt;
	  //val addingDocsCntPerIter = 10;
	  val startDocsList = new ArrayBuffer[JointDocACE];
    val restDocsList = new ArrayBuffer[JointDocACE];
	  for (j <- 0 until jointDocs.size) {
      if (j < startDocsCnt) {
        startDocsList += jointDocs(j);
      } else if ((j >= startDocsCnt) && (j < (startDocsCnt + restDocsCnt))) {
        restDocsList += (jointDocs(j));
      }
	  }

   // prepare a starting model
		  val iterTrainDocs = startDocsList;
		  val iterTestDocs = restDocsList;

		  ///////////////////////
		  // Build the featurizer, which involves building specific featurizers for each task
		  val featureIndexer = new Indexer[String]();
		  val maybeBrownClusters = if (Driver.brownPath != "") Some(BrownClusterInterface.loadBrownClusters(Driver.brownPath, 0)) else None
			val nerFeaturizer = MCNerFeaturizer(Driver.nerFeatureSet.split("\\+").toSet, featureIndexer, MCNerFeaturizer.StdLabelIndexer, iterTrainDocs.flatMap(_.rawDoc.words), None, maybeBrownClusters)
			val jointFeaturizer = buildFeaturizerShared(iterTrainDocs.map(_.docGraph.corefDoc), featureIndexer, nerFeaturizer, maybeBrownClusters);
		  val maybeWikipediaInterface: Option[WikipediaInterface] = if (Driver.wikipediaPath != "") Some(GUtil.load(Driver.wikipediaPath).asInstanceOf[WikipediaInterface]) else None;

		  ///////////////////////
		  // Cache features
		  val fgfAce = new FactorGraphFactoryACE(jointFeaturizer, maybeWikipediaInterface);
		  val computer = new JointComputerShared(fgfAce);
		  iterTrainDocs.foreach(jointDoc => {
			  fgfAce.getDocFactorGraph(jointDoc, true, true, true, PairwiseLossFunctions(Driver.lossFcn), JointLossFcns.nerLossFcn, JointLossFcns.wikiLossFcn);
			  fgfAce.getDocFactorGraph(jointDoc, false, true, true, PairwiseLossFunctions(Driver.lossFcn), JointLossFcns.nerLossFcn, JointLossFcns.wikiLossFcn);
		  });
		  PairwiseIndexingFeaturizer.printFeatureTemplateCounts(featureIndexer)
		  Logger.logss(featureIndexer.size + " total features");

		  val finalWeights = new GeneralTrainer[JointDocACE].trainAdagrad(iterTrainDocs, computer, featureIndexer.size, Driver.eta.toFloat, Driver.reg.toFloat, Driver.batchSize, Driver.numItrs);
		  val model = new JointPredictorACE(jointFeaturizer, finalWeights, corefPruner).pack;
		  // save model
      val startModelPath : String = "./selftrain/self_model_iter1.ser.gz";
      GUtil.save(model, startModelPath);

    // === Each iteration ======================================== 
    val iterCnt : Int = 5;
    for (i <- 0 until iterCnt) { // big cycle
      
		  ///////////////////////
		  // Evaluation of each part of the model
		  // Build dev docs
		  //val jointDevDocs = preprocessACEDocsForTrainEval(testPath, testSize, mentionPropertyComputer, corefPruner, Driver.wikiGoldPath, false);
		  val myPredPath = "tempcopy_train";
      val myPredWikiPath = "tempcopy_train_wiki";
      
      println("start_size = " + startDocsList.size);
      println("rest_size = " + restDocsList.size);
      
      val desDirPath = new File(myPredPath);
      for (extDocPath <- desDirPath.list()) {
        val tmpPath = (new File(myPredPath, extDocPath)).toPath();
        Files.delete(tmpPath); // delete all files
      }
      for (cpDoc <- restDocsList) {
        val fName = cpDoc.rawDoc.fileName;
        println("file name = " + fName);
        val originDir = trainPath;
        val desDir = myPredPath;
        Files.copy((new File(originDir, fName)).toPath(), (new File(desDir, fName)).toPath());
      }
 
      val outDir = Driver.myoutputPath;//"/scratch/EntityLinking2015/berkeley-entity-master/data/ace05/myoutput";
      val outWikiDir = Driver.myoutputWikiPath;//"/scratch/EntityLinking2015/berkeley-entity-master/data/ace05/myoutput_wiki";
      deleteAllInDir(outDir); // delete all
      deleteAllInDir(outWikiDir); // delete all
      deleteAllInDir(myPredWikiPath); // delete all
      
      val jointDevDocs = preprocessACEDocsForTrainEval(myPredPath, -1, mentionPropertyComputer, corefPruner, Driver.wikiGoldPath, false);
      val wikiLabelsInTrain: Set[String] = jointDocs.flatMap(_.goldWikiChunks.flatMap(_.flatMap(_.label)).toSet).toSet;
		  val iterModel = if (i == 0) {
        model;
      } else {
        val mpath = "./selftrain/" + "final-" + (i - 1) + ".ser.gz";
        GUtil.load(mpath).asInstanceOf[JointPredictorACE];
      }
      
      /// filter out some bad labels
      iterModel.decodeWriteOutputEvaluate(jointDevDocs, maybeWikipediaInterface, Driver.doConllPostprocessing, wikiLabelsInTrain);
      //confidenceFilter;


      
      // copy gold start doc and wiki files
      val allGoldWikiFiles = (new File(Driver.wikiGoldPath)).list();
      for (cpTrainDoc <- iterTrainDocs) {
        val fName = cpTrainDoc.rawDoc.fileName;
        val originDir = trainPath;
        val desDir = outDir;
        Files.copy((new File(originDir, fName)).toPath(), (new File(desDir, fName)).toPath());
        ///
        val allMatchWikiFiles = allGoldWikiFiles.filter(x => (x.contains(fName)));
        val matchWikiFile : String = allMatchWikiFiles(0);
        Files.copy((new File(Driver.wikiGoldPath, matchWikiFile)).toPath(), (new File(outWikiDir, matchWikiFile)).toPath());
      }
      
      ////////////////////////// start final training
      
      //////////////////////////
      // train new model
      Driver.modelPath = "./selftrain/" + "final-" + i + ".ser.gz";
      runTrainEvaluateACEWikiPath(outDir, -1, testPath, -1, outWikiDir);
	  }
    // === End of iteration ======================================== 

  }
*/
  def deleteAllInDir(dirPath : String) {
	  val desDirPath = new File(dirPath);
	  if (desDirPath.exists() && desDirPath.isDirectory()) {
		  for (extDocPath <- desDirPath.list()) {
			  val tmpPath = (new File(dirPath, extDocPath)).toPath();
			  Files.delete(tmpPath); // delete all files
		  }
	  }
  }
  
  
  /*
    val jointPredictor = GUtil.load(modelPath).asInstanceOf[JointPredictorACE];
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    val maybeWikipediaInterface: Option[WikipediaInterface] = if (Driver.wikipediaPath != "") Some(GUtil.load(Driver.wikipediaPath).asInstanceOf[WikipediaInterface]) else None;
    val jointDocs = preprocessACEDocsForTrainEval(path, size, mentionPropertyComputer, jointPredictor.corefPruner, Driver.wikiGoldPath, false);
    
    val myJointFeatIndexer = new Indexer[String](); // for ILP Learning only
    val myFeater = constructIlpFeaturizer[MCNerFeaturizer](jointDocs.map(_.docGraph.corefDoc), myJointFeatIndexer, jointPredictor.jointFeaturizer.nerFeaturizer, jointPredictor.jointFeaturizer.maybeBrownClusters);

    val jointILPinferencer = new JointACEInferenceILP(myFeater, jointPredictor);
    jointILPinferencer.inferenceEvaluate(jointDocs, maybeWikipediaInterface, Driver.doConllPostprocessing);
   */
  
  //// ILP joint variable weight training /////////////
/*
  def runTrainAceIlp(trainPath: String, trainSize: Int, testPath: String, testSize: Int) {

    println("ILP Joint Train!!!!!!");
    
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    
    // Load coref models
    val independentPredictor = GUtil.load(Driver.modelPath).asInstanceOf[JointPredictorACE];
    val corefPruner = independentPredictor.corefPruner;//CorefPruner.buildPruner(Driver.pruningStrategy)
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
    
    //val structPerceptronLearner = new StructurePerceptronLearner();
    //val myModel = structPerceptronLearner.runStructPerceptronTrain(jointDocs, prbConstructAce);
    //MyModel.saveMyModel(myModel, Driver.myjModelPath);
    
    
    
    
    
    
    //val finalWeights = new GeneralTrainer[JointDocACE].trainAdagrad(jointDocs, computer, featureIndexer.size, Driver.eta.toFloat, Driver.reg.toFloat, Driver.batchSize, Driver.numItrs);
    
    
    //val model = new JointPredictorACE(jointFeaturizer, finalWeights, corefPruner).pack;
    //if (Driver.modelPath != "") GUtil.save(model, Driver.modelPath);
    
    ///////////////////////
    // Evaluation of each part of the model
    // Build dev docs

  }
  
  def constructIlpFeaturizer[T](trainDocs: Seq[CorefDoc], myIndexer: Indexer[String], nerFeaturizer: T, maybeBrownClusters: Option[Map[String,String]]) = {

    myIndexer.getIndex(PairwiseIndexingFeaturizerJoint.UnkFeatName);
    val queryCounts: Option[QueryCountsBundle] = None;
    val lexicalCounts = LexicalCountsBundle.countLexicalItems(trainDocs, Driver.lexicalFeatCutoff);
    val semClasser: Option[SemClasser] = Some(new BasicWordNetSemClasser);
    val corefFeatureSetSpec = FeatureSetSpecification(Driver.pairwiseFeats, Driver.conjScheme, Driver.conjFeats, Driver.conjMentionTypes, Driver.conjTemplates);
    val corefFeaturizer = new PairwiseIndexingFeaturizerJoint(myIndexer, corefFeatureSetSpec, lexicalCounts, queryCounts, semClasser);
    
    val myfeater =  new FeaturizerILP[T](corefFeaturizer, nerFeaturizer, maybeBrownClusters, Driver.corefNerFeatures, Driver.corefWikiFeatures, Driver.wikiNerFeatures, myIndexer);
        //(val corefFeaturizer: PairwiseIndexingFeaturizer,
    //                   val nerFeaturizer: T,
    //                   val maybeBrownClusters: Option[Map[String,String]],
    //                   val corefNerFeatures: String,
    //                   val corefWikiFeatures: String,
    //                   val wikiNerFeatures: String,
    //                   val indexer: Indexer[String])
    myfeater;
  }
*/
}