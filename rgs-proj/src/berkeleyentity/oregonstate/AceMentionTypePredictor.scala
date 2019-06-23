package berkeleyentity.oregonstate

import java.io.File
import berkeleyentity.coref.MentionType
import edu.berkeley.nlp.futile.fig.basic.Indexer
import berkeleyentity.ner.CorpusCounts
import berkeleyentity.DepConstTree
import berkeleyentity.coref.Mention
import berkeleyentity.ner.NerExample
import scala.collection.mutable.ArrayBuffer
import berkeleyentity.ConllDoc
import berkeleyentity.coref.PronounDictionary
import berkeleyentity.lang.EnglishCorefLanguagePack
import berkeleyentity.sem.BrownClusterInterface
import berkeleyentity.coref.MentionPropertyComputer
import berkeleyentity.coref.NumberGenderComputer
import berkeleyentity.GUtil
import berkeleyentity.coref.CorefDoc
import berkeleyentity.lang.Language
import berkeleyentity.ConllDocReader
import berkeleyentity.coref.CorefDocAssembler
import edu.berkeley.nlp.futile.util.Logger
import berkeleyentity.Driver
import berkeleyentity.xgb.XgbMatrixBuilder
import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.java.{ DMatrix => JDMatrix }
import ml.dmlc.xgboost4j.scala.Booster
import ml.dmlc.xgboost4j.scala.XGBoost
import berkeleyentity.ner.NerFeaturizer
import scala.collection.mutable.HashMap

class MentExample(//val ment: Mention,
                  val words: Seq[String],
                  val poss: Seq[String],
                  val tree: DepConstTree,
                  val startIdx: Int,
                  val headIdx: Int,
                  val endIdx: Int,
                  val goldTyp: MentionType)  {
  
  def wordAt(i: Int) = NerExample.wordAt(words, i); 
  def posAt(i: Int) = NerExample.posAt(poss, i);
  
  def getGoldTypeIndex() : Int = {
    AceMentionTypePredictor.MentTypIndexer.getIndex(goldTyp);
  }
  
  var features: Array[Array[Int]] = null;
  def getArgMax(feats: Array[Array[Int]], weights: Array[Double]): Int = {
    var bestLbl = -1;
    var bestScore = -Double.MaxValue;
    for (l <- 0 until AceMentionTypePredictor.MentTypIndexer.size) {
       var score = NerTesting.computeScore(weights, feats(l));
       if (score > bestScore) {
          bestScore = score;
          bestLbl = l;
       }
    }
    bestLbl;
  }
}

object AceMentionTypePredictor {
  
  //val MentTypeSet = IndexedSeq(MentionType.PROPER.toString, MentionType.NOMINAL.toString, MentionType.PRONOMINAL.toString);
  val MentTypeSet = IndexedSeq(MentionType.PROPER, MentionType.NOMINAL, MentionType.PRONOMINAL);
  val MTypSetReduced = IndexedSeq("B", "I", "O");
  val MTagIndexer = new Indexer[String]();
  MTagIndexer.add("O");
  for (tag <- MentTypeSet) {
    MTagIndexer.add("B-" + tag);
    MTagIndexer.add("I-" + tag);
  }
  val MentTypIndexer = new Indexer[MentionType]();
  for (typ <- MentTypeSet) {
    MentTypIndexer.add(typ);
  }
  
  def main(args: Array[String]) {
    TrainMentionTypeACE();
  }
  
  
  
  def mentionToTypeEx(ment: Mention, rawDoc: ConllDoc): MentExample = {
    val pm = ment;
    if (pm.cachedMentionTypeGold == null) {
      pm.cachedMentionTypeGold = Mention.getGoldMEntType(pm.nerString);
    }
    val ex = new MentExample(rawDoc.words(pm.sentIdx), rawDoc.pos(pm.sentIdx), rawDoc.trees(pm.sentIdx), pm.startIdx, pm.headIdx, pm.endIdx, pm.cachedMentionTypeGold);
    ex;
  }
  
  def mentionInfoToTypeEx(words: Seq[String],
                  poss: Seq[String],
                  tree: DepConstTree,
                  startIdx: Int,
                  headIdx: Int,
                  endIdx: Int,
                  goldTyp: MentionType): MentExample = {
    val ex = new MentExample(words, poss, tree, startIdx, headIdx, endIdx, goldTyp);
    ex;
  }
  
  def extractExamples(corefDocs: Seq[CorefDoc]) = {
    val exs = new ArrayBuffer[MentExample];
    for (corefDoc <- corefDocs) {
      val rawDoc = corefDoc.rawDoc;
      val es = corefDoc.predMentions.map{ pm => mentionToTypeEx(pm, rawDoc) }
      exs ++= es;
    }
    println(exs.size + " mentions examples");
    exs.toSeq;
  }

 
  def TrainMentionTypeACE() = {
    
    val trainDataPath = "data/ace05/train";
    val devDataPath = "data/ace05/dev";
    val testDataPath = "data/ace05/test";
    
    Driver.numberGenderDataPath = "data/gender.data";
    Driver.brownPath = "data/bllip-clusters";

    val numItrs = 20
    
    // Read in CoNLL documents
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val goldTypePredictor = new GoldMentionTypePredictor();
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer), Some(goldTypePredictor));
    val assembler = CorefDocAssembler(Language.ENGLISH, true); //use gold mentions
    val trainDocs = ConllDocReader.loadRawConllDocsWithSuffix(trainDataPath, -1, "", Language.ENGLISH);
    val trainCorefDocs = trainDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    
    val devDocs = ConllDocReader.loadRawConllDocsWithSuffix(devDataPath, -1, "", Language.ENGLISH);
    val devCorefDocs = devDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    
    val testDocs = ConllDocReader.loadRawConllDocsWithSuffix(testDataPath, -1, "", Language.ENGLISH);
    val testCorefDocs = testDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    
    
    
    // Extract features
    val featIndexer = new Indexer[String]();
    val maybeBrownClusters = Some(BrownClusterInterface.loadBrownClusters(Driver.brownPath, 0));
    
    val unigramThreshold: Int = 1;
    val bigramThreshold: Int = 10;
    val prefSuffThreshold: Int = 1;
    val corpusCounts = CorpusCounts.countUnigramsBigrams(trainDocs.flatMap(_.words), unigramThreshold, bigramThreshold, prefSuffThreshold);
    val tfeaturizer = new AceMentionTypeFeaturizer(featIndexer, MentTypIndexer, corpusCounts , maybeBrownClusters);
    

    val trainExs = extractExamples(trainCorefDocs);
    val devExs = extractExamples(devCorefDocs);
    val testExs = extractExamples(testCorefDocs);
    
        
    for (ex <- trainExs) {
      ex.features = tfeaturizer.featurizeExample(ex, true);
    }
    for (ex <- devExs) {
    	ex.features = tfeaturizer.featurizeExample(ex, false);
    }
    for (ex <- testExs) {
    	ex.features = tfeaturizer.featurizeExample(ex, false);
    }
    

    // learn!
    val wght = structurePerceptrion(trainExs ++ devExs ++ testExs, featIndexer, testExs, 30);
    //val wght = structurePerceptrion(trainExs, featIndexer, testExs, 100);
    //val wght = multiClassSVM(allTrains, featIndexer, testEmpExs);
    //val wght = trainXgb(allTrains, featIndexer, testEmpExs);

    Logger.logss(featIndexer.size + " features");
    
    // Train
    //val gt = new GeneralTrainer[JointQueryDenotationExample]();
    //val weights = gt.trainAdagrad(trainExs, computer, featIndexer.size, 1.0F, lambda, batchSize, numItrs);
    //val chooser = new JointQueryDenotationChooser(featIndexer, weights)

    //testAceNerSystem(allTrains, wght, None);
    //testAceNerSystem(devEmpExs, wght, None);
    testTypeExamples(testExs, wght);

    //val booster = runQuickLearning(trainExs, testExs);
    //predictTypeAll(testExs, booster);
    
    val linearTpPredr = new AceMentionTypePredictor(tfeaturizer, wght);
    testPredictor(testExs, linearTpPredr);
    linearTpPredr;
  }
  
  def structurePerceptrion(trainExs: Seq[MentExample], 
                           featIndexer: Indexer[String],
                           testExs: Seq[MentExample],
                           Iteration: Int) : Array[Double] = {
    
    var weight = Array.fill[Double](featIndexer.size)(0);
    var weightSum = Array.fill[Double](featIndexer.size)(0);
    val learnRate = 0.1;
    val lambda = 1e-8;
    
    var updateCnt = 0;
    
    for (iter <- 0 until Iteration) {
      println("Iter " + iter);
      for (example <- trainExs) {
        val goldLabelIdx = example.getGoldTypeIndex();
        val bestLbl = example.getArgMax(example.features, weight);

        // update?
        if (bestLbl != goldLabelIdx) {
          updateCnt += 1;
          if (updateCnt % 1000 == 0) println("Update " + updateCnt);
          NerTesting.updateWeight(weight, 
                                  example.features(goldLabelIdx),
                                  example.features(bestLbl),
                                  learnRate,
                                  lambda);
          NerTesting.sumWeight(weightSum, weight);
        }
      }

      val tmpAvg = new Array[Double](weightSum.size)
      Array.copy(weightSum, 0, tmpAvg, 0, weightSum.size);
      NerTesting.divdeNumber(tmpAvg, updateCnt.toDouble);
      
      testTypeExamples(trainExs, tmpAvg);
      testTypeExamples(testExs, tmpAvg);
    }
    
    NerTesting.divdeNumber(weightSum, updateCnt.toDouble);
    weightSum;
  }

  
  def testTypeExamples(testExs: Seq[MentExample], weight: Array[Double]) {

    var total: Double = 0;
    var correct: Double = 0;
    
    for (testEx <- testExs) {
      val goldLabelIdx = testEx.getGoldTypeIndex();
      val bestLbl = testEx.getArgMax(testEx.features, weight);
      total += 1;
      if (goldLabelIdx == bestLbl) {
          correct += 1;
      }
    }

    val accuracy = correct / total;
    println("Total = " + total + ", correct = " + correct + ", acc = " + accuracy);
  }
  
  def testPredictor(testExs: Seq[MentExample], prdtr: AceMentionTypePredictor) {

    var total: Double = 0;
    var correct: Double = 0;
    
    for (testEx <- testExs) {
      val goldLabelIdx = testEx.getGoldTypeIndex();
      val predTp = prdtr.predictType(testEx);//testEx.getArgMax(testEx.features, weight);
      val bestLbl = AceMentionTypePredictor.MentTypIndexer.getIndex(predTp);
      total += 1;
      if (goldLabelIdx == bestLbl) {
          correct += 1;
      }
    }

    val accuracy = correct / total;
    println("Total = " + total + ", correct = " + correct + ", acc = " + accuracy);
  }
  
  
  ////////////////////////////////////////////////////
  ////////////////////////////////////////////////////
  ////////////////////////////////////////////////////
  
  def predictTypeAll(testExs: Seq[MentExample], booster: Booster) {
    var total: Double = 0;
    var correct: Double = 0;
    for (testEx <- testExs) {
      val goldLabelIdx = testEx.getGoldTypeIndex();
      val bestLbl = predictTypeBooster(testEx, booster);
      total += 1;
      if (goldLabelIdx == bestLbl) {
          correct += 1;
      }
    }
    val accuracy = correct / total;
    println("Total = " + total + ", correct = " + correct + ", acc = " + accuracy);
  }
  
  def predictTypeBooster(ment: MentExample, booster: Booster): Int = {
    val (oneInstanMtrx, wdt) = mentExsToDMatrix(Seq(ment), false);
    val predicts2 = booster.predict(oneInstanMtrx);
    var bestLbl = -1;
		var bestScore = -Double.MaxValue;
		for (j <- 0 until AceMentionTypePredictor.MentTypIndexer.size) {
			var score = predicts2(j)(0);
			if (score > bestScore) {
				bestScore = score;
				bestLbl = j;
			}
		}
		(bestLbl);
  }
  
  def runQuickLearning(trainExs: Seq[MentExample], testExs: Seq[MentExample]) : Booster = {
    
    val (trainMtrx, trWidth) = mentExsToDMatrix(trainExs, true);
    val (testMtrx, tstWidth) = mentExsToDMatrix(testExs, true);
	  
    //// train
	  val bstr = performLearningGvienTrainTestDMatrix(trainMtrx, testMtrx);
	  bstr;
	}

  def mentExsToDMatrix(mentExs: Seq[MentExample], verbose: Boolean): (DMatrix, Int) = {

    var totalCnt = 0;
    var posCnt = 0;
    var negCnt = 0;
    var maxFeatIdx = 0;
    val tlabels = new ArrayBuffer[Float]();
    val tdata   = new ArrayBuffer[Float]();
    val theaders = new ArrayBuffer[Long]();
    val tindex = new ArrayBuffer[Int]();
    val tgroup = new ArrayBuffer[Int]();

    var gpCnt: Int = 0;
    var rowCnt: Int = 0;
    var rowheader:Long = 0L;
    theaders += (rowheader);
    
    for (ment <- mentExs) {
      gpCnt += 1;
      
      if (gpCnt % 10000 == 0) {
        println("Feature " + gpCnt + " coref ments.");
      }
      
      val feats = ment.features;
      for (j <- 0 until AceMentionTypePredictor.MentTypIndexer.size) {
        rowCnt += 1;
        val featIdxs = feats(j).sortWith{_ < _};
        val lbl = if (ment.getGoldTypeIndex == j) 1.0f else 0.0f
        // dump features
        var thisMaxIdx = -1;
        for (i <- 0 until featIdxs.length) {
    			tdata += (1.0f);
    			val fidx = featIdxs(i) + 1
    			tindex += (fidx);
    			if (fidx > maxFeatIdx) {
    			  maxFeatIdx = fidx;
    			}
    			if (fidx > thisMaxIdx) {
    			  thisMaxIdx = fidx;
    			}
    		}
        
        var totalFeats = featIdxs.length.toLong;
        
    		rowheader += totalFeats;
    		theaders += (rowheader);
    		tlabels += (lbl);
    	}
      tgroup += (AceMentionTypePredictor.MentTypIndexer.size);
    }
    
    val splabels: Array[Float] = tlabels.toArray;
    val spgroups: Array[Int] = tgroup.toArray;
    val spdata: Array[Float] = tdata.toArray;
    val spcolIndex: Array[Int] = tindex.toArray;
    val sprowHeaders: Array[Long] = theaders.toArray;
    
    if (verbose) {
    	println("splabels = " + splabels.length);
    	println("spgroups = " + spgroups.length);
    	println("spdata = " + spdata.length);
    	println("spcolIndex = " + spcolIndex.length);
    	println("sprowHeaders = " + sprowHeaders.length);
    }
    
    
    val mx = new DMatrix(sprowHeaders, spcolIndex, spdata, JDMatrix.SparseType.CSR);
    mx.setLabel(splabels);
    mx.setGroup(spgroups);
    
    // print some statistics
    if (verbose) {
    	println("Groups: " + gpCnt);
    	println("Rows: " + rowCnt);
    	println("Max feature index: " + maxFeatIdx);
    }

    
    (mx, maxFeatIdx);
  }
  
  def performLearningGvienTrainTestDMatrix(trainMax: DMatrix, testMax: DMatrix): Booster = {
    
    //// train
	  
    println("Trainset size: " + trainMax.rowNum);
	  println("Testset size: " + testMax.rowNum);
    
    val params = new HashMap[String, Any]()
    val round = 370
    params += "eta" -> 0.1
    params += "max_depth" -> 50
    params += "silent" -> 0
    params += "objective" -> "rank:pairwise"
    params += "eval_metric" -> "pre@1"
    params += "nthread" -> 4
      
    val watches = new HashMap[String, DMatrix]()
    watches += "train" -> trainMax
    watches += "test" -> testMax

    // train a model
    val booster = XGBoost.train(trainMax, params.toMap, round, watches.toMap)
    
    // predict
    val predicts = booster.predict(testMax)
    // save model to model path
    val file = new File("./model")
    if (!file.exists()) {
      file.mkdirs()
    }
    booster.saveModel(file.getAbsolutePath + "/xgb-mentyp.model")
    
    booster;
  }
}

class AceMentionTypePredictor(val featurizer: AceMentionTypeFeaturizer, val weights: Array[Double]) {
  
  def scoreType(feats: Array[Int]): Double = {
    NerTesting.computeScore(weights, feats);
  }
  
  def predictType(mex: MentExample): MentionType = {
    val features = featurizer.featurizeExample(mex, false);
    var bestLbl = -1;
		var bestScore = -Double.MaxValue;
		for (j <- 0 until AceMentionTypePredictor.MentTypIndexer.size) {
			var score =  scoreType(features(j));
			if (score > bestScore) {
				bestScore = score;
				bestLbl = j;
			}
		}
		AceMentionTypePredictor.MentTypIndexer.getObject(bestLbl);
  }
  /*
  def predictType(m: Mention): MentionType = {
    val tempMentEx = AceMentionTypePredictor.mentionToTypeEx(m, m.rawDoc);
    predictType(tempMentEx);
  }
  */
   
 
}

class GoldMentionTypePredictor() extends AceMentionTypePredictor(null, null) {
  
  override def predictType(mex: MentExample): MentionType = {
    mex.goldTyp;
  }

}

class AceMentionTypeFeaturizer(val featureIndexer: Indexer[String],
                               val labelIndexer: Indexer[MentionType],
                               val corpusCounts: CorpusCounts,
                               val brownClusters: Option[Map[String,String]] = None) extends Serializable  {
   
  val useBrown = brownClusters.isDefined;
  val useBackoffs = true;//featureSet.contains("backoff");
  val useWikipedia = false;//featureSet.contains("wikipedia") && wikipediaDB.isDefined;
  val useExtraBrown = false;//featureSet.contains("extra-brown") && brownClusters.isDefined;
  val usePrefSuff = true;//featureSet.contains("prefsuff");
  
  
  def featurizeExample(mentEx: MentExample, addToIndexer: Boolean): Array[Array[Int]] = {
    val tokenFeats = featurizeEachTokens(mentEx);
    val wholeFeats = featurizeAddon(mentEx);
    
    val feats = new ArrayBuffer[Array[Int]]();
    for (labelIdx <- 0 until labelIndexer.size) {
      val typLabel = labelIndexer.getObject(labelIdx).toString();
      val lbFeats = new ArrayBuffer[Int]();
      
      for (tokId <- 0 until tokenFeats.length) {
        val tkFeat = tokenFeats(tokId);
        for (tft <- tkFeat) {
          val tft2 = typLabel + ":" + tft;
          if (addToIndexer || featureIndexer.contains(tft2)) lbFeats += featureIndexer.getIndex(tft2);
        }
      }
      for (wft <- wholeFeats) {
        val wft2 = typLabel + ":" + wft;
        if (addToIndexer || featureIndexer.contains(wft2)) lbFeats += featureIndexer.getIndex(wft2);
      }
      
      ////
      feats += (lbFeats.toArray);
    }
      
      /*
        val nerLabel = labelIndexer.getObject(labelIdx);
        val structuralType = NerSystemLabeled.getStructuralType(nerLabel);
        if (tokIdx == 0 && structuralType == "I") {
          null;
        } else {
          val feats = new ArrayBuffer[Int]();
          for (feat <- cachedSurfaceFeats) {
            val labeledFeat = feat + ":" + nerLabel;
            if (addToIndexer || featureIndexer.contains(labeledFeat)) feats += featureIndexer.getIndex(labeledFeat)
            if (useBackoffs && structuralType != "O") {
              val partiallyLabeledFeat = feat + ":" + structuralType;
              if (addToIndexer || featureIndexer.contains(partiallyLabeledFeat)) feats += featureIndexer.getIndex(partiallyLabeledFeat)
            }
          }
          feats.toArray;
        }
      */
    feats.toArray;
  }
 
  
  def featurizeAddon(ex: MentExample): Array[String] =  {
    val wholeMentFeats = new ArrayBuffer[String]();
    val endIdx = ex.endIdx;
    val startIdx = ex.startIdx;
    val headIdx = ex.headIdx;
    val words = ex.words;
    val poss = ex.poss;
    val langPack = new EnglishCorefLanguagePack();
    
    val len = endIdx - startIdx;
    wholeMentFeats += ("MentTkLen=" + String.valueOf(len));
    if ((len == 1) && PronounDictionary.isDemonstrative(words(headIdx))) {
    	wholeMentFeats += ("SingleWdDmnstr=1");//	MentionType.DEMONSTRATIVE;
    }
    if ((len == 1) && (PronounDictionary.isPronLc(words(headIdx)) || langPack.getPronominalTags.contains(poss(headIdx)))) {
    	wholeMentFeats += ("SingleWdPrn=1");//MentionType.PRONOMINAL;
    }
    if (langPack.getProperTags.contains(poss(headIdx))) {
    	wholeMentFeats += ("ProperTag=1");//MentionType.PROPER;
    }
    wholeMentFeats.toArray;
  }

      
  //def featurize(ex: MentExample, addToIndexer: Boolean, rangeRestriction: Option[(Int,Int)] = None): Array[Array[Array[Int]]] = {
  def featurizeEachTokens(ex: MentExample): Array[Array[String]] = {
    val range = (ex.startIdx -> (ex.headIdx + 1));//if (rangeRestriction.isDefined) rangeRestriction.get else (0, ex.words.size);
    val cachedShapes = ex.words.map(NerFeaturizer.shapeFor(_));
    val cachedClasses = ex.words.map(NerFeaturizer.classFor(_));
    val cachedPrefixes = ex.words.map(NerFeaturizer.prefixFor(_))
    val cachedSuffixes = ex.words.map(NerFeaturizer.suffixFor(_))
    
    val eachTokenFeats = Array.tabulate(range._2 - range._1)(tokIdxOffset => { // feat(token) = token_feat
      val tokIdx = tokIdxOffset + range._1;
      val wordAt = (i: Int) => ex.wordAt(tokIdx + i); 
      val posAt = (i: Int) => ex.posAt(tokIdx + i);
      val wordShapeAt = (i: Int) => if (tokIdx + i < 0) "<<START>>" else if (tokIdx + i >= ex.words.size) "<<END>>" else cachedShapes(tokIdx + i);
      val wordClassAt = (i: Int) => if (tokIdx + i < 0) "<<START>>" else if (tokIdx + i >= ex.words.size) "<<END>>" else cachedClasses(tokIdx + i);
      val prefixAt = (i: Int) => if (tokIdx + i < 0) "<ST>" else if (tokIdx + i >= ex.words.size) "<EN>" else cachedPrefixes(tokIdx + i);
      val suffixAt = (i: Int) => if (tokIdx + i < 0) "<ST>" else if (tokIdx + i >= ex.words.size) "<EN>" else cachedSuffixes(tokIdx + i);
      val cachedSurfaceFeats = new ArrayBuffer[String];
      
      for (offset <- -2 to 2) {
        if (corpusCounts.unigramCounts.containsKey(wordAt(offset))) cachedSurfaceFeats += offset + "W=" + wordAt(offset);
        cachedSurfaceFeats += offset + "P=" + posAt(offset);
        cachedSurfaceFeats += offset + "S=" + wordShapeAt(offset);
        cachedSurfaceFeats += offset + "C=" + wordClassAt(offset);
        if (usePrefSuff) {
          if (corpusCounts.prefixCounts.containsKey(prefixAt(offset))) cachedSurfaceFeats += offset + "PR=" + prefixAt(offset);
          if (corpusCounts.suffixCounts.containsKey(suffixAt(offset))) cachedSurfaceFeats += offset + "SU=" + suffixAt(offset);
        }
        if (offset < 2) {
          cachedSurfaceFeats += offset + "SS=" + wordShapeAt(offset) + "," + wordShapeAt(offset+1);
          val bigram = wordAt(offset) -> wordAt(offset+1);
          if (corpusCounts.bigramCounts.containsKey(bigram)) {
            cachedSurfaceFeats += offset + "WW=" + wordAt(offset) + "," + wordAt(offset+1);
          }
        }
      }
      //cachedSurfaceFeats ++= wikiFeats(tokIdx);
      if (useBrown) {
        for (offset <- if (useExtraBrown) (-1 to 1) else (0 to 0)) {
          val brownStr = if (brownClusters.get.contains(wordAt(offset))) brownClusters.get(wordAt(offset)) else "";
          // 4, 6, 10, 20 is prescribed by Ratinov et al. (2009) and Turian et al. (2010)
          cachedSurfaceFeats += offset + "B4=" + brownStr.substring(0, Math.min(brownStr.size, 4));
          cachedSurfaceFeats += offset + "B6=" + brownStr.substring(0, Math.min(brownStr.size, 6));
          cachedSurfaceFeats += offset + "B10=" + brownStr.substring(0, Math.min(brownStr.size, 10));
          cachedSurfaceFeats += offset + "B20=" + brownStr.substring(0, Math.min(brownStr.size, 20));
        }
      }
      cachedSurfaceFeats.toArray;
    });
    
    
    /////////////////////////////////////////////////////////////////
    /*
     Array.tabulate(labelIndexer.size)(labelIdx => {
        val nerLabel = labelIndexer.getObject(labelIdx);
        val structuralType = NerSystemLabeled.getStructuralType(nerLabel);
        if (tokIdx == 0 && structuralType == "I") {
          null;
        } else {
          val feats = new ArrayBuffer[Int]();
          for (feat <- cachedSurfaceFeats) {
            val labeledFeat = feat + ":" + nerLabel;
            if (addToIndexer || featureIndexer.contains(labeledFeat)) feats += featureIndexer.getIndex(labeledFeat)
            if (useBackoffs && structuralType != "O") {
              val partiallyLabeledFeat = feat + ":" + structuralType;
              if (addToIndexer || featureIndexer.contains(partiallyLabeledFeat)) feats += featureIndexer.getIndex(partiallyLabeledFeat)
            }
          }
          feats.toArray;
        }
      });
    */
    eachTokenFeats;
  }

}