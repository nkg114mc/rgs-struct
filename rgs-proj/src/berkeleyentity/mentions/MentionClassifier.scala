package berkeleyentity.mentions

import java.io.File
import java.io.PrintWriter
import berkeleyentity.Driver
import berkeleyentity.ConllDocReader
import berkeleyentity.lang.EnglishCorefLanguagePack
import berkeleyentity.coref.CorefDocAssemblerCopy
import berkeleyentity.coref.NumberGenderComputer
import berkeleyentity.coref.MentionPropertyComputer
import berkeleyentity.coref.Mention
import berkeleyentity.coref.CorefDoc
import berkeleyentity.sem.QueryCountsBundle
import berkeleyentity.sem.BasicWordNetSemClasser
import berkeleyentity.coref.FeatureSetSpecification
import berkeleyentity.sem.SemClasser
import berkeleyentity.coref.LexicalCountsBundle
import edu.berkeley.nlp.futile.fig.basic.Indexer
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.java.{ DMatrix => JDMatrix }
import ml.dmlc.xgboost4j.scala.XGBoost
import ml.dmlc.xgboost4j.scala.Booster
import util.control.Breaks._
import berkeleyentity.coref.CorefDocAssemblerFile

class MentionPredictionInstance(val ment: Mention, 
                                val goldLabel: Int) {
  var feature: Array[Int] = null;
  var predictLabel: Int = 0;
  var predictScore: Double = 0;
  var isTrain = false;
}

object MentionClassifier {
  
	def main(args: Array[String]) {
		runMentionPredictionClassify();
	}
/*
  def runMentionPredictionClassify() {
    
    val path1 = "/home/mc/workplace/rand_search/coref/berkfiles/data/ontonotes5/train";
    val suffix1 = "v4_auto_conll"
    val path2 = "/home/mc/workplace/rand_search/coref/berkfiles/data/ontonotes5/test";
    val suffix2 = "v9_auto_conll"
    
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    val assembler = new CorefDocAssemblerFile(new EnglishCorefLanguagePack(), false);
    val featureIndexer = new Indexer[String]();
    
    val rawDocs = ConllDocReader.loadRawConllDocsWithSuffix(path1, -1, suffix1);
    val corefDocs = rawDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    
    val rawTestDocs = ConllDocReader.loadRawConllDocsWithSuffix(path2, -1, suffix2);
    val testDocs = rawTestDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    
    val trnExs = extractMentInstannces(corefDocs, featureIndexer, true)
    val tstExs = extractMentInstannces(testDocs, featureIndexer, false)
    
    val modelPath = "./model/mention.xgb.model"
    //val predictor = loadOrLearn(trnExs, tstExs, modelPath)
    val predictor = loadOrLearn(trnExs ++ tstExs, tstExs, modelPath)
    
    predictLabels(trnExs, predictor)
    predictLabelsWithPrecision(tstExs, predictor, 0.8)
    
    // evaluation
    runEvaluation(tstExs, predictor)
    

    checkDocNames(trnExs, tstExs)
    val allExs = new ArrayBuffer[MentionPredictionInstance]();
    allExs ++= (trnExs)
    allExs ++= (tstExs)
    
    //dumpMentionPredictionWithLabel(allExs, "mentDump.txt", predictor);
    dumpMentionPredictionWithLabel(allExs, "mentDump80.txt", predictor);
    //dumpMentionPredictionIgnoreLabel(allExs, "mentDumpFull.txt");

  }
*/
  def runMentionPredictionClassify() {
    
    val path1 = "/home/mc/workplace/rand_search/coref/berkfiles/data/ontonotes5/train";
    val suffix1 = "v4_auto_conll"
    val path2 = "/home/mc/workplace/rand_search/coref/berkfiles/data/ontonotes5/test";
    val suffix2 = "v9_auto_conll"
    
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    val assembler = new CorefDocAssemblerFile(new EnglishCorefLanguagePack(), false);
    val featureIndexer = new Indexer[String]();
    
    val rawDocs = ConllDocReader.loadRawConllDocsWithSuffix(path1, -1, suffix1);
    val corefDocs = rawDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    
    val rawTestDocs = ConllDocReader.loadRawConllDocsWithSuffix(path2, -1, suffix2);
    val testDocs = rawTestDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    
    val trnExs = extractMentInstannces(corefDocs, featureIndexer, true)
    val tstExs = extractMentInstannces(testDocs, featureIndexer, false)
    
    val modelPath = "./model/mention.xgb.model"
    val predictor = loadOrLearn(trnExs, tstExs, modelPath)
    //val predictor = loadOrLearn(trnExs ++ tstExs, tstExs, modelPath)
    
    predictLabels(trnExs, predictor)
    predictLabels(tstExs, predictor)
    
    // evaluation
    runEvaluation(trnExs, predictor)
    runEvaluation(tstExs, predictor)
    
    val allExs = new ArrayBuffer[MentionPredictionInstance]();
    allExs ++= (trnExs)
    allExs ++= (tstExs)
    dumpMentionPredictionWithLabel(allExs, "mentDumpHot0.1.txt", predictor);

  }
  
  def predictLabels(mentions: Seq[MentionPredictionInstance], model: Booster) {
    for (mi <- mentions) {
      mi.predictLabel = predictLabel(mi, model)
      mi.predictScore = predictScore(mi, model)
    }
  }
  
  def predictLabelsWithPrecision(mentions: Seq[MentionPredictionInstance], model: Booster, precision: Double) {
    for (mi <- mentions) {
      mi.predictLabel = predictLabel(mi, model)
      mi.predictScore = predictScore(mi, model)
    }
    
    val scoredMents = mentions.toList.sortWith(_.predictScore < _.predictScore)
    var cnt = 0;
    breakable {
    	for (i <- 0 until scoredMents.size) {
    	  val mi = scoredMents(i)
    	  if (mi.predictLabel > 0 && mi.goldLabel == 0) {
    	    mi.predictLabel = 0
    		  cnt += 1
    			if (cnt > 5000) {
    				 break;
    			}
    	  }
    	}
    }
    
  }
  
  def dumpMentionPredictionWithLabel(mentions: Seq[MentionPredictionInstance], path: String,  model: Booster) {
    println("Predict labels ...")
    // predict
    //predictLabels(mentions, model)
    //predictLabelsWithPrecision(mentions, model, 0.8)
    
    var cnt = 0
    println("Start dumping " + path + "...")
    val docSet = new HashSet[String]();
    val writer = new PrintWriter(path)
    for (mi <- mentions) {
      cnt += 1
      if (cnt % 100000 == 0) {
        println("Dump " + cnt + " mentions...")
      }
      val docID = mi.ment.rawDoc.getDocNameWithPart()
      docSet += docID
      val predLb = mi.predictLabel//predictLabel(mi, model)
      if (predLb > 0) {
        val m = mi.ment
        val trnName = if (mi.isTrain) "train" else "test"
        writer.println(docID + "\t" + m.sentIdx + "\t" + m.startIdx + "\t" + m.endIdx + "\t" + m.headIdx + "\t" + trnName)
      }
    }
    writer.close()
    
    println("DocCnt = " + docSet.size)
  }
  
  def dumpMentionPredictionIgnoreLabel(mentions: Seq[MentionPredictionInstance], path: String) {
    val docSet = new HashSet[String]();
    val writer = new PrintWriter(path)
    for (mi <- mentions) {
      val docID = mi.ment.rawDoc.getDocNameWithPart()
      docSet += docID
      val m = mi.ment
      writer.println(docID + "\t" + m.sentIdx + "\t" + m.startIdx + "\t" + m.endIdx + "\t" + m.headIdx)
    }
    writer.close()
    println("DocCnt = " + docSet.size)
  }

  def extractMentInstannces(corefDocs: Seq[CorefDoc], 
                            featureIndexer: Indexer[String], 
                            isTrain: Boolean) = {

    val mentFeaturizer = getMentFeaturizer(featureIndexer, corefDocs)
    val mentInsts = new ArrayBuffer[MentionPredictionInstance]();
    val docSet = new HashSet[String]();
    
    var mcnt = 0;
    var posn = 0;
    var negn = 0;
    for (doc <- corefDocs) {
      //println(doc.rawDoc.docID)
      //println(doc.rawDoc.getDocNameWithPart());
      docSet += doc.rawDoc.getDocNameWithPart()
      val predMents = doc.predMentions;
      val goldMents = doc.goldMentions;
      for (pm <- predMents) {
        mcnt += 1
        val lb = getLabel(pm, goldMents)
        val mInst = (new MentionPredictionInstance(pm, lb))
        mInst.feature = mentFeaturizer.featurizeIndex(pm, isTrain)
        mInst.isTrain = isTrain
        mentInsts += mInst
        if (lb > 0) {
          posn += 1
        } else {
          negn += 1
        }
      }
      
    }
    
    println("FeatSize = " + featureIndexer.size())
    println("MentCnt = " + mcnt + " pos =  " + posn + " negn = " + negn)
    println("DocCnt = " + docSet.size)
    mentInsts
  }
  
  def extractMentInstanncesNoFeature(corefDocs: Seq[CorefDoc]) = {
    val mentInsts = new ArrayBuffer[MentionPredictionInstance]();
    val docSet = new HashSet[String]();
    
    var mcnt = 0;
    var posn = 0;
    var negn = 0;
    for (doc <- corefDocs) {
      docSet += doc.rawDoc.getDocNameWithPart()
      val predMents = doc.predMentions;
      val goldMents = doc.goldMentions;
      for (pm <- predMents) {
        mcnt += 1
        val lb = getLabel(pm, goldMents)
        val mInst = (new MentionPredictionInstance(pm, lb))
        mentInsts += mInst
        if (lb > 0) {
          posn += 1
        } else {
          negn += 1
        }
      }
    }
    println("MentCnt = " + mcnt + " pos =  " + posn + " negn = " + negn)
    println("DocCnt = " + docSet.size)
    mentInsts
  }
  
  
  def checkDocNames(trainExs: Seq[MentionPredictionInstance],
                    testExs: Seq[MentionPredictionInstance]) {
    val dnames = new HashSet[String]()
    for (pm <- trainExs) {
      dnames += pm.ment.rawDoc.getDocNameWithPart()
    }
    
    for (n2 <- testExs) {
      if (dnames.contains(n2.ment.rawDoc.getDocNameWithPart())) {
        println("Test repeat name: " + n2.ment.rawDoc.getDocNameWithPart()) 
      }
    }
  }
  
  def runEvaluation(testExs: Seq[MentionPredictionInstance], model: Booster) {
    
    //predictLabelsWithPrecision(testExs, model, 0.8)
    
    var tpos = 0
    var tneg = 0
    var fpos = 0
    var fneg = 0
    for (pm <- testExs) {
    	val predLb = pm.predictLabel//predictLabel(pm, model)
    	val lbl = pm.goldLabel
    	if (predLb > 0 && lbl > 0) {
    		tpos += 1
    	} else if (predLb == 0 && lbl > 0) {
    		fneg += 1
    	} else if (predLb > 0 && lbl == 0) {
    		fpos += 1
    	} else if (predLb == 0 && lbl == 0) {
    		tneg += 1
    	}
    }
    
    val precision = tpos.toDouble / (tpos + fpos).toDouble
    println("Precision: " + tpos + " / " + (tpos + fpos) + " = " + precision);
    val recall = tpos.toDouble / (19764).toDouble
    println("Recall: " + tpos + " / " + 19764 + " = " + recall);
  }
  
  def predictLabel(ment: MentionPredictionInstance, model: Booster): Int = {
    val threshold = 0.1;
    val predictSc = predictScore(ment, model)
    val predLabel = if (predictSc > threshold) 1 else 0
    (predLabel);
  }
  
  def predictScore(ment: MentionPredictionInstance, model: Booster): Float = {
    var totalCnt = 0;
    var posCnt = 0;
    var negCnt = 0;
    var maxFeatIdx = 0;
    val tlabels = new ArrayBuffer[Float]();
    val tdata   = new ArrayBuffer[Float]();
    val theaders = new ArrayBuffer[Long]();
    val tindex = new ArrayBuffer[Int]();

    var rowCnt: Int = 0;
    var rowheader:Long = 0L;
    theaders += (rowheader);
    
    rowCnt += 1;
    if (rowCnt % 10000 == 0) {
    	println("Feature " + rowCnt + " ments.");
    }

    val featIdxs = ment.feature.sortWith{_ < _};
    val lbl = ment.goldLabel.toFloat
    		// dump features
    		for (i <- 0 until featIdxs.length) {
    			tdata += (1.0f);
    			val fidx = featIdxs(i) + 1
    					tindex += (fidx);
    		}


    var totalFeats = featIdxs.length.toLong;

    rowheader += totalFeats;
    theaders += (rowheader);
    tlabels += (lbl);
    
    val splabels: Array[Float] = tlabels.toArray;
    val spdata: Array[Float] = tdata.toArray;
    val spcolIndex: Array[Int] = tindex.toArray;
    val sprowHeaders: Array[Long] = theaders.toArray;

    val mx = new DMatrix(sprowHeaders, spcolIndex, spdata, JDMatrix.SparseType.CSR);
    
    val predictScArr = model.predict(mx)
    val predictSc = predictScArr(0)(0)
    (predictSc);
  }
  
  def loadOrLearn(trainExs: Seq[MentionPredictionInstance], 
                  testExs: Seq[MentionPredictionInstance],
                  modelPath: String) = {
    val mdFile = new File(modelPath);
    val md = if (mdFile.exists()) {
      XGBoost.loadModel(modelPath)
    } else {
      runLearning(trainExs,testExs,modelPath)
    }
    md
  }
  
  def runLearning(trainExs: Seq[MentionPredictionInstance], 
                  testExs: Seq[MentionPredictionInstance],
                  modelPath: String) = {
    val trnMax = makeDMatrix(trainExs, true)
    val tstMax = makeDMatrix(testExs, true)
    
    val params = new HashMap[String, Any]()
    val round = 800
    //params += "distribution" -> "bernoulli"
    params += "eta" -> 0.1
    params += "max_depth" -> 20
    params += "silent" -> 0
    //params += "colsample_bytree" -> 0.9
    //params += "min_child_weight" -> 10
    params += "objective" -> "binary:logistic"
    params += "eval_metric" -> "error"
    params += "nthread" -> 8
      
    val watches = new HashMap[String, DMatrix]
    watches += "train" -> trnMax
    watches += "test" -> tstMax

    // train a model
    val booster = XGBoost.train(trnMax, params.toMap, round, watches.toMap)
    
    // predict
    val predicts = booster.predict(tstMax)
    // save model to model path
    booster.saveModel(modelPath)//("./model/xgb.model")
    
    booster
  }
  
  def makeDMatrix(exs: Seq[MentionPredictionInstance], verbose: Boolean): DMatrix = {

    var totalCnt = 0;
    var posCnt = 0;
    var negCnt = 0;
    var maxFeatIdx = 0;
    val tlabels = new ArrayBuffer[Float]();
    val tdata   = new ArrayBuffer[Float]();
    val theaders = new ArrayBuffer[Long]();
    val tindex = new ArrayBuffer[Int]();

    var rowCnt: Int = 0;
    var rowheader:Long = 0L;
    theaders += (rowheader);
    
    for (ment <- exs) {
      rowCnt += 1;
      if (rowCnt % 10000 == 0) {
        println("Feature " + rowCnt + " ments.");
      }
 
        val featIdxs = ment.feature.sortWith{_ < _};
        val lbl = ment.goldLabel.toFloat
        		// dump features

        		for (i <- 0 until featIdxs.length) {
        			tdata += (1.0f);
        			val fidx = featIdxs(i) + 1
        					tindex += (fidx);
        		}


        var totalFeats = featIdxs.length.toLong;

        rowheader += totalFeats;
        theaders += (rowheader);
        tlabels += (lbl);
    }
    
    val splabels: Array[Float] = tlabels.toArray;
    val spdata: Array[Float] = tdata.toArray;
    val spcolIndex: Array[Int] = tindex.toArray;
    val sprowHeaders: Array[Long] = theaders.toArray;
    
    if (verbose) {
    	println("splabels = " + splabels.length);
    	println("spdata = " + spdata.length);
    	println("spcolIndex = " + spcolIndex.length);
    	println("sprowHeaders = " + sprowHeaders.length);
    }
    
    
    val mx = new DMatrix(sprowHeaders, spcolIndex, spdata, JDMatrix.SparseType.CSR);
    mx.setLabel(splabels);
    
    // print some statistics
    if (verbose) {
    	println("Rows: " + rowCnt);
    }
    
    (mx);
  }
  
  def getLabel(curMent: Mention, goldMentions: Seq[Mention]) = {
    val pm = curMent
    val matchGolds = goldMentions.filter(gm => (gm.sentIdx == pm.sentIdx && gm.startIdx == pm.startIdx && gm.endIdx == pm.endIdx))
    //matchGolds.map { m => println(m) }
    val matchSz = matchGolds.size
    //if (matchSz > 1) {
    //  throw new RuntimeException("More than one matched gold: " + matchSz)
    //}
    val label = if (matchSz >= 1) {
    	1;
    } else {
    	0
    }
    label
  }
  
  def getMentFeaturizer(featureIndexer: Indexer[String], corefDocs: Seq[CorefDoc]) = {
    featureIndexer.getIndex(MentionSpanFeaturizer.UnkFeatName);
    val queryCounts: Option[QueryCountsBundle] = None;
    val lexicalCounts = LexicalCountsBundle.countLexicalItems(corefDocs, Driver.lexicalFeatCutoff);
    val semClasser: Option[SemClasser] = Some(new BasicWordNetSemClasser);
    val corefFeatureSetSpec = FeatureSetSpecification(Driver.pairwiseFeats, Driver.conjScheme, Driver.conjFeats, Driver.conjMentionTypes, Driver.conjTemplates);
    val mentFeaturizer = new MentionSpanFeaturizer(featureIndexer, corefFeatureSetSpec, lexicalCounts, queryCounts, semClasser);
    mentFeaturizer;
  }
}