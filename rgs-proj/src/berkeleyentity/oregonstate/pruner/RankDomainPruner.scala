package berkeleyentity.oregonstate.pruner

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import edu.berkeley.nlp.futile.fig.basic.Indexer
import java.io.File
import berkeleyentity.oregonstate.QueryWikiValue
import berkeleyentity.oregonstate.AceJointTaskExample
import berkeleyentity.oregonstate.VarValue


class RankDomainPruner(val weight: Array[Double],
                       val topK1: Int,
                       val topK2: Int,
                       val topK3: Int) extends StaticDomainPruner {
  
	// special process?
	val topKCoref = topK1;
	val topKNer = topK2;
	val topKWiki = topK3
  
  def pruneDomainBatch(testStructs: Seq[AceJointTaskExample], topK1: Int, topK2: Int, topK3: Int) {
    for (ex <- testStructs) {
      pruneDomainOneExmp(ex, weight, topK1, topK2, topK3);
    }
  }

  def pruneDomainOneExmp(ex: AceJointTaskExample, wght: Array[Double], topK1: Int, topK2: Int, topK3: Int) {
    for (i <- 0 until ex.corefVars.size) {
      variableUnaryScoring[Int](ex.corefVars(i).values, wght, topK1);
    }
    for (i <- 0 until ex.nerVars.size) {
      variableUnaryScoring[String](ex.nerVars(i).values, wght, topK2);
    }
    for (i <- 0 until ex.wikiVars.size) {
      variableUnaryScoring[QueryWikiValue](ex.wikiVars(i).values, wght, topK3);
    }
  }

  def testRecallWithOraclePredictor() {
      // pruner parameters

  }
  
  override def scoreCorefValue(cvalue: VarValue[Int]): Double = {
    cvalue.computeScore(weight);
  };
  
  override def scoreNerValue(nvalue: VarValue[String]): Double = {
		nvalue.computeScore(weight);
  };
  
  override def scoreWikiValue(wvalue: VarValue[QueryWikiValue]): Double = {
    wvalue.computeScore(weight);
  };
  

  // do pruning!
  def variableUnaryScoring[T](varValues: Array[VarValue[T]], wght: Array[Double], topK: Int) {
    val valueElements = new ArrayBuffer[DomainElement]();
    for (i <- 0 until varValues.size) {
      val score = varValues(i).computeScore(wght);
      val ve = new DomainElement(i, score);
      valueElements += ve;
      varValues(i).isPruned = true;
      varValues(i).unaryScore = score;
    }
    
    // sort!
    val sortv = (valueElements.toSeq.sortWith(_.rankingWeight > _.rankingWeight)).toArray;
    //for (decs <- sortc) {
    //  println("c: " + decs.score + " " + decs.isCorrect);
    //}

    val nonPrunedValuesNumber = if (sortv.size > topK) topK else sortv.size
    for (j <- 0 until nonPrunedValuesNumber) {
      val idx = sortv(j).vIndex;
      varValues(idx).isPruned = false; // keep top K values! 
    }
  }

  

}

object RankDomainPruner {
  
  ///////////////////////////////////////////////////////////////////
  //// Pruner Learning  /////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////
  
  def runPrunerTraining() {
    
  }
/*  
  def runQuickLearning(trainExs: Seq[AceMultiTaskExample], 
                       testExs: Seq[AceMultiTaskExample]) {
	  
	  ///// data prepare

	  val trainTestSplitter = new DataPointsSplitter();
	  val tfeaturizer = new TitleFeaturizer(new Indexer[String](), new KeyWordDictionary(), new SuperDictionary(), maybeAspectDicts);
	  val trainDataProportion: Double = 0.8;
	  
	  
	  val (trainMax, testMax) = if ((maybeAllOpers != None) && (maybeAllProducts == None)) {
		  val (trainCv, testCv) = trainTestSplitter.trainTestSplitOperations(maybeAllOpers.get, trainDataProportion);
		  XgbMatrixBuilder.dumpBinaryMtrxToFile(trainCv, "model/trainFeatDump.txt");
		  XgbMatrixBuilder.dumpBinaryMtrxToFile(testCv, "model/testFeatDump.txt")
		  //val trnm = XgbMatrixBuilder.getDMatrixFromUserOper(trainCv);
		  //val tstm = XgbMatrixBuilder.getDMatrixFromUserOper(testCv);
		  val trnm = new DMatrix("model/trainFeatDump.txt");
		  val tstm = new DMatrix("model/testFeatDump.txt");
		  (trnm, tstm);
	  } else if ((maybeAllOpers == None) && (maybeAllProducts != None)) {
      XgbMatrixBuilder.prepareDMatrixFromScratch(maybeAllProducts.get, trainDataProportion, tfeaturizer);
	  } else {
		  throw new RuntimeException("No input data was specified!");
	  }
	  
    //// train
	  performLearningGvienTrainTestDMatrix(trainMax, testMax);
	}

  
  def performLearningGvienTrainTestDMatrix(trainMax: DMatrix, testMax: DMatrix) {
    
    //// train
	  
    println("Trainset size: " + trainMax.rowNum);
	  println("Testset size: " + testMax.rowNum);
    
    val params = new HashMap[String, Any]()
    val round = 500
    //params += "distribution" -> "bernoulli"
    params += "eta" -> 0.1
    params += "max_depth" -> 10
    params += "silent" -> 0
    //params += "colsample_bytree" -> 0.9
    //params += "min_child_weight" -> 10
    params += "objective" -> "binary:logistic"
    params += "eval_metric" -> "auc"
    params += "nthread" -> 8
      
    val watches = new HashMap[String, DMatrix]
    watches += "train" -> trainMax
    watches += "test" -> testMax


    // train a model
    val booster = XGBoost.train(trainMax, params.toMap, round, watches.toMap)
    
    //val bestScore = booster.
    //val bestIteration = model.best_iteration + 1   // note that xgboost start building tree with index 0
    //print("best_score: %s" % best_score)
    //print("opmital # of trees: %s" % best_iteration)
    
    // predict
    val predicts = booster.predict(testMax)
    // save model to model path
    val file = new File("./model")
    if (!file.exists()) {
      file.mkdirs()
    }
    booster.saveModel(file.getAbsolutePath + "/xgb.model")
    // dump model
    //booster.getModelDump(file.getAbsolutePath + "/dump.raw.txt", true)
    // dump model with feature map
    //booster.getModelDump(file.getAbsolutePath + "/featmap.txt", true)
    // save dmatrix into binary buffer
    trainMax.saveBinary(file.getAbsolutePath + "/dtrain.buffer")
    testMax.saveBinary(file.getAbsolutePath + "/dtest.buffer")
    
    ///////// test
    
    // reload model and data
    val booster2 = XGBoost.loadModel(file.getAbsolutePath + "/xgb.model")
    val testMax2 = new DMatrix(file.getAbsolutePath + "/dtest.buffer")
    val predicts2 = booster2.predict(testMax2);

    // check predicts
    val predValCnt = new HashSet[Float]();
    println(predicts2.size + "," + predicts2(0).size);
    for (i <- 0 until predicts.size) {
      //println("Score(" + i + ") = " + predicts(i)(0));
      predValCnt += predicts(i)(0);
    }
    println("Value size = " + predValCnt.size + " / " + predicts.size);
    //println(checkPredicts(predicts, predicts2))
  }
  

  def productsToDMatrix(products: Seq[ProductItem]) : DMatrix = {
    
    var totalCnt = 0;
    var posCnt = 0;
    var negCnt = 0;
    var maxFeatIdx = 0;
    val tlabels = new ArrayBuffer[Float]();
    val tdata = new ArrayBuffer[Float]();
    val theaders = new ArrayBuffer[Long]();
    val tindex = new ArrayBuffer[Int]();

    var rowheader:Long = 0L;
    theaders += (rowheader);
    
    for (pdt <- products) {
    	val opers = UserOperation.generateUserOperationsFromItemTrainingData(pdt);
    	val featIdxs = pdt.features;
    	for (oper <- opers) {
    	  // count label
    	  totalCnt += 1;
    	  if (oper.getClickLabel() > 0) {
    	    posCnt += 1;
    	  } else {
    	    negCnt += 1;
    	  }
    		for (i <- 0 until featIdxs.length) {
    			tdata += (1.0f);
    			val fidx = featIdxs(i) + 1
    			tindex += (fidx);
    			if (fidx > maxFeatIdx) {
    			  maxFeatIdx = fidx;
    			}
    		}
    		rowheader += featIdxs.length.toLong;
    		theaders += (rowheader);
    		tlabels += (oper.getClickLabel.toFloat);
    	}
    }
    
    val splabels: Array[Float] = tlabels.toArray;
    val spdata: Array[Float] = tdata.toArray;
    val spcolIndex: Array[Int] = tindex.toArray;
    val sprowHeaders: Array[Long] = theaders.toArray;
    //val spData = new CSRSparseData(splabels, spdata, spcolIndex, sprowHeaders);
    //spData;
    val mx = new DMatrix(sprowHeaders, spcolIndex, spdata, JDMatrix.SparseType.CSR);
    mx.setLabel(splabels);
    //mx.setWeight(weights);
    
    // print some statistics
    
    println("Products: " + products.size);
    println("Rows: " + totalCnt + " = " + posCnt + " / " + negCnt);
    println("Max feature index: " + maxFeatIdx);
    
    mx;
  }
*/
}