package berkeleyentity.xgb

import java.io._

import scala.collection.mutable.ArrayBuffer

import ml.dmlc.xgboost4j.java.{ DMatrix => JDMatrix }
import ml.dmlc.xgboost4j.scala.DMatrix

import berkeleyentity.oregonstate.CorefDecisionExample


object XgbMatrixBuilder {
  
  ////////////////////////////////////////////////////////////////////////////////
  ///// Construct DMatrix from product items  ////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  def corefExmpToDMatrix(corefMents: Seq[CorefDecisionExample], maxIdx: Int, verbose: Boolean): (DMatrix, Int) = {

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
    
    for (ment <- corefMents) {
      gpCnt += 1;
      
      if (gpCnt % 10000 == 0) {
        println("Feature " + gpCnt + " coref ments.");
      }
      
      val feats = ment.features;
      for (j <- 0 until ment.values.length) {
        rowCnt += 1;
        // for jth value
        val ante = ment.values(j);
        val featIdxs = feats(j).sortWith{_ < _};
        val lbl = if (ment.isCorrect(ante)) 1.0f else 0.0f
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
        /*if (maxIdx > 0) {
        	if (gpCnt == 1) {
        		if (thisMaxIdx < maxIdx) {
              tdata += (0.0f);
    			    tindex += (maxIdx);
    			    totalFeats += 1;
        		}
        	}
        }*/

        
    		rowheader += totalFeats;
    		theaders += (rowheader);
    		tlabels += (lbl);
    	}
      tgroup += (ment.values.length);
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
/*
  // do data split and translate to DMatrix
  def prepareDMatrixFromScratch(allProducts: Seq[ProductItem], trainPercent: Double, featurizer: TitleFeaturizer) : (DMatrix, DMatrix) = {
    
    // trainPercent
    
    val total = allProducts.size.toDouble;
    val trainCvSz = (total * trainPercent).toInt;
    val testCvSz = (total - trainCvSz).toInt;

    val trainProducts = allProducts.slice(0, trainCvSz);
    val testProducts = allProducts.slice(trainCvSz + 1, total.toInt);
    
    featurizer.featurizeAll(trainProducts, true);
    featurizer.featurizeAll(testProducts, false);
    println("Sparse feature size: " + featurizer.indexer.size());
    featurizer.printIndexedFeatureNames();

    val trnMtrx = productsToDMatrix(trainProducts);
    val tstMtrx = productsToDMatrix(testProducts);
    
    (trnMtrx, tstMtrx);
  }
  

  
  def productsToDMatrixDense(products: Seq[ProductItem], featurizer: TitleFeaturizer) : DMatrix = {
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
    	val featArr = featurizer.featurizeNonBinary(pdt);
    	for (oper <- opers) {
    	  // count label
    	  totalCnt += 1;
    	  if (oper.getClickLabel() > 0) {
    	    posCnt += 1;
    	  } else {
    	    negCnt += 1;
    	  }
    		for (i <- 0 until featArr.length) {
    			tdata += featArr(i).toFloat;
    			val fidx = i + 1
    			tindex += fidx;
    			if (fidx > maxFeatIdx) {
    			  maxFeatIdx = fidx;
    			}
    		}
    		rowheader += featArr.length.toLong;
    		theaders += (rowheader);
    		tlabels += (oper.getClickLabel.toFloat);
    	}
    }
    
    val splabels: Array[Float] = tlabels.toArray;
    val spdata: Array[Float] = tdata.toArray;
    val spcolIndex: Array[Int] = tindex.toArray;
    val sprowHeaders: Array[Long] = theaders.toArray;
    val mx = new DMatrix(sprowHeaders, spcolIndex, spdata, JDMatrix.SparseType.CSR);
    mx.setLabel(splabels);
    //mx.setWeight(weights);
    
    // print some statistics
    println("Products: " + products.size);
    println("Rows: " + totalCnt + " = " + posCnt + " / " + negCnt);
    println("Max feature index: " + maxFeatIdx);
    mx;
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ///// Construct DMatrix from user operations  //////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  def dumpBinaryMtrxToFile(allOpers: Seq[UserOperation], fn: String) {
	  val writer = new PrintWriter(fn);
	  for (oper <- allOpers) {
		  val featIdxs = oper.product.features;
		  var sb = new StringBuilder("");
		  sb.append(oper.getClickLabel());
		  for (i <- 0 until featIdxs.length) {
			  //val tdata = (1.0f);
			  val tindex = (featIdxs(i) + 1);
			  sb.append(" " + (tindex + ":" + "1"));
		  }
		  writer.println(sb.toString());
	  }
	  writer.close();
  }
/*
  def denseVectorToStr(featArr: Array[Double], label: Int): String = {
		  var sb = new StringBuilder("");
		  sb.append(label); // label
		  for (i <- 0 until featArr.length) {
			  val fidx = i + 1;
			  val fval = featArr(i).toFloat;
			  if (fval.isNaN() || fval.isInfinity) {
			    throw new RuntimeException(fidx + ":" + fval + " " + featArr(i));
			  }
			  sb.append(" " + (fidx + ":" + fval));
		  }
		  sb.toString();
  }
*/
  def sparseBinaryVectorToStr(featArr: Array[Int], label: Int, maxIdx: Int): String = {
		  var sb = new StringBuilder("");
		  sb.append(label); // label
		  var hasMaxSlot = false;
		  for (i <- 0 until featArr.length) {
			  val fidx = featArr(i) + 1;
			  if (fidx == maxIdx) {
			    hasMaxSlot = true;
			  }
			  val fval = 1.0f;
			  sb.append(" " + (fidx + ":" + fval));
		  }
		  if (maxIdx >= 0) { // going to use this
			  if (hasMaxSlot == false) {
				  sb.append(" " + (maxIdx + ":" + 0));
			  }
		  }
		  sb.toString();
  }
  
  // dump feature to file in SVM format
  def dumpDenseMtrxToFile(products: Seq[ProductItem], featurizer: TitleFeaturizer, fn: String, repeatAsOperations: Boolean) {
	  val writer = new PrintWriter(fn);

	  var totalCnt = 0;
	  var posCnt = 0;
	  var negCnt = 0;
	  var maxFeatIdx = 0;

	  for (pdt <- products) {
		  val opers = UserOperation.generateUserOperationsFromItemTrainingData(pdt);
		  val featArr = featurizer.featurizeNonBinary(pdt);
		  val fidx = featArr.length + 1;
		  if (fidx > maxFeatIdx) {
			  maxFeatIdx = fidx;
		  }
		  if (repeatAsOperations) {
			  /// Treat each user operation as a data point
			  for (oper <- opers) {
				  // count label
				  totalCnt += 1;
				  if (oper.getClickLabel() > 0)  posCnt += 1 else negCnt += 1;

				  //// Dump feature to file ////
				  val featStr = denseVectorToStr(featArr, oper.getClickLabel());
				  writer.println(featStr);
			  }
		  } else {
			  /// Treat each product as a data point
			  totalCnt += 1;
			  if (pdt.sumClick > 0) posCnt += 1 else negCnt += 1
				//// Dump feature to file ////
				val featStr = denseVectorToStr(featArr, pdt.getClickIntLabel());
			  writer.println(featStr);
		  }
		  
		  /// show count
		  if ((totalCnt % 100000) == 0) {
        println("Dumped " + totalCnt + " lines ...");
      }
	  }

	  // print some statistics
	  println("Products: " + products.size);
	  println("Rows: " + totalCnt + " = " + posCnt + " / " + negCnt);
	  println("Max feature index: " + maxFeatIdx);
	  writer.close();
  }
  
  // dump feature to file in SPARSE SVM format
  def dumpSparseMtrxToFile(products: Seq[ProductItem], featurizer: TitleFeaturizer, addToIndexer: Boolean, fn: String, repeatAsOperations: Boolean) {
	  val writer = new PrintWriter(fn);

	  var totalCnt = 0;
	  var posCnt = 0;
	  var negCnt = 0;
	  var maxFeatIdx = 0;
	  
	  val trainMaxFeatIdx: Int = if (addToIndexer) {
	    -1; // in training, no need
	  } else {
	    featurizer.indexer.size(); // in testing
	  }

	  for (pdt <- products) {
		  val opers = UserOperation.generateUserOperationsFromItemTrainingData(pdt);
		  val featArr = featurizer.featurize(pdt,addToIndexer);
		  // feat index counting
		  for (idx <- featArr) {
		    val fidx = idx + 1;
		    if (fidx > maxFeatIdx) {
		      maxFeatIdx = fidx;
		    }
		  }
      ///////////////////////////////////////////////
		  if (repeatAsOperations) {
			  /// Treat each user operation as a data point
			  for (oper <- opers) {
				  // count label
				  totalCnt += 1;
				  if (oper.getClickLabel() > 0)  posCnt += 1 else negCnt += 1;
				  //// Dump feature to file ////
				  val featStr = sparseBinaryVectorToStr(featArr, oper.getClickLabel(), trainMaxFeatIdx);
				  writer.println(featStr);
			  }
		  } else {
			  /// Treat each product as a data point
			  totalCnt += 1;
			  if (pdt.sumClick > 0) posCnt += 1 else negCnt += 1
				//// Dump feature to file ////
				val featStr = sparseBinaryVectorToStr(featArr, pdt.getClickIntLabel(), trainMaxFeatIdx);
			  writer.println(featStr);
		  }
		  
		  /// show count
		  if ((totalCnt % 100000) == 0) {
        println("Dumped " + totalCnt + " lines ...");
      }
	  }

	  // print some statistics
	  println("Products: " + products.size);
	  println("Rows: " + totalCnt + " = " + posCnt + " / " + negCnt);
	  println("Max feature index: " + maxFeatIdx);
	  writer.close();
  }
  
  
  
  def getDMatrixFromUserOper(allOpers: Seq[UserOperation]): DMatrix = {
    val spData = getCSRSparseDataFromUserOper(allOpers);
    val mx = new DMatrix(spData.rowHeaders, spData.colIndex, spData.data, JDMatrix.SparseType.CSR);
    mx.setLabel(spData.labels);
    mx;
  }
  
  def getCSRSparseDataFromUserOper(allOpers: Seq[UserOperation]): CSRSparseData = {
    
    val tlabels = new ArrayBuffer[Float]();
    val tdata = new ArrayBuffer[Float]();
    val theaders = new ArrayBuffer[Long]();
    val tindex = new ArrayBuffer[Int]();

    var rowheader:Long = 0L;
    theaders += (rowheader);
    
    for (oper <- allOpers) {
      val featIdxs = oper.product.features;
      for (i <- 0 until featIdxs.length) {
        tdata += (1.0f);
        tindex += (featIdxs(i) + 1);
      }
      rowheader += featIdxs.length.toLong;
      theaders += (rowheader);
      tlabels += (oper.getClickLabel.toFloat);
    }
    
    val splabels: Array[Float] = tlabels.toArray;
    val spdata: Array[Float] = tdata.toArray;
    val spcolIndex: Array[Int] = tindex.toArray;
    val sprowHeaders: Array[Long] = theaders.toArray;
    val spData = new CSRSparseData(splabels, spdata, spcolIndex, sprowHeaders);
    spData;
  }
*/
}