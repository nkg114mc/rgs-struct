package berkeleyentity.oregonstate


import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.ObjectInputStream
import java.io.ObjectOutputStream
import scala.Array.canBuildFrom
import scala.collection.JavaConverters.asScalaBufferConverter
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import berkeleyentity.Chunk
import berkeleyentity.Driver.WikifierType
import berkeleyentity.joint.FactorGraphFactoryACE
import berkeleyentity.joint.FactorGraphFactoryOnto
import berkeleyentity.joint.GeneralTrainer
import berkeleyentity.joint.JointComputerShared
import berkeleyentity.joint.JointDoc
import berkeleyentity.joint.JointDocACE
import berkeleyentity.joint.JointFeaturizerShared
import berkeleyentity.joint.JointLossFcns
import berkeleyentity.wiki.WikificationEvaluator
import berkeleyentity.wiki.WikipediaInterface
import berkeleyentity.lang.Language
import berkeleyentity.ner.MCNerFeaturizer
import berkeleyentity.ner.NEEvaluator
import berkeleyentity.ner.NerFeaturizer
import berkeleyentity.ner.NerSystemLabeled
import berkeleyentity.sem.BasicWordNetSemClasser
import berkeleyentity.sem.QueryCountsBundle
import berkeleyentity.sem.SemClasser
import berkeleyentity.wiki._
import edu.berkeley.nlp.futile.fig.basic.IOUtils
import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.futile.util.Logger
import berkeleyentity.xdistrib.CorefComputerDistrib
import berkeleyentity.xdistrib.ComponentFeaturizer
import berkeleyentity.xdistrib.DocumentGraphComponents
import edu.berkeley.nlp.futile.fig.exec.Execution
import berkeleyentity.Driver
import berkeleyentity.GUtil
import berkeleyentity.ConllDoc
import berkeleyentity.WordNetInterfacer
import berkeleyentity.ConllDocWriter
import berkeleyentity.ConllDocReader
import berkeleyentity.sem.BrownClusterInterface
import berkeleyentity.ner.NerPrunerFromMarginals
import berkeleyentity.ner.NerPruner
import berkeleyentity.coref._
import berkeleyentity.joint.JointPredictor
import java.util.ArrayList
import java.io.PrintWriter
import java.nio.file.Files
import java.nio.file.CopyOption._
import java.io.File
import scala.util.control.Breaks._
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
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
import berkeleyentity.coref.CorefEvaluator
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
import berkeleyentity.ConllDocReader
import berkeleyentity.Chunk
import berkeleyentity.ner.MCNerExample
import berkeleyentity.Driver
import berkeleyentity.wiki.ACETester
import berkeleyentity.wiki.WikipediaAuxDB
import berkeleyentity.wiki.WikipediaCategoryDB
import berkeleyentity.wiki.WikipediaInterface
import berkeleyentity.wiki.WikipediaLinkDB
import berkeleyentity.wiki.WikipediaRedirectsDB
import berkeleyentity.wiki.WikipediaTitleGivenSurfaceDB
import berkeleyentity.wiki.JointQueryDenotationChoiceComputer
import berkeleyentity.wiki.JointQueryDenotationChooser
import berkeleyentity.wiki.JointQueryDenotationExample
import berkeleyentity.wiki.Query
import berkeleyentity.wiki.WikiAnnotReaderWriter
import berkeleyentity.wiki.WikificationEvaluator
import berkeleyentity.wiki.CorpusWikiAnnots
import berkeleyentity.wiki.DocWikiAnnots
import berkeleyentity.Driver
import berkeleyentity.lang.Language
import edu.berkeley.nlp.futile.LightRunner
import berkeleyentity.coref.CorefDocAssembler
import berkeleyentity.ConllDocReader
import berkeleyentity.coref.MentionPropertyComputer
import berkeleyentity.GUtil
import edu.berkeley.nlp.futile.fig.basic.Indexer
import berkeleyentity.joint.LikelihoodAndGradientComputer
import scala.collection.mutable.ArrayBuffer
import berkeleyentity.coref.CorefDoc
import edu.berkeley.nlp.futile.math.SloppyMath
import edu.berkeley.nlp.futile.util.Logger
import berkeleyentity.Chunk
import berkeleyentity.joint.GeneralTrainer
import berkeleyentity.coref.NumberGenderComputer
import berkeleyentity.coref.Mention
import berkeleyentity.ilp.SingleDecision
import berkeleyentity.ilp.HistgramRecord
import java.util.Arrays
import edu.berkeley.nlp.futile.util.Counter
import scala.collection.JavaConverters._
import berkeleyentity.ner.MCNerFeaturizerBase



object SingleTaskStructTesting {
  
  // quick test!
  def quickTest(testExs: ArrayBuffer[AceMultiTaskExample],
                w: Array[Double]) {
    var sumTotal : Double = 0;
    var sumErr: Double = 0
    var sumErr1: Double = 0
    var sumErr2: Double = 0
    var sumErr3: Double = 0
    
	  for (ex <- testExs) {
		  val predBestOutput = ex.infereceIndepBest(w); // my prediction
		  val err = ex.getZeroOneError(predBestOutput);
      val (err1, err2, err3) = ex.getZeroOneErrorEachTask(predBestOutput, 0);
		  val total = ex.totalSize;
      sumErr += err;
      sumTotal += total;
      
      sumErr1 += err1;
      sumErr2 += err2;
      sumErr3 += err3;
	  }
    
    //val errRate = sumErr / sumTotal;
    val eachSum = sumTotal / 3;
    val crct = sumTotal - sumErr;
    val acc = crct / sumTotal;
    println("Error each task = [" + sumErr1 + "," + sumErr2 + "," + sumErr3 +  "] / " + eachSum );
    println("quick test: 01-Acc = " + crct + "/" + sumTotal + " = " + acc);
    //println("quick test: 01-Err0r = " + sumErr + "/" + sumTotal + " = " + errRate);
  }
  
  def addjustGoldwithPredictOutput(ex: AceMultiTaskExample, predBestOutput: Array[Int], goldBestOutput: Array[Int]) = {
    val adjusted = new Array[Int](goldBestOutput.length);
    Array.copy(goldBestOutput, 0, adjusted, 0, goldBestOutput.length); // copy!
    for (i <- 0 until goldBestOutput.length) {
      val vari = ex.getVariableGivenIndex(i);
      if (vari.getCorrectValueIndices().length == 0) { // no correct value
        adjusted(i) = predBestOutput(i);
        println("Adjust gold label at "+i+" to " + predBestOutput(i));
      }
    }
    adjusted;
  }
  
  
  
  //////////////////////////////////////////////////////////////////////////////////////////
/*
  def extractInstancesNer(exs: ArrayBuffer[tmpExample], featIndexer: Indexer[String]) = {

	  val nerIdxer = MCNerFeaturizer.StdLabelIndexer;
	  val labelsMapping = new java.util.HashMap[String, Integer]();
	  for (idx <- 0 until nerIdxer.size) {
		  labelsMapping.put(nerIdxer.getObject(idx), new Integer(idx));
	  }
	  val res = new LabeledMultiClassData(labelsMapping, featIndexer.size);

	  for (ex <- exs) {
		  val mi = new NerClassInstance(ex.feats, featIndexer.size, nerIdxer.size, ex.getGoldIdx);
		  res.instanceList.add(mi);
		  res.goldStructureList.add(new MultiClassLabel(mi.goldLabelIdx));
	  }

	  res;
  }
  */
  
  // Small instance learning (no structure at all)
  def independentInstanceLearning(allTrains: ArrayBuffer[AceMultiTaskExample], 
                           featIndexer: Indexer[String],
                           testExs: ArrayBuffer[AceMultiTaskExample]): Array[Double] = {
    
      var weight = Array.fill[Double](featIndexer.size)(0);
      var weightSum = Array.fill[Double](featIndexer.size)(0);
      var lastWeight = Array.fill[Double](featIndexer.size)(0);

      val Iteration = 100; ////
      val learnRate = 0.1;
      val lambda = 1e-8;

      var updateCnt = 0;
      var lastUpdtCnt = 0;

      for (iter <- 0 until Iteration) {
        lastUpdtCnt = updateCnt;
        Array.copy(weight, 0, lastWeight, 0, weight.length);

        println("Iter " + iter);
        for (example <- allTrains) {

        	for (idx <- 0 until example.totalSize) {

        		val curVariable = example.getVariableGivenIndex(idx);
            val varValues = curVariable.values;

        		val predBestOutput = curVariable.getBestValue(weight); // my prediction
        		val goldBestOutput = curVariable.getCorrectBestValue(weight);  // gold best

        		// Do not update for the variable that have no correct value!

        		// update?
        		if (!varValues(predBestOutput).isCorrect) {
        			updateCnt += 1;
        			if (updateCnt % 1000 == 0) println("Update " + updateCnt);

        			val featGold = varValues(goldBestOutput).feature;
        			val featPred = varValues(predBestOutput).feature;

        			NerStructUtils.updateWeight(weight, featGold, featPred, learnRate, lambda);
        			sumWeight(weightSum, weight);
        		}
        	}

        }

        ///////////////////////////////////////////////////
        // have a test after each iteration (for learning curve)
        val tmpAvg = new Array[Double](weightSum.size)
        Array.copy(weightSum, 0, tmpAvg, 0, weightSum.size);
        divdeNumber(tmpAvg, updateCnt.toDouble);

        quickTest(allTrains, tmpAvg);
        quickTest(testExs, tmpAvg);
        println("Iter Update Cnt = " + (updateCnt - lastUpdtCnt));
      }

      divdeNumber(weightSum, updateCnt.toDouble);
      weightSum;    
  }

  
  //////////////////////////////////////////////////////////////////////////////////////////
  
  // structural learning
	def structurePerceptrion(allTrains: ArrayBuffer[AceMultiTaskExample], 
			                     featIndexer: Indexer[String],
			                     testExs: ArrayBuffer[AceMultiTaskExample]): Array[Double] = {

			var weight = Array.fill[Double](featIndexer.size)(0);
			var weightSum = Array.fill[Double](featIndexer.size)(0);
			var lastWeight = Array.fill[Double](featIndexer.size)(0);

			val Iteration = 100; ////
			val learnRate = 0.1;
			val lambda = 1e-8;

			var updateCnt = 0;
			var lastUpdtCnt = 0;

			/*
      var exId2 = 0;
      for (extst <- testExs) {
        exId2 += 1;
        val domains = constructWikiExampleDomains(extst, wikiDB);
        for (l <- 0 until domains.size) {
          logger2.println(domains(l).getFeatStr(exId2));
        }
      }*/

			for (iter <- 0 until Iteration) {
				lastUpdtCnt = updateCnt;
				Array.copy(weight, 0, lastWeight, 0, weight.length);

				println("Iter " + iter);
				var exId = 0;
				for (example <- allTrains) {

					exId += 1;

          //val predBestOutput = example.infereceIndepBest(weight); // my prediction
          val predBestOrigin = example.infereceIndepBest(weight); // my prediction
					val goldBestOutput = example.infereceIndepGoldBest(weight);  // gold best

          // Do not update for the variable that have no correct value!
          val predBestOutput = predBestOrigin;//addjustGoldwithPredictOutput(example, predBestOrigin, goldBestOutput);
          
          //println("Pred = " + predBestOutput);
          //println("Gold = " + goldBestOutput);
          
					// update?
					if (!example.isCorrectOutput(predBestOutput)) {
						updateCnt += 1;
						if (updateCnt % 1000 == 0) println("Update " + updateCnt);
            
            val featGold = example.featurize(goldBestOutput);
            val featPred = example.featurize(predBestOutput);
            
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
				println("Iter Update Cnt = " + (updateCnt - lastUpdtCnt));

				//val wdiff = checkWeight(weight, lastWeight);
				//println("Weight diff = " + wdiff);
			}

			divdeNumber(weightSum, updateCnt.toDouble);

			/*
      for (i <- 0 until weightSum.length) {
        if (weightSum(i) != 0) {
          println("weight(" + i + ") = " + weightSum(i));
        }
      }
			 */

			//logger.close();
			//logger2.close();
			weightSum;
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
  
  
  ////////////////////////////////////////////////////////////
  // Berkeley AdaGradient
  def trainAdagrad(trainExs: Seq[AceMultiTaskExample],
                   testExs: Seq[AceMultiTaskExample],
                   numFeats: Int,
                   eta: Double,
                   lambda: Double,
                   batchSize: Int,
                   numItrs: Int): Array[Double] = {
    val weights = Array.fill(numFeats)(0.0);
    val reusableGradientArray = Array.fill(numFeats)(0.0);
    val diagGt = Array.fill(numFeats)(0.0);
    for (i <- 0 until numItrs) {
      Logger.logss("ITERATION " + i);
      val startTime = System.nanoTime();
      Logger.startTrack("Computing gradient");
      var currIdx = 0;
      var currBatchIdx = 0;
      val printFreq = (trainExs.size / batchSize) / 10 // Print progress 10 times per pass through the data
      while (currIdx < trainExs.size) {
        if (printFreq == 0 || currBatchIdx % printFreq == 0) {
          Logger.logs("Computing gradient on " + currIdx);
        }
        takeAdagradStepL1R(trainExs.slice(currIdx, Math.min(trainExs.size, currIdx + batchSize)),
                           weights,
                           reusableGradientArray,
                           diagGt,
                           eta,
                           lambda);
        currIdx += batchSize;
        currBatchIdx += 1;
      }
      Logger.endTrack();
      
      quickTest((new ArrayBuffer[AceMultiTaskExample]()) ++ trainExs, weights);
      quickTest((new ArrayBuffer[AceMultiTaskExample]()) ++ testExs, weights);
      
      Logger.logss("NONZERO WEIGHTS: " + weights.foldRight(0)((weight, count) => if (Math.abs(weight) > 1e-15) count + 1 else count));
      Logger.logss("WEIGHT VECTOR NORM: " + weights.foldRight(0.0)((weight, norm) => norm + weight * weight));
      Logger.logss("MILLIS FOR ITER " + i + ": " + (System.nanoTime() - startTime) / 1000000.0);
    }
    weights;
  }
  
  def takeAdagradStepL1R(exs: Seq[AceMultiTaskExample],
                         weights: Array[Double],
                         reusableGradientArray: Array[Double],
                         diagGt: Array[Double],
                         eta: Double,
                         lambda: Double) {
    Arrays.fill(reusableGradientArray, 0.0);
    var nanoTime = System.nanoTime();
    for (ex <- exs) {
      //computer.addUnregularizedStochasticGradient(ex, weights, reusableGradientArray);
      val predBestOutput = ex.infereceIndepBest(weights); // my prediction
      val goldBestOutput = ex.infereceIndepGoldBest(weights);  // gold best
      val featGold = ex.featurize(goldBestOutput);
      val featPred = ex.featurize(predBestOutput);

      //var gradient = Array.fill[Double](currentWeight.length)(0);//new Array[Double](currentWeight.length);
      for ((j, vpred) <- featPred) {
        reusableGradientArray(j) -= (vpred);
      }
      for ((i, vgold) <- featGold) {
        reusableGradientArray(i) += (vgold);
      }
    }
    nanoTime = System.nanoTime();
    // Precompute this so dividing by batch size is a multiply and not a divide
    val batchSizeMultiplier = 1.0F/exs.size;
    var i = 0;
    while (i < reusableGradientArray.size) {
      val xti = weights(i);
      // N.B. We negate the gradient here because the Adagrad formulas are all for minimizing
      // and we're trying to maximize, so think of it as minimizing the negative of the objective
      // which has the opposite gradient
      // Equation (25) in http://www.cs.berkeley.edu/~jduchi/projects/DuchiHaSi10.pdf
      // eta is the step size, lambda is the regularization
      val gti = -reusableGradientArray(i) * batchSizeMultiplier;
      // Update diagGt
      diagGt(i) += gti * gti;
      val Htii = 1F + Math.sqrt(diagGt(i)).toFloat;
      // Avoid divisions at all costs...
      val etaOverHtii = eta / Htii;
      val newXti = xti - etaOverHtii * gti;
      weights(i) = Math.signum(newXti) * Math.max(0, Math.abs(newXti) - lambda * etaOverHtii);
      i += 1;
    }
  }
  /////////////////////////////////////////////////////////////


  // structural example extractor
  
  def extractAllTaskExamples(docGraphs: Seq[DocumentGraph], 
                             nerFeaturizer: MCNerFeaturizerBase,//MCNerFeaturizer,
                             goldWikification: CorpusWikiAnnots, wikiDB: WikipediaInterface, filterImpossible: Boolean = false, jqdcomputer: JointQueryDenotationChoiceComputer,
                             pairwiseIndexingFeaturizer: PairwiseIndexingFeaturizer) = {
    
    val result = ArrayBuffer[AceMultiTaskExample]();
    for (d <- docGraphs) {
      result += extractAllTaskOneExample(d, nerFeaturizer, goldWikification, wikiDB, filterImpossible, jqdcomputer, pairwiseIndexingFeaturizer);
    }
    result;
  }
  
  def extractAllTaskOneExample(docGraph: DocumentGraph, 
                                                  nerFeaturizer: MCNerFeaturizerBase,//MCNerFeaturizer,
                                                  goldWikification: CorpusWikiAnnots, wikiDB: WikipediaInterface, filterImpossible: Boolean, jqdcomputer: JointQueryDenotationChoiceComputer,
                                                  pairwiseIndexingFeaturizer: PairwiseIndexingFeaturizer) = {
    
    val corefDoc: CorefDoc = docGraph.corefDoc;
    val rawDoc = corefDoc.rawDoc;
    val docName = rawDoc.docID;
    val addToIdxer: Boolean = docGraph.addToFeaturizer;
    
    
    ////// Coref Coref Coref Coref Coref Coref Coref Coref Coref
    val docCorefVars = new ArrayBuffer[IndepVariable[Int]];
    
    // featurizing!
    docGraph.featurizeIndexNonPrunedUseCache(pairwiseIndexingFeaturizer); 
    
    for (i <- 0 until docGraph.size) {
      val ment = docGraph.getMention(i);
      val corefValArr = new ArrayBuffer[VarValue[Int]]();
      val corefGoldArr = new ArrayBuffer[VarValue[Int]]();

      var valueCnt = 0;
      
      val prunedDomain = docGraph.getPrunedDomain(i, false);
      val goldPrunedAntecedents = docGraph.getGoldAntecedentsUnderCurrentPruning(i);
      for (j <- prunedDomain) {
        val correct = goldPrunedAntecedents.contains(j);//docGraph.isGoldNoPruning(i, j);
        val anteValue = new VarValue[Int](valueCnt, j, docGraph.cachedFeats(i)(j), correct);
        valueCnt += 1;
          
        corefValArr += anteValue;
        if (correct) {
          corefGoldArr += anteValue;
        }
      }
      /*
      for (j <- 0 to i) {
        if (!docGraph.prunedEdges(i)(j)) {
          //require(featsChart(i)(j).size > 0);
          //featurizer.featurizeIndex(docGraph, i, j, false);
          //scoreChart(i)(j) = GUtil.scoreIndexedFeats(featsChart(i)(j), scorer.weights);
          val correct = docGraph.isGoldNoPruning(i, j);
          
          val anteValue = new VarValue[Int](valueCnt, j, docGraph.cachedFeats(i)(j), correct);
          valueCnt += 1;
          
          corefValArr += anteValue;
          if (correct) {
            corefGoldArr += anteValue;
          }
        } else {
          // was pruned
          //  scoreChart(i)(j) = Float.NegativeInfinity;
        }
      }
      */
      docCorefVars += (new IndepVariable[Int](corefValArr.toArray, corefGoldArr.toArray, corefValArr(0)));
    }
    
    val structCorefExmp = new AceSingleTaskStructExample[Int](docCorefVars.toArray);
    ////// End of Coref ============================================
    
    
    
    ////// NER NER NER NER NER NER NER NER NER NER NER NER
    val docNerVars = new ArrayBuffer[IndepVariable[String]];
    
    val nerLabelSet = MCNerFeaturizer.StdLabelIndexer;//MCNerFeaturizer.StdLabelIndexer;
    //for (corefDoc <- corefDocs) {
      
      for (i <- 0 until corefDoc.predMentions.size) {
        val pm = docGraph.getMention(i);//corefDoc.predMentions(i);
        val nerExmp = new MCNerExample(rawDoc.words(pm.sentIdx), rawDoc.pos(pm.sentIdx), rawDoc.trees(pm.sentIdx), pm.startIdx, pm.headIdx, pm.endIdx, pm.nerString);
        nerExmp.ment = pm;
        val featEachLabel = nerFeaturizer.featurize(nerExmp, addToIdxer);
        
        val goldTag = NerStructUtils.getGoldNerTag(nerExmp.goldLabel);
        val goldLabelIdx = nerLabelSet.indexOf(goldTag);
        
        // constructing variables ///////////////////
        val nerVals = new ArrayBuffer[VarValue[String]]();
        val correctVals = new ArrayBuffer[VarValue[String]]();
        //val currentVal = 

        for (l <- 0 until nerLabelSet.size) {
          val lstr = nerLabelSet.getObject(l);
          val correct = (l == goldLabelIdx);
          
          val myNerValue = new VarValue[String](l, lstr, featEachLabel(l), correct);
          nerVals += myNerValue;
          if (correct) {
            correctVals += myNerValue;
          }
        }
        
        val nerVar = new IndepVariable[String](nerVals.toArray, correctVals.toArray, nerVals(0));
        /////////////////////////////////////////////
        
        docNerVars += nerVar;
      }
    //}
    //Logger.logss(docNerVars.size + " ner chunks");
    
    // construct structure example
    val structNerExmp = new AceSingleTaskStructExample[String](docNerVars.toArray);
    ////// End of NER ============================================
    
    
    
    ////// Wiki Wiki Wiki Wiki Wiki Wiki Wiki Wiki Wiki Wiki
    val docQuerWikiVars = new ArrayBuffer[IndepVariable[QueryWikiValue]];
    //val writer = new PrintWriter("wiki_men.log");
    //val exs = new ArrayBuffer[JointQueryDenotationExample];
    var numImpossible = 0;
    
      val docGoldWikis = goldWikification(docName);
      if (docGoldWikis.isEmpty) {
        throw new RuntimeException("Emtpy label for doc "+docName+"!!!!!!!!");
      }
      for (i <- 0 until corefDoc.predMentions.size) {
        val ment = corefDoc.predMentions(i);
        
        
        /*
        // Discard "closed class" mentions (pronouns) since these don't have interesting entity links
        val jex = if (!ment.mentionType.isClosedClass()) {
          // There are multiple possible gold Wikipedia titles for some mentions. Note that
          // NIL (no entry in Wikipedia) is included as an explicit choice, so this includes NILs (as
          // it should according to how the task is defined)
          //val goldLabel = getGoldWikification(goldWikification(docName), ment)
          val goldLabel = getGoldWikLabels(docGoldWikis, ment);
          if (goldLabel.size >= 1) {
            val queries = Query.extractQueriesBest(ment, true);
            val queryDisambigs = queries.map(wikiDB.disambiguateBestGetAllOptions(_));
            val rawQueryDenotations = Query.extractDenotationSetWithNil(queries, queryDisambigs, WikiStructUtils.maxNumWikificationOptions);
            //val correctDenotations = rawQueryDenotations.filter(denotation => isCorrect(goldLabel, denotation))
            // N.B. The use of "isCorrect" here is needed to canonicalize 
            val correctIndices = rawQueryDenotations.zipWithIndex.filter(denotationIdx => isCorrect(goldLabel, denotationIdx._1)).map(_._2);
            val correctDenotations = rawQueryDenotations.filter(den => isCorrect(goldLabel, den));

            if (filterImpossible && correctIndices.isEmpty) {
              numImpossible += 1;
              println("Impossible Ment: "+ ment.spanToString);
              getAnEmptyJointQueryDenotationExample(ment);
              //new JointQueryDenotationExample(queries, denotations, correctDenotations, goldLabel);
            } else {
              val jqe = new JointQueryDenotationExample(queries, rawQueryDenotations, correctDenotations, goldLabel);
              jqdcomputer.featurizeUseCache(jqe, addToIdxer);
              jqe;
            }
          } else { // NIL only for unlabeled mentions
            //throw new RuntimeException("Unlabeled wiki example!");
            println("Unlabeled gold wiki example!");
            getAnEmptyJointQueryDenotationExample(ment);
          }
        } else { // NIL only for closed type mentions
          getAnEmptyJointQueryDenotationExample(ment);
        }
        */


        val jex = if (!ment.mentionType.isClosedClass()) {
        	// There are multiple possible gold Wikipedia titles for some mentions. Note that
        	// NIL (no entry in Wikipedia) is included as an explicit choice, so this includes NILs (as
        	// it should according to how the task is defined)
        	val goldLabel = getGoldWikification(goldWikification(docName), ment)

          val queries = Query.extractQueriesBest(ment, true);
        	val queryDisambigs = queries.map(wikiDB.disambiguateBestGetAllOptions(_));

        	val denotationsRaw = Query.extractDenotationSetWithNil(queries, queryDisambigs, WikiStructUtils.maxNumWikificationOptions);
        	var correctDenotations = denotationsRaw.filter(den => isCorrect(goldLabel, den));
        	if (correctDenotations.isEmpty) correctDenotations = denotationsRaw;
          
        	val jqe = new JointQueryDenotationExample(queries, denotationsRaw, correctDenotations, goldLabel);
          jqdcomputer.featurizeUseCache(jqe, addToIdxer);
          jqe;
          
        } else {
          getAnEmptyJointQueryDenotationExample(ment);
        }

/*   
        // gold wiki test ... 2016-10-15
        val jex = if (!ment.mentionType.isClosedClass()) {
        	val goldLabel = getGoldWikification(goldWikification(docName), ment)

          val queries = Query.extractQueriesBest(ment, true);
        	val queryDisambigs = queries.map(wikiDB.disambiguateBestGetAllOptions(_));

        	val denotationsRaw = Query.extractDenotationSetWithNil(queries, queryDisambigs, WikiStructUtils.maxNumWikificationOptions);
        	var correctDenotations = denotationsRaw.filter(den => isCorrect(goldLabel, den));
        	if (correctDenotations.isEmpty) correctDenotations = denotationsRaw;
          
        	val jqe = new JointQueryDenotationExample(queries, correctDenotations, correctDenotations, goldLabel);
          jqdcomputer.featurizeUseCache(jqe, addToIdxer);
          jqe;
          
        } else {
          getAnEmptyJointQueryDenotationExample(ment);
        }
*/
        // construct wiki variable
        val qwVarVals = new ArrayBuffer[VarValue[QueryWikiValue]]();
        val qwVarValsCorrect = new ArrayBuffer[VarValue[QueryWikiValue]]();
        
        val qwVals = constructWikiVariableDomains(jex, wikiDB);
        //println("wiki variable domain size = " + qwVals.size);
        for (qwidx <- 0 until qwVals.size) {
          val qwv = qwVals(qwidx);
          val thisQwv = (new VarValue[QueryWikiValue](qwidx, qwv, qwv.concatenateFeats, qwv.isCorrect));
          qwVarVals += thisQwv;
          if (thisQwv.isCorrect) {
            qwVarValsCorrect += thisQwv;
          }
        }
        
        docQuerWikiVars += (new IndepVariable[QueryWikiValue](qwVarVals.toArray, qwVarValsCorrect.toArray, qwVarVals(0)));
      }
    //}
    //Logger.logss(docQuerWikiVars.size + " possible, " + numImpossible + " impossible");
    
    val structWikiExmp = new AceSingleTaskStructExample[QueryWikiValue](docQuerWikiVars.toArray);
    //writer.close();
    //exs;
    
    ////// End of Wiki ============================================
    
    
    

    
    // final construction!
    val alltaskStrucEx = new AceMultiTaskExample(structCorefExmp, structNerExmp, structWikiExmp, docGraph);
    alltaskStrucEx;
  }
  
  def getGoldWikLabels(goldWikiChunks: DocWikiAnnots, ment: Mention): Seq[String] = {
    val matchingChunk = goldWikiChunks(ment.sentIdx).filter(chunk => chunk.start == ment.startIdx && chunk.end == ment.endIdx);
    //if (matchingChunk.size > 0) matchingChunk.head.label else Seq(ExcludeToken);
    if (matchingChunk.size == 1) matchingChunk.head.label else Seq(ExcludeToken);
  }
  
  def getAnEmptyJointQueryDenotationExample(ment: Mention) = {
	  val nilQueryDomain = Query.makeNilQuery(ment);
	  val excludeWikiDomain = ExcludeToken;
	  val goldWikiLabel = ExcludeToken;
	  val jqe = new JointQueryDenotationExample(Seq(nilQueryDomain), Seq(excludeWikiDomain), Seq(goldWikiLabel), Seq(goldWikiLabel));
    jqe.cachedFeatsEachQuery = Array(Array());
    jqe.cachedFeatsEachQueryDenotation = Array(Array(Array()));
    jqe;
  }
  

  
  
  def constructWikiVariableDomains(ex: JointQueryDenotationExample,
                                   wikiDB: WikipediaInterface): ArrayBuffer[QueryWikiValue] = {
      val qwvals = new ArrayBuffer[QueryWikiValue]();
      val queries = ex.queries;
      val denotations = ex.allDenotations;
      val queryOutcomes = queries.map(query => wikiDB.disambiguateBestGetAllOptions(query));
      
      //val correctDenoSet = (new HashSet[String]()) ++ ();
      
      for (qidx <- 0 until queries.size) {
        val qvals = new ArrayBuffer[QueryWikiValue]();
        var containCrr = false;
        for (didx <- 0 until denotations.size) {
          val q = queries(qidx);
          val d = denotations(didx);
          if ((d.equals(ExcludeToken)) || (d.equals(NilToken)) || (queryOutcomes(qidx).containsKey(d))) {
            val fq = ex.cachedFeatsEachQuery(qidx);
            val fqd = ex.cachedFeatsEachQueryDenotation(qidx)(didx);
            //println(fq.length + ", " + fqd.length)
            //val isCorct = isCorrect(ex.correctDenotations, d);//ex.correctDenotationIndices.contains(didx);
            val isCorct = ex.correctDenotationIndices.contains(didx);
            if (isCorct) {
              containCrr = true;
            }
            //qwvals += (new QueryWikiValue(q, d, fq, fqd, isCorct));
            qvals += (new QueryWikiValue(q, d, fq, fqd, isCorct));
          }
        }
        if (containCrr)
        qwvals ++= qvals;
      }
      
      qwvals
  }
  
  def main(args: Array[String]) {

    // set some configs
    Driver.numberGenderDataPath = "data/gender.data";
    Driver.brownPath = "data/bllip-clusters";
    //Driver.wordNetPath = "data/dict";
    Driver.useGoldMentions = true;
    Driver.doConllPostprocessing = false;
    Driver.pruningStrategy = "build:./corefpruner-ace.ser.gz:-5:5";
    Driver.lossFcn = "customLoss-1-1-1";

    
    Driver.corefNerFeatures = "indicators+currlex+antlex";
    Driver.wikiNerFeatures = "categories+infoboxes+appositives";
    Driver.corefWikiFeatures = "basic+lastnames";
    
    trainStructuralAllTasks();
  }
  
  def trainStructuralAllTasks() {
    
    val trainDataPath = "data/ace05/train";
    val devDataPath = "data/ace05/dev";
    val testDataPath = "data/ace05/test";
    val wikiPath = "data/ace05/ace05-all-conll-wiki"
    val wikiDBPath = "models/wiki-db-ace.ser.gz"
    
    
    val featIndexer = new Indexer[String]();
    
    
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    val assembler = CorefDocAssembler(Language.ENGLISH, true); //use gold mentions
    val trainDocs = ConllDocReader.loadRawConllDocsWithSuffix(trainDataPath, -1, "", Language.ENGLISH);
    val trainCorefDocs = trainDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    
    val maybeBrownClusters = if (Driver.brownPath != "") Some(BrownClusterInterface.loadBrownClusters(Driver.brownPath, 0)) else None;
    val nerFeaturizer = MCNerFeaturizer(Driver.nerFeatureSet.split("\\+").toSet, featIndexer, MCNerFeaturizer.StdLabelIndexer, trainDocs.flatMap(_.words), None, maybeBrownClusters);
    
    
    // Read in gold Wikification labels
    val goldWikification = WikiAnnotReaderWriter.readStandoffAnnotsAsCorpusAnnots(wikiPath)
    // Read in the title given surface database
    val wikiDB = GUtil.load(wikiDBPath).asInstanceOf[WikipediaInterface];
    
    val jqdcomputer = new JointQueryDenotationChoiceComputer(wikiDB, featIndexer);
    //for (trainEx <- trainExs) {
    //  jqdcomputer.featurizeUseCache(trainEx, true);
    //}

    
    val queryCounts: Option[QueryCountsBundle] = None;
    val lexicalCounts = LexicalCountsBundle.countLexicalItems(trainCorefDocs, Driver.lexicalFeatCutoff);
    val semClasser: Option[SemClasser] = Driver.semClasserType match {
      case "basic" => Some(new BasicWordNetSemClasser);
      case e => throw new RuntimeException("Other semclassers not implemented");
    }
    val trainDocGraphs = trainCorefDocs.map(new DocumentGraph(_, true));

    CorefStructUtils.preprocessDocsCacheResources(trainDocGraphs);
    //CorefPruner.buildPruner(Driver.pruningStrategy).pruneAll(trainDocGraphs);

    featIndexer.getIndex(PairwiseIndexingFeaturizerJoint.UnkFeatName);
    val featureSetSpec = FeatureSetSpecification(Driver.pairwiseFeats, Driver.conjScheme, Driver.conjFeats, Driver.conjMentionTypes, Driver.conjTemplates);
    val basicFeaturizer = new PairwiseIndexingFeaturizerJoint(featIndexer, featureSetSpec, lexicalCounts, queryCounts, semClasser);
    val featurizerTrainer = new CorefFeaturizerTrainer();
    //featurizerTrainer.featurizeBasic(trainDocGraphs, basicFeaturizer);

    // joint featurizer
    val jointFeaturier = new JointFeaturizerShared[MCNerFeaturizer](basicFeaturizer, nerFeaturizer, maybeBrownClusters, Driver.corefNerFeatures, Driver.corefWikiFeatures, Driver.wikiNerFeatures, featIndexer);
    
    
    // extract structural examples!
    val trainStructs = extractAllTaskExamples(trainDocGraphs, 
                                              nerFeaturizer,
                                              goldWikification, wikiDB, true, jqdcomputer,
                                              basicFeaturizer);
    
    PairwiseIndexingFeaturizer.printFeatureTemplateCounts(featIndexer);

    
    // test examples
    val testDocGraphs = testGetdocgraphs(testDataPath, -1, mentionPropertyComputer);
    val testStructs = extractAllTaskExamples(testDocGraphs, 
                                             nerFeaturizer,
                                             goldWikification, wikiDB, false, jqdcomputer,
                                             basicFeaturizer);
    
    // Learning!
    //val weight = structurePerceptrion(trainStructs, featIndexer, testStructs);
    //val weight = independentInstanceLearning(trainStructs, featIndexer, testStructs);
    val weight = SingleTaskSVMLearning.NerSingleStructLearning(trainStructs, featIndexer, testStructs);
    //val weight = SingleTaskSVMLearning.CorefSingleStructLearning(trainStructs, featIndexer, testStructs);
    
    
    // test as oracle?
    //testStructuraAllTaskOracle(testStructs, featIndexer.size());
    
    // test!
    val histgram = new HistgramRecord();
    testStructuralAllTasks(trainStructs, weight, histgram);
    testStructuralAllTasks(testStructs, weight, histgram);
    histgram.printHistgram();
    
    // evaluate
    evaluateAceStructs(trainStructs, goldWikification);
    evaluateAceStructs(testStructs, goldWikification);
  }
  
  
  def testStructuralAllTasks(testStructs: Seq[AceMultiTaskExample], wght: Array[Double], histg: HistgramRecord) {
      for (ex <- testStructs) {
        ex.corefOutput.currentOutput = ex.corefOutput.infereceIndepBest(wght);
        ex.nerOutput.currentOutput = ex.nerOutput.infereceIndepBest(wght);
        ex.wikiOutput.currentOutput = ex.wikiOutput.infereceIndepBest(wght);
        computeErrorHistgram(ex, wght, histg);
      }
  }
  
  def testGetdocgraphs(devPath: String, devSize: Int, mentionPropertyComputer: MentionPropertyComputer) = {
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val devDocs = CorefStructUtils.loadCorefDocs(devPath, devSize, Driver.docSuffix, mentionPropertyComputer);
    val devDocGraphs = devDocs.map(new DocumentGraph(_, false));
    CorefStructUtils.preprocessDocsCacheResources(devDocGraphs);
    //CorefPruner.buildPruner(Driver.pruningStrategy).pruneAll(devDocGraphs);
    devDocGraphs;
  }
  
  
  
  def testStructuraAllTaskOracle(testStructs: Seq[AceMultiTaskExample], wlen: Int) {
      val wght = Array.fill[Double](wlen)(0);
      for (ex <- testStructs) {
        ex.corefOutput.currentOutput = ex.corefOutput.infereceIndepGoldBest(wght);
        ex.nerOutput.currentOutput = ex.nerOutput.infereceIndepGoldBest(wght);
        ex.wikiOutput.currentOutput = ex.wikiOutput.infereceIndepGoldBest(wght);
      }
  }
  
  def computeCorefBackPointer(struct: AceMultiTaskExample, coutput: Array[Int]): Array[Int] = {
    val corefBackPointer = new Array[Int](coutput.size);
    val corefStruct = struct.corefOutput;
    for (i <- 0 until coutput.size) {
      val j = coutput(i);
      corefBackPointer(i) = corefStruct.variables(i).values(j).value;
    }
    corefBackPointer
  }
  
  def evaluateAceStructs(testStructs: Seq[AceMultiTaskExample], 
                         goldWikification: HashMap[String, DocWikiAnnots]) {

    // for coref
    val allPredBackptrs = new Array[Array[Int]](testStructs.size);
    val docseq = new ArrayBuffer[DocumentGraph]();

    // for wiki
    val goldTestDenotationsAsTrivialChunks = new ArrayBuffer[Chunk[Seq[String]]]();
    val predTestDenotationsAsTrivialChunks = new ArrayBuffer[Chunk[String]]();
    
    // for ner
    val neLabelIndexer = MCNerFeaturizer.StdLabelIndexer;
    var nerCorrCnt: Double = 0;
    var nerTotalCnt: Double = 0;
    val goldNerChunks = new ArrayBuffer[Chunk[String]]();
    val predNerChunks = new ArrayBuffer[Chunk[String]]();
    
    // wiki count
    var wikiCnt = 0;
    var noGold = 0;
    
	  for (i <- 0 until testStructs.size) {
      
      val ex = testStructs(i);
      val gdwiki = goldWikification(ex.docGraph.corefDoc.rawDoc.docID);

		  //// Coref
		  val corefStruct = ex.corefOutput;
      //val coutput = corefStruct.currentOutput;
      allPredBackptrs(i) = computeCorefBackPointer(ex, corefStruct.currentOutput);//coutput;
      docseq += ex.docGraph;


		  //// Ner
		  val nerStruct = ex.nerOutput;
		  for (j <- 0 until nerStruct.variables.size) {
			  val nerv = nerStruct.variables(j);
			  nerTotalCnt += 1;
			  if (nerv.values(nerStruct.currentOutput(j)).isCorrect) {
				  nerCorrCnt += 1;
			  }
        
        var correctj = -1;
        for (j2 <- 0 until nerv.values.length) {
          if (nerv.values(j2).isCorrect) {
            correctj = j2;
          }
        }

        val neStart = i * 5000 + j;
        val neEnd = i * 5000 + j + 1;
        val pnechunk = new Chunk[String](neStart, neEnd, nerv.values(nerStruct.currentOutput(j)).value);
        val gnechunk = new Chunk[String](neStart, neEnd, nerv.values(correctj).value);
        goldNerChunks += gnechunk;
        predNerChunks += pnechunk;
		  }

		  //// Wiki
		  val wikiStruct = ex.wikiOutput;
      val docWikiChunks = new ArrayBuffer[Chunk[String]]();
      val goldDocChunks = new ArrayBuffer[Chunk[Seq[String]]]();
      val predDocChunks = new ArrayBuffer[Chunk[String]]();
		  for (k <- 0 until wikiStruct.variables.size) {
			  val qwv = wikiStruct.variables(k);
			  //var correctVal = getGoldWikification(gdwiki, ex.docGraph.corefDoc.predMentions(k));//qwv.correcValues.map { corrv => { corrv.value.wiki } };
        var correctVal = getGoldWikificationUnique(gdwiki, ex.docGraph.corefDoc.predMentions(k));

        //if (correctVal.size == 0) {
        //  correctVal = Array(UknGoldLabelToken);
        //}
        val corrSet = (new HashSet[String]()) ++ correctVal;
			  if (!(corrSet.contains(ExcludeToken))) {
          wikiCnt += 1;
				  val trivialStart = i * 5000 + k;
				  val trivialEnd = i * 5000 + k + 1;
				  val gchunk = new Chunk[Seq[String]](trivialStart, trivialEnd, correctVal);

				  var predIdx = wikiStruct.currentOutput(k);
          if (predIdx < 0) {
             noGold += 1;
          }
          if (wikiStruct.currentOutput(k) < 0) predIdx = 0;

					val predVal = qwv.values(predIdx);
				  val pchunk = new Chunk[String](trivialStart, trivialEnd, (predVal.value.wiki));

				  //println(gchunk);
				  //println(pchunk);

				  goldTestDenotationsAsTrivialChunks += gchunk;
				  predTestDenotationsAsTrivialChunks += pchunk;
          
          goldDocChunks += gchunk;
          predDocChunks += pchunk;
          
          //docWikiChunks += new Chunk[String](ex.docGraph.corefDoc.predMentions(k).startIdx, ex.docGraph.corefDoc.predMentions(k).endIdx, (predVal.value.wiki));
			  } else {
          //docWikiChunks += new Chunk[String](ex.docGraph.corefDoc.predMentions(k).startIdx, ex.docGraph.corefDoc.predMentions(k).endIdx, ExcludeToken);
        }
        
		  }
      
      //println("DocName: " + ex.docGraph.corefDoc.rawDoc.docID);
      //for (chnk <- gdwiki.values.toSeq.flatten) {
      ///  println(chnk.start + ", " + chnk.end + ", " + chnk.label);
      //}
      //WikificationEvaluator.evaluateFahrniMetrics(Seq(goldDocChunks), Seq(predDocChunks), Set());

	  }
    
    
    println("------- COREF:");
    val allPredClusteringsSeq = (0 until allPredBackptrs.length).map(i => OrderedClustering.createFromBackpointers(allPredBackptrs(i)));
    val allPredClusteringsArr = allPredClusteringsSeq.toArray;
    val scoreOutput = CorefEvaluator.evaluateAndRender(docseq.toSeq, allPredBackptrs, allPredClusteringsArr, Driver.conllEvalScriptPath, "DEV: ", Driver.analysesToPrint);
    println(scoreOutput);
    
    
    println("------- NER:");
    val acc = nerCorrCnt / nerTotalCnt;
    println("Accuracy: " + nerCorrCnt + "/" + nerTotalCnt + " = " + acc);
    NEEvaluator.evaluateChunksBySent(Seq(goldNerChunks), Seq(predNerChunks));
    
    println("------- WIKI:");
    println("total count:" + wikiCnt);
    println("no-gold count:" + noGold);
    val pseq = predTestDenotationsAsTrivialChunks.toArray;
    val gseq = goldTestDenotationsAsTrivialChunks.toArray;
    WikificationEvaluator.evaluateFahrniMetrics(Seq(gseq), Seq(pseq), Set());

  }
  
  def nerErrorAnalysisJoint(testStructs: Seq[AceJointTaskExample], testDocs: Seq[CorefDoc]) {
    val multiStructs = testStructs.map( struct => { struct.toMultiTaskStructs() } );
    nerErrorAnalysis(multiStructs, testDocs);
  }
  
  def nerErrorAnalysis(testStructs: Seq[AceMultiTaskExample], testDocs: Seq[CorefDoc]) {

    val fileWriter = new PrintWriter("ner_err_joint.txt");
    
    val multiStructs = testStructs;// testStructs.map( struct => { struct.toMultiTaskStructs() } );
    
    val confuseCnt = new Counter[String]();
    val typCnt = new Counter[String]();
    
    
    val neLabelIndexer = MCNerFeaturizer.StdLabelIndexer;
    var nerCorrCnt: Double = 0;
    var nerTotalCnt: Double = 0;
    val goldNerChunks = new ArrayBuffer[Chunk[String]]();
    val predNerChunks = new ArrayBuffer[Chunk[String]]();
    
    // wiki count
    var wikiCnt = 0;
    var noGold = 0;
    
	  for (i <- 0 until multiStructs.size) {
	    val cdoc = testDocs(i);
	    val ex = multiStructs(i);
    	//// Ner
		  val nerStruct = ex.nerOutput;
		  for (j <- 0 until nerStruct.variables.size) {
			  // one mention
		    val ment = cdoc.predMentions(j);
		    val nerv = nerStruct.variables(j);
			  nerTotalCnt += 1;
			  val mypred = nerStruct.currentOutput(j);
			  if (nerv.values(nerStruct.currentOutput(j)).isCorrect) {
				  nerCorrCnt += 1;
			  }
        var correctj = -1;
        for (j2 <- 0 until nerv.values.length) {
          if (nerv.values(j2).isCorrect) {
            correctj = j2;
          }
        }
        
        if (mypred != correctj) { // an error
          println("["+ ment.spanToString + "] " + ment.mentionType.toString() + " " + NerTesting.getGoldMentTyp(ment.nerString) + " " + neLabelIndexer.getObject(mypred) + " shouldbe " + neLabelIndexer.getObject(correctj));
          val my = neLabelIndexer.getObject(mypred)
          var cr = neLabelIndexer.getObject(correctj)
          confuseCnt.incrementCount(my + "-" + cr, 1.0);
          typCnt.incrementCount(ment.mentionType.toString(), 1.0);

        }
		  }
	  }
    
    println("========================");
    val allPairs = confuseCnt.getEntrySet.asScala.toList.sortWith(_.getValue > _.getValue);
    for (pr <- allPairs) {
    	println(pr);
    }
    println("========================");
    val allTyps = typCnt.getEntrySet.asScala.toList.sortWith(_.getValue > _.getValue);
    for (tp <- allTyps) {
      println(tp);
    }
	  
  }
  
  //////////////////////
/* 
  def computeErrorHistgram(testEx: AceMultiTaskExample, histg: HistgramRecord) {
    val corefCrrctRank = Array.fill[Int](10000)(0);//new Array[Int](10000);
    val nerCrrctRank = Array.fill[Int](10000)(0);//new Array[Int](10000);
    val wikiCrrctRank = Array.fill[Int](10000)(0);//new Array[Int](10000);
    
    val doc = testEx.docGraph.corefDoc;
    val dg = testEx.docGraph;
    for (i <- 0 until dg.size) {
      val ment = dg.getMention(i)
      
      //val goldNer = doc.getGoldLabel(ment);
      //val goldWikAnnots = doc.getGoldWikLabels(ment);

      val cNode = factorGraph.corefNodes(i);
      val cval = factorGraph.corefNodes(i).domain;
      val cdecisions = new Array[SingleDecision](cval.size);
      val cscores = cNode.getMarginals();
      for (cj <- 0 until cval.size) {
        val crrct = dg.isGoldNoPruning(i, cval.value(cj));
        cdecisions(cj) = new SingleDecision(cscores(cj), crrct, cj);
      }
      
      ////////////////////////////////////
      
      val nNode = factorGraph.nerNodes(i);
      val nval = nNode.domain;
      val ndecisions = new Array[SingleDecision](nval.size);
      val nscores = nNode.getMarginals();
      for (nj <- 0 until nval.size) {
        val crrct = NerTesting.getGoldNerTag(goldNer).equals(nval.value(nj));
        //println(goldNer +"=="+ nval.value(nj) + " =  " + crrct)
        ndecisions(nj) = new SingleDecision(nscores(nj), crrct, nj);
      }
      
     ////////////////////////////////////
      
      val wNode = factorGraph.wikiNodes(i);
      val wval = wNode.domain;
      val wdecisions = new Array[SingleDecision](wval.size);
      val wscores = wNode.getMarginals();
      for (wj <- 0 until wval.size) {
        val crrct = isCorrect(goldWikAnnots, wval.value(wj));
        wdecisions(wj) = new SingleDecision(wscores(wj), crrct, wj);
      }
      
      // sort!
      val sortc = (cdecisions.toSeq.sortWith(_.score > _.score)).toArray;
      val sortn = (ndecisions.toSeq.sortWith(_.score > _.score)).toArray;
      val sortw = (wdecisions.toSeq.sortWith(_.score > _.score)).toArray;
      
      for (decs <- sortc) {
        println("c: " + decs.score + " " + decs.isCorrect);
      }
      for (decs <- sortn) {
        println("n: " + decs.score + " " + decs.isCorrect);
      }
      for (decs <- sortw) {
        println("w: " + decs.score + " " + decs.isCorrect);
      }
      
      histg.increaseForMention(sortc, sortn, sortw);
    }
  }
*/
  def computeErrorHistgram(testEx: AceMultiTaskExample, wght: Array[Double], histg: HistgramRecord) {
    val corefCrrctRank = Array.fill[Int](10000)(0);//new Array[Int](10000);
    val nerCrrctRank = Array.fill[Int](10000)(0);//new Array[Int](10000);
    val wikiCrrctRank = Array.fill[Int](10000)(0);//new Array[Int](10000);
    
    val doc = testEx.docGraph.corefDoc;
    val dg = testEx.docGraph;
    for (i <- 0 until dg.size) {
      val ment = dg.getMention(i)
      /*
      ex.corefOutput.currentOutput = ex.corefOutput.infereceIndepGoldBest(wght);
        ex.nerOutput.currentOutput = ex.nerOutput.infereceIndepGoldBest(wght);
        ex.wikiOutput*/
      
      val cval = testEx.corefOutput.variables(i).values;
      val cdecisions = new Array[SingleDecision](cval.size);
      for (cj <- 0 until cval.size) {
        //val crrct = dg.isGoldNoPruning(i, cval.value(cj));
        val cscore = cval(cj).computeScore(wght);
        cdecisions(cj) = new SingleDecision(cscore, cval(cj).isCorrect, cj);
      }
      
      ////////////////////////////////////
      
      val nval = testEx.nerOutput.variables(i).values;
      val ndecisions = new Array[SingleDecision](nval.size);
      for (nj <- 0 until nval.size) {
        //val crrct = NerTesting.getGoldNerTag(goldNer).equals(nval.value(nj));
        //println(goldNer +"=="+ nval.value(nj) + " =  " + crrct)
        //ndecisions(nj) = new SingleDecision(nscores(nj), crrct, nj);
        val nscore = nval(nj).computeScore(wght);
        ndecisions(nj) = new SingleDecision(nscore, nval(nj).isCorrect, nj);
      }
      
     ////////////////////////////////////
      
      val wval = testEx.wikiOutput.variables(i).values;
      val wdecisions = new Array[SingleDecision](wval.size);
      for (wj <- 0 until wval.size) {
        //val crrct = isCorrect(goldWikAnnots, wval.value(wj));
        //wdecisions(wj) = new SingleDecision(wscores(wj), crrct, wj);
        val wscore = wval(wj).computeScore(wght);
        wdecisions(wj) = new SingleDecision(wscore, wval(wj).isCorrect, wj);
      }
      
      // sort!
      val sortc = (cdecisions.toSeq.sortWith(_.score > _.score)).toArray;
      val sortn = (ndecisions.toSeq.sortWith(_.score > _.score)).toArray;
      val sortw = (wdecisions.toSeq.sortWith(_.score > _.score)).toArray;
      
      /*
      for (decs <- sortc) {
        println("c: " + decs.score + " " + decs.isCorrect);
      }
      for (decs <- sortn) {
        println("n: " + decs.score + " " + decs.isCorrect);
      }
      for (decs <- sortw) {
        println("w: " + decs.score + " " + decs.isCorrect);
      }
      */
      
      histg.increaseForMention(sortc, sortn, sortw);
    }
  }
  
  
  
  
  
  
  
  
  ////////////////////////////////////////////////
  /// Structural SVM learner /////////////////////
  ////////////////////////////////////////////////
  
}


object NerStructUtils {
  /*
  def learnFromSingleTaskStructs(trainExs: ArrayBuffer[AceMultiTaskExample], 
                                featIndexer: Indexer[String],
                                testExs: ArrayBuffer[AceMultiTaskExample]): Array[Double] = {
    
    
    
    
  }*/
  
  
  
  
  
  class tmpExample(val exmp: MCNerExample, 
                   val feats: Array[Array[Int]]) {
    
  }
 
  def extractExamples(corefDocs: Seq[CorefDoc]) = {
    val exs = new ArrayBuffer[MCNerExample];
    for (corefDoc <- corefDocs) {
      val rawDoc = corefDoc.rawDoc;
      val docName = rawDoc.docID
      for (i <- 0 until corefDoc.predMentions.size) {
        val pm = corefDoc.predMentions(i);
        val nerExmp = new MCNerExample(rawDoc.words(pm.sentIdx), rawDoc.pos(pm.sentIdx), rawDoc.trees(pm.sentIdx), pm.startIdx, pm.headIdx, pm.endIdx, pm.nerString);
        exs += nerExmp;
      }
    }
    Logger.logss(exs.size + " ner chunks");
    exs;
  }
 
  def NerTestingInterfaceACE() {
    
    val trainDataPath = "data/ace05/train";
    val devDataPath = "data/ace05/dev";
    val testDataPath = "data/ace05/test";
    val wikiPath = "data/ace05/ace05-all-conll-wiki"
    val wikiDBPath = "models/wiki-db-ace.ser.gz"
    
    Driver.numberGenderDataPath = "data/gender.data";
    Driver.brownPath = "data/bllip-clusters";
    
 
    val lambda = 1e-8F
    val batchSize = 1
    val numItrs = 20
    
    
    // Read in CoNLL documents
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    val assembler = CorefDocAssembler(Language.ENGLISH, true); //use gold mentions
    val trainDocs = ConllDocReader.loadRawConllDocsWithSuffix(trainDataPath, -1, "", Language.ENGLISH);
    val trainCorefDocs = trainDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    

    

    // Make training examples, filtering out those with solutions that are unreachable because
    // they're not good for training
    val trainExs = extractExamples(trainCorefDocs);
    //val testExs = extractExamples(testCorefDocs, goldWikification, wikiDB, filterImpossible = true);
    
    println("ACE NER chunks: " + trainExs.size);
    
    // Extract features
    val featIndexer = new Indexer[String]
    val maybeBrownClusters = if (Driver.brownPath != "") Some(BrownClusterInterface.loadBrownClusters(Driver.brownPath, 0)) else None;
    val nerFeaturizer = MCNerFeaturizer(Driver.nerFeatureSet.split("\\+").toSet, featIndexer, MCNerFeaturizer.StdLabelIndexer, trainDocs.flatMap(_.words), None, maybeBrownClusters);
    
    var sumLen: Int = 0;
    var maxLen: Int = 0;
    var minLen: Int = 300000;
    var qid = 0;
    
    var allTrainFeats = new Array[Array[Array[Int]]](trainExs.size);
    var allTrains = new ArrayBuffer[tmpExample]();
    for (trainEx <- trainExs) {
      qid += 1;
      val featEachLabel = nerFeaturizer.featurize(trainEx, true);
      allTrainFeats(qid - 1) = featEachLabel;
      allTrains += (new tmpExample(trainEx, featEachLabel));
      for (idx <- 0 until 7) {
        val len = featEachLabel(idx).length;
        sumLen = sumLen + len;
        if (maxLen < len) {
          maxLen = len;
        }
        if (minLen > len) {
          minLen = len
        }
      }
    }

    /////////////// for testing
    val testDocs = ConllDocReader.loadRawConllDocsWithSuffix(testDataPath, -1, "", Language.ENGLISH);
    val testCorefDocs = testDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    val testExs = extractExamples(testCorefDocs);
    val testEmpExs = new ArrayBuffer[tmpExample]();
      for (testEx <- testExs) {
      val featEachLabel = nerFeaturizer.featurize(testEx, false);
      val thisTmp = new tmpExample(testEx, featEachLabel);
      testEmpExs += (thisTmp); // add to list
    }
    
    
    
    // learn!
    val wght = structurePerceptrion(allTrains, featIndexer, testEmpExs);
    
    var avglen: Double = sumLen;
    avglen = avglen / ((trainExs.size).toDouble);
    Logger.logss(featIndexer.size + " features");
    Logger.logss("min = " + minLen + ", maxLen = " + maxLen + ", avgLen = " + avglen);
    // Train
    //val gt = new GeneralTrainer[JointQueryDenotationExample]();
    //val weights = gt.trainAdagrad(trainExs, computer, featIndexer.size, 1.0F, lambda, batchSize, numItrs);
    //val chooser = new JointQueryDenotationChooser(featIndexer, weights)

    testAceNerSystem(testEmpExs, wght, None);

    Logger.logss("All Done!");
  }
 
  def testAceNerSystem(testExs: ArrayBuffer[tmpExample], 
                       weight: Array[Double],
                       mayLogger: Option[PrintWriter]) {
    var total: Double = 0;
    var correct: Double = 0;
    for (testEx <- testExs) {
      val featEachLabel = testEx.feats;//nerFeaturizer.featurize(testEx, false);
      //outputRankingExamples(featWriter2, testEx, qid, featEachLabel);
      val goldLabelIdx = MCNerFeaturizer.StdLabelIndexer.indexOf(getGoldNerTag(testEx.exmp.goldLabel));
      var bestLbl = -1;
      var bestScore = -Double.MaxValue;
      for (l <- 0 until 7) {
        var score = computeScore(weight, featEachLabel(l));
        if (score > bestScore) {
          bestScore = score;
          bestLbl = l;
        }
      }
      //println("bestscore = " + bestScore + ", bestLbl = " + bestLbl + ", gold = " + goldLabelIdx);
      total += 1;
      if (goldLabelIdx == bestLbl) {
          correct += 1
      }
    }
    //featWriter.close();
    
    val accuracy = correct / total;
    println("Total = " + total + ", correct = " + correct + ", acc = " + accuracy);
    
    if (mayLogger != None) {
      val writer = mayLogger.get;
      writer.println(total + ", " + correct + ", " + accuracy);
      writer.flush();
    }
  }
  
  def getGoldNerTag(nerSymbol: String) = {
    if (nerSymbol.contains("-")) {
      nerSymbol.substring(0, nerSymbol.indexOf("-"));
    } else {
      nerSymbol
    }
  }
 
  def structurePerceptrion(allTrains: ArrayBuffer[tmpExample], 
                           featIndexer: Indexer[String],
                           testExs: ArrayBuffer[tmpExample]) : Array[Double] = {
   
    val logger = new PrintWriter("ner_ace05_learning_curve_devset.csv");
    
    
    var weight = Array.fill[Double](featIndexer.size)(0);//new Array[Double](featIndexer.size());
    var weightSum = Array.fill[Double](featIndexer.size)(0);
    val Iteration = 500;
    val learnRate = 0.1;
    val lambda = 1e-8;
    
    var updateCnt = 0;
    
    for (iter <- 0 until Iteration) {
      println("Iter " + iter);
      for (example <- allTrains) {
        val goldTag = getGoldNerTag(example.exmp.goldLabel);
        val goldLabelIdx = MCNerFeaturizer.StdLabelIndexer.indexOf(goldTag);
        var bestLbl = -1;
        var bestScore = -Double.MaxValue;
        for (l <- 0 until 7) {
          var score = computeScore(weight, example.feats(l));
          if (score > bestScore) {
            bestScore = score;
            bestLbl = l;
          }
        }
        
        // update?
        if (bestLbl != goldLabelIdx) {
          updateCnt += 1;
          if (updateCnt % 1000 == 0) println("Update " + updateCnt);
          updateWeight(weight: Array[Double], 
                       example.feats(goldLabelIdx),
                       example.feats(bestLbl),
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
      testAceNerSystem(testExs, tmpAvg, Some(logger));
    }
    
    divdeNumber(weightSum, updateCnt.toDouble);
    
    for (i <- 0 until weightSum.length) {
      if (weightSum(i) != 0) {
        println("weight(" + i + ") = " + weightSum(i));
      }
    }
    
    logger.close();
    
    weightSum;
  }
  
  def updateWeight(currentWeight: Array[Double], 
                   featGold: Array[Int],
                   featPred: Array[Int],
                   eta: Double,
                   lambda: Double) {
    var gradient = Array.fill[Double](currentWeight.length)(0);//new Array[Double](currentWeight.length);
    for (i <- featGold) {
      if (i >= 0) {
        gradient(i) += (1.0);
      }
    }
    for (j <- featPred) {
      if (j >= 0) {
        gradient(j) -= (1.0);
      }
    }
    
    // do L2 Regularization
    //var l1norm = getL1Norm(currentWeight);
    for (i2 <- 0 until currentWeight.length) {
      //var regularizerNum: Double = Math.max(0, b);
      //var regularizerDen: Double = Math.max(0, b);
      var reg: Double = 1.0 - (eta * lambda)
      var curWeightVal = currentWeight(i2) * reg;
      currentWeight(i2) = curWeightVal + (gradient(i2) * eta);
      //currentWeight(i2) += (gradient(i2) * eta);
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
  
  
  def outputRankingExamples(writer: PrintWriter, exmp: MCNerExample, globalQid: Int, featEachLabel: Array[Array[Int]]) {
    val goldTag = getGoldNerTag(exmp.goldLabel);
    val goldLabelIdx = MCNerFeaturizer.StdLabelIndexer.indexOf(goldTag);
    println(goldTag+" = "+goldLabelIdx);
    var includeTrue = false;
    for (l <- 0 until 7) {
        val feat = featEachLabel(l);
        val rank = if (goldLabelIdx == l) 1 else 0;
        
        // print!
        writer.print(rank);
        writer.print(" qid:" + globalQid);        
        for (i <- 0 until feat.length) {
          val idx = feat(i) + 1;
          writer.print(" " + idx + ":1.0");
        }
        writer.println();
        
        if (goldLabelIdx == l) {
          includeTrue = true;
        }
    }
    if (!includeTrue) {
      throw new RuntimeException("No ground truth!");
    }
  }
    
  def main(args: Array[String]) {
    NerTestingInterfaceACE();
  }
  
  
  
  
  
  
  
  
  
  
  
}

object WikiStructUtils {
  
  
  
  
  
  
    
  class queryWikiValue(val query: Query,
      val wiki: String,
      val qfeats: Array[Int],
      val qwfeats: Array[Int],
      val isCorrect: Boolean) {
    
      val concatenateFeats = qfeats ++ qwfeats;

    def computeScore(wght: Array[Double]): Double = {
      var result : Double = 0;
      for (idx1 <- qfeats) {
        result += (wght(idx1));
      }
      for (idx2 <- qwfeats) {
        result += (wght(idx2));
      }
      result;
    }
    
    def getConcatenateFeatures() = {
      concatenateFeats
    }
    
    def getFeatStr(qid: Int): String = {
      var result = "";
      val feat = concatenateFeats;
      result += (if (isCorrect) "1" else "0");
      result += (" qid:" + qid);
      for (i <- feat) {
        result += (" " + String.valueOf(i + 1) + ":" + String.valueOf(1.0));
      }
      result;
    }

  }
  
  class myWikiPredictor(val featIndexer: Indexer[String],
                        val weights: Array[Double],
                         wikiDB: WikipediaInterface) {
    
    // compute the wiki value with highest score
    def predictBestDenotation(ex: JointQueryDenotationExample) : String = {
      val domains = constructWikiExampleDomains(ex, wikiDB);
      var bestLbl = -1;
      var bestScore = -Double.MaxValue;
      for (l <- 0 until domains.size) {
          var score = domains(l).computeScore(weights);
          if (score > bestScore) {
            bestScore = score;
            bestLbl = l;
          }
      }
      domains(bestLbl).wiki;
    }
    
  }

  def getGoldWikification(goldWiki: DocWikiAnnots, ment: Mention): Seq[String] = {
    if (!goldWiki.contains(ment.sentIdx)) {
      Seq[String]();
    } else {
      val matchingChunks = goldWiki(ment.sentIdx).filter(chunk => chunk.start == ment.startIdx && chunk.end == ment.endIdx);
      if (matchingChunks.isEmpty) Seq[String]() else matchingChunks(0).label;
    }
  }
  
  def isCorrect(gold: Seq[String], guess: String): Boolean = {
    (gold.map(_.toLowerCase).contains(guess.toLowerCase.replace(" ", "_"))); // handles the -NIL- case too
  }
  
  
  def extractExamples(corefDocs: Seq[CorefDoc], goldWikification: CorpusWikiAnnots, wikiDB: WikipediaInterface, filterImpossible: Boolean = false) = {
    val writer = new PrintWriter("wiki_men.log");
    val exs = new ArrayBuffer[JointQueryDenotationExample];
    var numImpossible = 0;
    // Go through all mentions in all documents
    for (corefDoc <- corefDocs) {
      val docName = corefDoc.rawDoc.docID
      for (i <- 0 until corefDoc.predMentions.size) {
        // Discard "closed class" mentions (pronouns) since these don't have interesting entity links
        if (!corefDoc.predMentions(i).mentionType.isClosedClass()) {
          val ment = corefDoc.predMentions(i);
          // There are multiple possible gold Wikipedia titles for some mentions. Note that
          // NIL (no entry in Wikipedia) is included as an explicit choice, so this includes NILs (as
          // it should according to how the task is defined)
          val goldLabel = getGoldWikification(goldWikification(docName), ment)
          if (goldLabel.size >= 1) {
            val queries = Query.extractQueriesBest(ment, true);
            val queryDisambigs = queries.map(wikiDB.disambiguateBestGetAllOptions(_));
//            val denotations = queries.map(wikiDB.disambiguateBestNoDisambig(_));
            val denotations = Query.extractDenotationSetWithNil(queries, queryDisambigs, maxNumWikificationOptions);
            val correctDenotations = denotations.filter(denotation => isCorrect(goldLabel, denotation))
            // N.B. The use of "isCorrect" here is needed to canonicalize 
            val correctIndices = denotations.zipWithIndex.filter(denotationIdx => isCorrect(goldLabel, denotationIdx._1)).map(_._2);
//            if (correctIndices.isEmpty && 
            if (filterImpossible && correctIndices.isEmpty) {
              numImpossible += 1;
              //println("Impossible Ment: "+ ment.spanToString + " " + goldLabel);
              //for () { 
              //}
              writer.println("Impossible Ment: "+ ment.spanToString);
              writer.println(" === Gold Labels === ");
              for (gl <- goldLabel) {
                writer.println("GoldLbl: " + gl);
              }
              writer.println(" === Gen Labels === ");
              for (ii <- 0 until denotations.size) {
                writer.println("GenLbl: " + denotations(ii));
              }
              writer.println(" === ======== === ");
            } else {
              exs += new JointQueryDenotationExample(queries, denotations, correctDenotations, goldLabel)
            }
          }
        }
      }
    }
    Logger.logss(exs.size + " possible, " + numImpossible + " impossible");
    writer.close();
    exs;
  }

  
  val trainDataPath = "data/ace05/train";
  val devDataPath = "data/ace05/dev";
  val testDataPath = "data/ace05/test";
  val wikiPath = "data/ace05/ace05-all-conll-wiki"
  val wikiDBPath = "models/wiki-db-ace.ser.gz"
  
  val NilToken = "-NIL-";
  
  val lambda = 1e-8F
  val batchSize = 1
  val numItrs = 20
  
  val maxNumWikificationOptions = 7
  
  def trainWikificationACE(args: Array[String]) {
    
    // Read in CoNLL documents 
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    
    val assembler = CorefDocAssembler(Language.ENGLISH, true);
    val trainDocs = ConllDocReader.loadRawConllDocsWithSuffix(trainDataPath, -1, "", Language.ENGLISH);
    val trainCorefDocs = trainDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    
    // Read in gold Wikification labels
    val goldWikification = WikiAnnotReaderWriter.readStandoffAnnotsAsCorpusAnnots(wikiPath)
    // Read in the title given surface database
    val wikiDB = GUtil.load(wikiDBPath).asInstanceOf[WikipediaInterface];
    // Make training examples, filtering out those with solutions that are unreachable because
    // they're not good for training
    val trainExs = extractExamples(trainCorefDocs, goldWikification, wikiDB, filterImpossible = true)
    
    // Extract features
    val featIndexer = new Indexer[String]
    val computer = new JointQueryDenotationChoiceComputer(wikiDB, featIndexer);
    for (trainEx <- trainExs) {
      computer.featurizeUseCache(trainEx, true);
    }
    Logger.logss(featIndexer.size + " features");

    
    // extract test examples
    // Build the test examples and decode the test set
    // No filtering now because we're doing test
    val testDocs = ConllDocReader.loadRawConllDocsWithSuffix(testDataPath, -1, "", Language.ENGLISH);
    val testCorefDocs = testDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    val testExs = extractExamples(testCorefDocs, goldWikification, wikiDB, filterImpossible = false);
    for (tstEx <- testExs) {
      computer.featurizeUseCache(tstEx, false);
    }
    
    // Train
    //val gt = new GeneralTrainer[JointQueryDenotationExample]();
    //val weights = gt.trainAdagrad(trainExs, computer, featIndexer.size, 1.0F, lambda, batchSize, numItrs);
    val weights = structurePerceptrion(trainExs, featIndexer, wikiDB, testExs);//trainWikification(trainExs, featIndexer, testExs);
    //val chooser = new JointQueryDenotationChooser(featIndexer, weights)
    val myModel = new myWikiPredictor(featIndexer, weights, wikiDB);
    
    val goldTestDenotationsAsTrivialChunks = (0 until testExs.size).map(i => new Chunk[Seq[String]](i, i+1, testExs(i).rawCorrectDenotations))
    //val predTestDenotationsAsTrivialChunks = (0 until testExs.size).map(i => new Chunk[String](i, i+1, chooser.pickDenotation(testExs(i).queries, wikiDB)))
    val predTestDenotationsAsTrivialChunks = (0 until testExs.size).map(i => new Chunk[String](i, i+1, myModel.predictBestDenotation(testExs(i))))
    // Hacky but lets us reuse some code that normally evaluates things with variable endpoints
    WikificationEvaluator.evaluateFahrniMetrics(Seq(goldTestDenotationsAsTrivialChunks), Seq(predTestDenotationsAsTrivialChunks), Set())

  }
  
  
  // structural learning
  //def trainWikification(trainExs: ArrayBuffer[JointQueryDenotationExample],
  //                     featIndexer: Indexer[String],
  //                      testExs: ArrayBuffer[JointQueryDenotationExample]): Array[Double] = {
  //  ???
  //}

  def constructWikiExampleDomains(ex: JointQueryDenotationExample,
      wikiDB: WikipediaInterface): ArrayBuffer[queryWikiValue] = {
      val qwvals = new ArrayBuffer[queryWikiValue]();
      val queries = ex.queries;
      val denotations = ex.allDenotations;
      val queryOutcomes = queries.map(query => wikiDB.disambiguateBestGetAllOptions(query));
      
      for (qidx <- 0 until queries.size) {
        for (didx <- 0 until denotations.size) {
          val q = queries(qidx);
          val d = denotations(didx);
          if ((d == NilToken) || (queryOutcomes(qidx).containsKey(d))) {
            val fq = ex.cachedFeatsEachQuery(qidx);
            val fqd = ex.cachedFeatsEachQueryDenotation(qidx)(didx);
            //println(fq.length + ", " + fqd.length)
            val isCorrect = ex.correctDenotationIndices.contains(didx);
            qwvals += (new queryWikiValue(q, d, fq, fqd, isCorrect));
          }
        }
      }
      qwvals
  }
   
  def structurePerceptrion(allTrains: ArrayBuffer[JointQueryDenotationExample],
                           featIndexer: Indexer[String],
                           wikiDB: WikipediaInterface,
                           testExs: ArrayBuffer[JointQueryDenotationExample]) : Array[Double] = {
   
    val logger = new PrintWriter("wiki_ace05_train.txt");
    val logger2 = new PrintWriter("wiki_ace05_test.txt");
  
    var weight = Array.fill[Double](featIndexer.size)(0);//new Array[Double](featIndexer.size());
    var weightSum = Array.fill[Double](featIndexer.size)(0);
    var lastWeight = Array.fill[Double](featIndexer.size)(0);
    
    val Iteration = 1;
    val learnRate = 0.1;
    val lambda = 1e-8;
    
    var updateCnt = 0;
    var lastUpdtCnt = 0;
    
    var exId2 = 0;
    for (extst <- testExs) {
      exId2 += 1;
      val domains = constructWikiExampleDomains(extst, wikiDB);
      for (l <- 0 until domains.size) {
        logger2.println(domains(l).getFeatStr(exId2));
      }
    }
    
    for (iter <- 0 until Iteration) {
      lastUpdtCnt = updateCnt;
      Array.copy(weight, 0, lastWeight, 0, weight.length);
        
      println("Iter " + iter);
      var exId = 0;
      for (example <- allTrains) {
        
        exId += 1;
        val domains = constructWikiExampleDomains(example, wikiDB);

        var bestLbl = -1;
        var bestScore = -Double.MaxValue;
        var bestCorrectLbl = -1; // latent best
        var bestCorrectScore = -Double.MaxValue;
        for (l <- 0 until domains.size) {
          //println(domains(l).getFeatStr(exId));
          logger.println(domains(l).getFeatStr(exId));
          
          var score = domains(l).computeScore(weight);
          if (score > bestScore) {
            bestScore = score;
            bestLbl = l;
          }
          if (domains(l).isCorrect) {
            if (score > bestCorrectScore) {
              bestCorrectScore = score;
              bestCorrectLbl = l;
            }
          }
        }
        
        //println("size = " + domains.size + " pred = " + bestLbl + " correct = " + bestCorrectLbl)
        
        // update?
        if (!domains(bestLbl).isCorrect) {
          updateCnt += 1;
          if (updateCnt % 1000 == 0) println("Update " + updateCnt);
          updateWeight(weight, 
                       domains(bestCorrectLbl).concatenateFeats,
                       domains(bestLbl).concatenateFeats,
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
      //testAceNerSystem(testExs, tmpAvg, Some(logger));
      //quickTest(testExs, tmpAvg);
      
      quickTest(allTrains, tmpAvg, wikiDB);
      quickTest(testExs, tmpAvg, wikiDB);
      println("Iter Update Cnt = " + (updateCnt - lastUpdtCnt));
      
      val wdiff = checkWeight(weight, lastWeight);
      println("Weight diff = " + wdiff);
    }
    
    divdeNumber(weightSum, updateCnt.toDouble);

    for (i <- 0 until weightSum.length) {
      if (weightSum(i) != 0) {
        println("weight(" + i + ") = " + weightSum(i));
      }
    }
    
    logger.close();
    logger2.close();
    weightSum;
  }
  
  def checkWeight(curWeight: Array[Double],
                  oldWeight: Array[Double]): Int = {
    var different: Int = 0;
    for (i <- 0 until curWeight.length) {
      if (curWeight(i) != oldWeight(i)) different += 1;
    }
    different;
  }
  
  def quickTest(testExs: ArrayBuffer[JointQueryDenotationExample],
                           weight: Array[Double],
                            wikiDB: WikipediaInterface) {
    
    var correct: Double = 0.0;
    
    for (ex  <- testExs) {
      val domains = constructWikiExampleDomains(ex, wikiDB);
      var bestLbl = -1;
      var bestScore = -Double.MaxValue;
      for (l <- 0 until domains.size) {
          var score = domains(l).computeScore(weight);
          if (score > bestScore) {
            bestScore = score;
            bestLbl = l;
          }
      }
      if (domains(bestLbl).isCorrect) {
        correct += 1.0;
      }
    }
    
    var total = testExs.size.toDouble;
    var acc = correct / total;
    println("Acc: " + correct + "/" + total + " = " + acc);
  }
  
  def updateWeight(currentWeight: Array[Double], 
                   featGold: Array[Int],
                   featPred: Array[Int],
                   eta: Double,
                   lambda: Double) {
    var gradient = Array.fill[Double](currentWeight.length)(0);//new Array[Double](currentWeight.length);
    
    
    val possibleNonZeroIdx = new ArrayBuffer[Int]();
    
    for (i <- featGold) {
      if (i >= 0) {
        gradient(i) += (1.0);
        possibleNonZeroIdx += (i);
      }
    }
    for (j <- featPred) {
      if (j >= 0) {
        gradient(j) -= (1.0);
        possibleNonZeroIdx += (j);
      }
    }
    /*
    var cnt = 0;
    for (nzidx <- possibleNonZeroIdx) {
      if (gradient(nzidx) != 0) cnt += 1;
    }*/
    
    //println("gradient non-zero element cnt: " + cnt);
    
    // do L2 Regularization
    //var l1norm = getL1Norm(currentWeight);
    for (i2 <- 0 until currentWeight.length) {
      //var regularizerNum: Double = Math.max(0, b);
      //var regularizerDen: Double = Math.max(0, b);
      var reg: Double = 1.0 - (eta * lambda)
      var curWeightVal = currentWeight(i2) * reg;
      currentWeight(i2) = curWeightVal + (gradient(i2) * eta);
      //currentWeight(i2) += (gradient(i2) * eta);
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
  
  
  
  
  
  
  
  
  
  
}

object CorefStructUtils {
  

/*
class CorefTesting(val featurizer: PairwiseIndexingFeaturizer,
                   val weights: Array[Double]) {
  
  def inference(docGraph: DocumentGraph) : Array[Int] = {
    
    val featsChart = docGraph.featurizeIndexNonPrunedUseCache(featurizer);
    for (i <- 0 until docGraph.size) {
      for (j <- 0 to i) {
        //if (!prunedEdges(i)(j)) {
          //require(featsChart(i)(j).size > 0);
          featurizer.featurizeIndex(docGraph, i, j, false);
          
          //scoreChart(i)(j) = GUtil.scoreIndexedFeats(featsChart(i)(j), scorer.weights);
        //} else {
        //  scoreChart(i)(j) = Float.NegativeInfinity;
        //}
      }
    }
   // (featsChart, scoreChart)
    
    
    ???
  }
  
  // treat each mention decision as independent example, and predict
  def inferenceDecisionByDecision(docGraph: DocumentGraph) : Array[Int] = {
    
    val featsChart = docGraph.featurizeIndexNonPrunedUseCache(featurizer);
    val decisions = CorefTesting.extractDeciExmlOneDoc(docGraph);
    
    val result = new Array[Int](decisions.length);
    for (i <- 0 until decisions.length) {
      result(i) = decisions(i).getBestValue(weights);
      //result(i) = decisions(i).getOracleBestValue(weights);
    }
    
    result;
  }
  
  def mentionDecisoin() {
    
  }

  def numWeights(): Int = {
    weights.size
  }
}
*/
  

/*
class CorefDecisionExample(val index: Int,
                           val values: Array[Int], 
                           val goldValues: Array[Int],
                           val features: Array[Array[Int]]) {
  
  val correctSet = ((new HashSet[Int]()) ++ (goldValues));
  
  // return the value that has the highest score
  def getBestValue(wght: Array[Double]): Int = {
      var bestLbl = -1;
      var bestScore = -Double.MaxValue;
      for (j <- 0 until values.size) {
        val l = values(j);
        var score = CorefTesting.computeScore(wght, features(j));
        if (score > bestScore) {
          bestScore = score;
          bestLbl = j;
        }
      }
      values(bestLbl);
  }
  
  def getOracleBestValue(wght: Array[Double]): Int = {
      var bestLbl = -1;
      var bestScore = -Double.MaxValue;
      var bestGoldLbl = -1;
      var bestGoldScore = -Double.MaxValue;
      for (j <- 0 until values.size) {
        val l = values(j);
        var score = CorefTesting.computeScore(wght, features(j));
        if (score > bestScore) {
          bestScore = score;
          bestLbl = l;
        }
        if (correctSet.contains(l)) {
          if (score > bestGoldScore) {
            bestGoldScore = score;
            bestGoldLbl = l;
          }
        }
      }
      if (bestGoldLbl < 0) {
        bestLbl;
      }
      bestGoldLbl;
  }
  
  def isCorrect(value: Int): Boolean = {
    correctSet.contains(value);
  }
  
}
*/

//object CorefTesting {
  
  def mainCoref(args: Array[String]) {
    val trainDataPath = "data/ace05/train";
    val devDataPath = "data/ace05/dev";
    val testDataPath = "data/ace05/test";
    
    // set some configs
    Driver.useGoldMentions = true;
    Driver.doConllPostprocessing = false;
    
    TrainBerkCorefACE(trainDataPath, testDataPath);
  }
  
/*
  def loadCorefDocs(path: String, size: Int, suffix: String, maybeNumberGenderComputer: Option[NumberGenderComputer]): Seq[CorefDoc] = {
    val docs = ConllDocReader.loadRawConllDocsWithSuffix(path, size, suffix);
    val assembler = CorefDocAssembler(Driver.lang, Driver.useGoldMentions);
    val mentionPropertyComputer = new MentionPropertyComputer(maybeNumberGenderComputer);
    val corefDocs = if (Driver.useCoordination) {
      docs.map(doc => assembler.createCorefDocWithCoordination(doc, mentionPropertyComputer));
    } else {
      docs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    }
    CorefDocAssembler.checkGoldMentionRecall(corefDocs);
    corefDocs;
  }
*/
  def loadCorefDocs(path: String, size: Int, suffix: String,
                    //maybeNumberGenderComputer: Option[NumberGenderComputer], 
                    mentionPropertyComputer: MentionPropertyComputer): Seq[CorefDoc] = {
    val docs = ConllDocReader.loadRawConllDocsWithSuffix(path, size, suffix);
    val assembler = CorefDocAssembler(Driver.lang, Driver.useGoldMentions);
    //val mentionPropertyComputer = new MentionPropertyComputer(maybeNumberGenderComputer);
    val corefDocs = if (Driver.useCoordination) {
      docs.map(doc => assembler.createCorefDocWithCoordination(doc, mentionPropertyComputer));
    } else {
      docs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    }
    CorefDocAssembler.checkGoldMentionRecall(corefDocs);
    corefDocs;
  }
    
  def preprocessDocsCacheResources(allDocGraphs: Seq[DocumentGraph]) {
    if (Driver.wordNetPath != "") {
      val wni = new WordNetInterfacer(Driver.wordNetPath);
      allDocGraphs.foreach(_.cacheWordNetInterfacer(wni));
    }
  }
  

  
  def TrainBerkCorefACE(trainPath: String, testPath: String) {
    
    val trainSize = -1;
    
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    val queryCounts: Option[QueryCountsBundle] = None;
    val trainDocs = loadCorefDocs(trainPath, trainSize, Driver.docSuffix, mentionPropertyComputer);
    // Randomize
    val trainDocsReordered = new scala.util.Random(0).shuffle(trainDocs);
    val lexicalCounts = LexicalCountsBundle.countLexicalItems(trainDocs, Driver.lexicalFeatCutoff);
    val semClasser: Option[SemClasser] = Driver.semClasserType match {
    case "basic" => Some(new BasicWordNetSemClasser);
    case e => throw new RuntimeException("Other semclassers not implemented");
    }
    val trainDocGraphs = trainDocsReordered.map(new DocumentGraph(_, true));
    preprocessDocsCacheResources(trainDocGraphs);
    CorefPruner.buildPruner(Driver.pruningStrategy).pruneAll(trainDocGraphs);

    val featureIndexer = new Indexer[String]();
    featureIndexer.getIndex(PairwiseIndexingFeaturizerJoint.UnkFeatName);
    val featureSetSpec = FeatureSetSpecification(Driver.pairwiseFeats, Driver.conjScheme, Driver.conjFeats, Driver.conjMentionTypes, Driver.conjTemplates);
    val basicFeaturizer = new PairwiseIndexingFeaturizerJoint(featureIndexer, featureSetSpec, lexicalCounts, queryCounts, semClasser);
    val featurizerTrainer = new CorefFeaturizerTrainer();
    featurizerTrainer.featurizeBasic(trainDocGraphs, basicFeaturizer);
    PairwiseIndexingFeaturizer.printFeatureTemplateCounts(featureIndexer)

    
    val decisionTrainExmps = extractDecisionExamples(trainDocGraphs);
    
    
    ////////////// About testing examples //////////////////////
    val devDocGraphs = prepareTestDocuments(testPath, -1, mentionPropertyComputer);
    featurizerTrainer.featurizeBasic(devDocGraphs, basicFeaturizer);
    val decisionTestExmps = extractDecisionExamples(devDocGraphs);
    /////////////////////////////////////////////////////////
    
    
    val wght = structurePerceptrion(decisionTrainExmps, featureIndexer, decisionTestExmps);
    val mmodel = new CorefTesting(basicFeaturizer, wght);
    testCorefOnDocs(testPath, -1, mmodel);
    
 /*   
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
    
    val corefModel = new PairwiseScorer(basicFeaturizer, firstPassWeights).pack;
*/
    
    
  }
  
  def extractDecisionExamples(docGraphs: Seq[DocumentGraph]): ArrayBuffer[CorefDecisionExample] = {
    val result = new ArrayBuffer[CorefDecisionExample]();
    for (i <- 0 until docGraphs.size) {
      result ++= extractDeciExmlOneDoc(docGraphs(i));
    }
    result;
  }
  def extractDeciExmlOneDoc(docGraph: DocumentGraph): ArrayBuffer[CorefDecisionExample] = {
   
    val exmpleArr = new ArrayBuffer[CorefDecisionExample]();
    
    for (i <- 0 until docGraph.size) {
      val valArr = new ArrayBuffer[Int]();
      val featArr = new ArrayBuffer[Array[Int]]();
      val goldArr = new ArrayBuffer[Int]();

      for (j <- 0 to i) {
        if (!docGraph.prunedEdges(i)(j)) {
          //require(featsChart(i)(j).size > 0);
          //featurizer.featurizeIndex(docGraph, i, j, false);
          //scoreChart(i)(j) = GUtil.scoreIndexedFeats(featsChart(i)(j), scorer.weights);
          valArr += j;
          featArr += (docGraph.cachedFeats(i)(j));
          if (docGraph.isGoldNoPruning(i, j)) {
            goldArr += j
          }
        } else {
          // was pruned
          //  scoreChart(i)(j) = Float.NegativeInfinity;
        }
      }
      
      exmpleArr += (new CorefDecisionExample(i, valArr.toArray, goldArr.toArray, featArr.toArray));
    }
    
    exmpleArr;
  }
  
  def structurePerceptrion(allTrains: ArrayBuffer[CorefDecisionExample],
                           featIndexer: Indexer[String],
                           testExs: ArrayBuffer[CorefDecisionExample]) : Array[Double] = {

      //val logger = new PrintWriter("wiki_ace05_train.txt");
      //val logger2 = new PrintWriter("wiki_ace05_test.txt");

      var weight = Array.fill[Double](featIndexer.size)(0);//new Array[Double](featIndexer.size());
      var weightSum = Array.fill[Double](featIndexer.size)(0);
      var lastWeight = Array.fill[Double](featIndexer.size)(0);

      val Iteration = 5;
      val learnRate = 0.1;
      val lambda = 1e-8;

      var updateCnt = 0;
      var lastUpdtCnt = 0;

      /*
      var exId2 = 0;
      for (extst <- testExs) {
        exId2 += 1;
        val domains = constructWikiExampleDomains(extst, wikiDB);
        for (l <- 0 until domains.size) {
          logger2.println(domains(l).getFeatStr(exId2));
        }
      }*/

      for (iter <- 0 until Iteration) {
        lastUpdtCnt = updateCnt;
        Array.copy(weight, 0, lastWeight, 0, weight.length);

        println("Iter " + iter);
        var exId = 0;
        for (example <- allTrains) {

          exId += 1;
          val domains = example.values;
          val prunOk = (example.goldValues.length != 0);
          if (prunOk) {

            var bestLbl = -1;
            var bestScore = -Double.MaxValue;
            var bestCorrectLbl = -1; // latent best
            var bestCorrectScore = -Double.MaxValue;
            for (j <- 0 until domains.size) {
              val l = domains(j);
              //println(domains(l).getFeatStr(exId));
              //logger.println(domains(l).getFeatStr(exId));
              var score = computeScore(weight, example.features(j));
              if (score > bestScore) {
                bestScore = score;
                bestLbl = j;
              }
              if (example.isCorrect(l)) {
                if (score > bestCorrectScore) {
                  bestCorrectScore = score;
                  bestCorrectLbl = j;
                }
              }
            }

            //println("size = " + domains.size + " pred = " + bestLbl + " correct = " + bestCorrectLbl)

            // update?
            if (!example.isCorrect(domains(bestLbl))) {
              updateCnt += 1;
              if (updateCnt % 1000 == 0) println("Update " + updateCnt);
              updateWeight(weight, 
                           example.features(bestCorrectLbl),
                           example.features(bestLbl),
                           learnRate,
                           lambda);
              sumWeight(weightSum, weight);
            }
          }
        }

        ///////////////////////////////////////////////////
        // have a test after each iteration (for learning curve)
        val tmpAvg = new Array[Double](weightSum.size)
            Array.copy(weightSum, 0, tmpAvg, 0, weightSum.size);
        divdeNumber(tmpAvg, updateCnt.toDouble);

        //quickTest(allTrains, tmpAvg, wikiDB);
        quickTest(testExs, tmpAvg);
        println("Iter Update Cnt = " + (updateCnt - lastUpdtCnt));

        val wdiff = checkWeight(weight, lastWeight);
        println("Weight diff = " + wdiff);
      }

      divdeNumber(weightSum, updateCnt.toDouble);

      /*
      for (i <- 0 until weightSum.length) {
        if (weightSum(i) != 0) {
          println("weight(" + i + ") = " + weightSum(i));
        }
      }
      */

      //logger.close();
      //logger2.close();
      weightSum;
  }
  
  def checkWeight(curWeight: Array[Double],
                  oldWeight: Array[Double]): Int = {
    var different: Int = 0;
    for (i <- 0 until curWeight.length) {
      if (curWeight(i) != oldWeight(i)) different += 1;
    }
    different;
  }
  

  def quickTest(testExs: ArrayBuffer[CorefDecisionExample],
                weight: Array[Double]) {
    
    var correct: Double = 0.0;
    
    for (ex  <- testExs) {
      val pred = ex.getBestValue(weight);
      if (ex.isCorrect(pred)) {
        correct += 1.0;
      }
    }
    
    var total = testExs.size.toDouble;
    var acc = correct / total;
    println("Acc: " + correct + "/" + total + " = " + acc);
  }

  

  def updateWeight(currentWeight: Array[Double], 
                   featGold: Array[Int],
                   featPred: Array[Int],
                   eta: Double,
                   lambda: Double) {
    var gradient = Array.fill[Double](currentWeight.length)(0);//new Array[Double](currentWeight.length);
    
    
    val possibleNonZeroIdx = new ArrayBuffer[Int]();
    
    for (i <- featGold) {
      if (i >= 0) {
        gradient(i) += (1.0);
        possibleNonZeroIdx += (i);
      }
    }
    for (j <- featPred) {
      if (j >= 0) {
        gradient(j) -= (1.0);
        possibleNonZeroIdx += (j);
      }
    }
    
    // do L2 Regularization
    //var l1norm = getL1Norm(currentWeight);
    for (i2 <- 0 until currentWeight.length) {
      //var regularizerNum: Double = Math.max(0, b);
      //var regularizerDen: Double = Math.max(0, b);
      var reg: Double = 1.0 - (eta * lambda)
      var curWeightVal = currentWeight(i2) * reg;
      currentWeight(i2) = curWeightVal + (gradient(i2) * eta);
      //currentWeight(i2) += (gradient(i2) * eta);
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

  //////////////////////////////////////////////////////////////////////////////////////////

  
  def prepareTestDocuments(devPath: String, devSize: Int, mentionPropertyComputer: MentionPropertyComputer): Seq[DocumentGraph] = {
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val devDocs = loadCorefDocs(devPath, devSize, Driver.docSuffix, mentionPropertyComputer);
    val devDocGraphs = devDocs.map(new DocumentGraph(_, false));
    preprocessDocsCacheResources(devDocGraphs);
    CorefPruner.buildPruner(Driver.pruningStrategy).pruneAll(devDocGraphs);
    devDocGraphs;
  }
  
  def testCorefOnDocs(devPath: String, 
                      devSize: Int,
                      myModel: CorefTesting) {
    
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    val devDocGraphs = prepareTestDocuments(devPath, devSize, mentionPropertyComputer);

    Logger.startTrack("My Coref Testing:");
    val (allPredBackptrs, allPredClusterings) = testDocGraphs(devDocGraphs, myModel);  
    val scoreOutput = CorefEvaluator.evaluateAndRender(devDocGraphs, allPredBackptrs, allPredClusterings, Driver.conllEvalScriptPath, "DEV: ", Driver.analysesToPrint);
    Logger.logss(scoreOutput);
    Logger.endTrack();
  }
  
  def testDocGraphs(docGraphs: Seq[DocumentGraph], myModel: CorefTesting): (Array[Array[Int]], Array[OrderedClustering]) = {
    
    val corefFeatizer = new CorefFeaturizerTrainer();
    corefFeatizer.featurizeBasic(docGraphs, myModel.featurizer);
    
    // results
    val allPredBackptrs = new Array[Array[Int]](docGraphs.size);

    for (i <- 0 until docGraphs.size) {
      val docGraph = docGraphs(i);
      Logger.logs("Decoding " + i);
      val predBackptrs = myModel.inferenceDecisionByDecision(docGraph);//, (i + 1));//myModel.inference(docGraph);
      allPredBackptrs(i) = predBackptrs;
    }
    
    ////
    val allPredClusteringsSeq = (0 until docGraphs.size).map(i => OrderedClustering.createFromBackpointers(allPredBackptrs(i)));
    val allPredClusteringsArr = allPredClusteringsSeq.toArray;
    
    // return
    (allPredBackptrs, allPredClusteringsArr);
  }
  
  
}