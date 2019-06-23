package berkeleyentity.oregonstate

import edu.berkeley.nlp.futile.fig.basic.Indexer

import edu.illinois.cs.cogcomp.sl.core._
import edu.illinois.cs.cogcomp.sl.learner._
import edu.illinois.cs.cogcomp.sl.util._
import edu.illinois.cs.cogcomp.sl.core.IStructure
import edu.illinois.cs.cogcomp.sl.core.IInstance
import edu.illinois.cs.cogcomp.sl.util.IFeatureVector
import edu.illinois.cs.cogcomp.sl.core.SLModel
import edu.illinois.cs.cogcomp.sl.core.SLParameters
import edu.illinois.cs.cogcomp.sl.core.SLProblem
import edu.illinois.cs.cogcomp.sl.core.AbstractFeatureGenerator
import edu.illinois.cs.cogcomp.sl.util.SparseFeatureVector
import edu.illinois.cs.cogcomp.sl.util.FeatureVectorBuffer
import edu.illinois.cs.cogcomp.sl.core.AbstractInferenceSolver

import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._

import java.util.ArrayList;
import java.nio.file.Files;
import java.nio.file.CopyOption._;
import java.io.PrintWriter;
import java.io.File;
import scala.util.control.Breaks._
import edu.illinois.cs.cogcomp.sl.latentsvm.AbstractLatentInferenceSolver
import edu.illinois.cs.cogcomp.sl.latentsvm.LatentLearner
import berkeleyentity.MyTimeCounter
import edu.berkeley.nlp.futile.util.Counter
import berkeleyentity.oregonstate.pruner.StaticDomainPruner

class MyLatentSvmLearner(val baseLearner: Learner, 
                         val fg: AbstractFeatureGenerator, 
                         val params: SLParameters, 
                         val solver: AbstractLatentInferenceSolver) extends Learner(solver,fg, params) {
  
  def train(problem: SLProblem) = {
		train(problem, new WeightVector(10000));
	}
  
	def train(problem: SLProblem, w_init: WeightVector): WeightVector = {
	  
	  val infr = solver.asInstanceOf[StructuralSVMLearner.MultiTaskInferencer];

		var w = w_init; // new WeightVector(100000);//baseLearner.train(problem); // init w

		//for (int outerIter = 0; outerIter < params.MAX_NUM_ITER; outerIter++) {
		for (outerIter <- 0 until params.MAX_NUM_ITER) {

		  val new_prob = runLatentStructureInference(problem, w, solver); // returns structured problem with (x_i,h_i)
			w = baseLearner.train(new_prob, w); // update weight vector
			
			//w.checkFloatDoubleConsistency();

			if (params.PROGRESS_REPORT_ITER > 0 && (outerIter+1) % params.PROGRESS_REPORT_ITER == 0 && this.f != null) {
				f.run(w, solver);
			}
		}

		return w;
	}
	
  def train(problem: SLProblem,
            w_init: WeightVector,
            searcher: SearchBasedLearner,
            testExs: Seq[AceJointTaskExample],
            beamSize: Int,
            restart: Int,
            pruner: StaticDomainPruner): WeightVector = {
	  
	  //val infr = solver.asInstanceOf[StructuralSVMLearner.MultiTaskInferencer];

		var w = w_init; // new WeightVector(100000);//baseLearner.train(problem); // init w

		//for (int outerIter = 0; outerIter < params.MAX_NUM_ITER; outerIter++) {
		for (outerIter <- 0 until params.MAX_NUM_ITER) {

		  val new_prob = runLatentStructureInference(problem, w, solver); // returns structured problem with (x_i,h_i)
			w = baseLearner.train(new_prob, w); // update weight vector
			
			//w.checkFloatDoubleConsistency();

			if (params.PROGRESS_REPORT_ITER > 0 && (outerIter+1) % params.PROGRESS_REPORT_ITER == 0 && this.f != null) {
				f.run(w, solver);
			}
			
			
			// have a check the performance
      searcher.beamSearchQuickTest(testExs, beamSize, w.getDoubleArray, pruner, restart);
		}

		return w;
	}

	def runLatentStructureInference(problem: SLProblem,
														      w: WeightVector, 
                                  inference: AbstractLatentInferenceSolver): SLProblem = {
	  
		val p = new SLProblem();
		for (i <- 0 until problem.size()) {
			val x = problem.instanceList.get(i);
			val gold = problem.goldStructureList.get(i);
			val y = inference.getBestLatentStructure(w, x, gold); // best
			// explaining
			// latent
			// structure
			p.instanceList.add(x);
			p.goldStructureList.add(y);
		}

		return p;
	}
}

object StructuralSVMLearner {
  
  //////////////////////////////////////
  // for UIUC Structural Learning Lib //
  //////////////////////////////////////
  
  class UiucSvmModel extends SLModel {
    
  }

  
  class UiucOutput(var output: Array[Int]) extends IStructure {
    
    def setOutput(newOutput: Array[Int]) {
      if (newOutput.length != output.length) {
        throw new RuntimeException("Wrong length for output: " + newOutput.length);
      }
      output = newOutput;
    }
    
    //@Override
    override def equals(o1: Any): Boolean = {
      val vec = o1.asInstanceOf[UiucOutput];
      if (vec.output.length != output.length) {
        throw new RuntimeException("Wrong length for output: " + vec.output.length);
      }
      output.corresponds(vec.output){_ == _};
    }
  }
  
  
  class MultiTaskFeatGener extends AbstractFeatureGenerator {
    def getFeatureVector(xi: IInstance, yhati: IStructure): IFeatureVector = {
      val x = xi.asInstanceOf[AceJointTaskExample];
      val yhat = yhati.asInstanceOf[UiucOutput];
      
      val fb = new FeatureVectorBuffer();
      val valMap: HashMap[Int,Double] = x.featurize(yhat.output);//x.getSparseFeatureVector(yhat.output);
      //val idxArr = new ArrayBuffer[Int]();
      //val valArr = new ArrayBuffer[Double]();
      for (idx <- valMap.keySet) {
        //fb.addFeature(idx + 1, valMap.get(idx).get);
        fb.addFeature(idx, valMap.get(idx).get);
      }
      fb.toFeatureVector();
    }
    override def getFeatureVectorDiff(x: IInstance, y1: IStructure, y2: IStructure): IFeatureVector = {
    		val f1 = getFeatureVector(x, y1);
    		val f2 = getFeatureVector(x, y2);		
    		return f1.difference(f2);
    }
  }
  
  class MultiTaskInferencer(val beamSize: Int,
                            val searcher: SearchBasedLearner,
                            val pruner: StaticDomainPruner,
                            val featIndexer: Indexer[String]) extends AbstractLatentInferenceSolver {
    
    val callCount = new Counter[Int](); // for efficiency statistic
    
    def getBestStructure(wi: WeightVector, xi: IInstance): IStructure = {
      callCount.incrementCount(1, 1);
      getLossAugmentedBestStructure(wi, xi, null);
    }

    def getLoss(xi: IInstance, ystari: IStructure, yhati: IStructure): Float = {
      val x = xi.asInstanceOf[AceJointTaskExample];
      val yhat = yhati.asInstanceOf[UiucOutput];
      val loss = x.getWeightedZeroOneError(yhat.output);
      loss.toFloat;//x.getZeroOneError(yhat.output).toFloat; // non-normalized loss
    }
    
    def getLossAugmentedBestStructure(wi: WeightVector, xi: IInstance, ystari: IStructure): IStructure = {
      
      callCount.incrementCount(0, 1);
    	val example = xi.asInstanceOf[AceJointTaskExample];
    	val ystar = ystari.asInstanceOf[UiucOutput];

    	val wgtArr = wi.getDoubleArray();

      val initState = SearchBasedLearner.getInitStateWithUnaryScore(example, pruner, true);
      val initWithMusk = searcher.prunedActionSpaceMusk(example, initState);
      
      val isLossAugment = (ystari != null); // if no gold output, run regular inference
      //val predBestOutput = searcher.hillClimbing(example, initWithMusk, wgtArr, false, isLossAugment).output; // my prediction
      val predBestOutput = searcher.beamSearch(example, initWithMusk, beamSize, wgtArr,false, isLossAugment).output; // my prediction
      //val predBestOutput = SingleTaskInferener.unaryFactorInference(example, wgtArr, false, isLossAugment).output; // my prediction
      
      (new UiucOutput(predBestOutput));
    }
    
    override def getBestLatentStructure(weight: WeightVector, ins: IInstance, gold: IStructure): IStructure = {
      callCount.incrementCount(2, 1);
    	val example = ins.asInstanceOf[AceJointTaskExample];
    	val gout = gold.asInstanceOf[UiucOutput];
    		
    	val wgtArr = weight.getDoubleArray();//weight.getWeightArray();//getWeightArray(weight, featIndexer.size);//getDoubleWeightVector(weight);
    		
    	val initState = SearchBasedLearner.getInitStateWithUnaryScore(example, pruner, true);//.getRandomInitState(example);
    	//val initWithMusk = prunedActionSpaceMusk(example, initState);
    	//val predBestOutput = hillClimbing(example, initWithMusk, weight, false).output;  // my prediction
    	val goldInit = searcher.constructGoldMuskNoPredict(example, initState);
    	
    	//val goldBestOutput = searcher.hillClimbing(example, goldInit, wgtArr, true, false).output; // gold best
    	val goldBestOutput = searcher.beamSearch(example, goldInit, beamSize, wgtArr, true, false).output; // gold best
    	//val goldBestOutput = SingleTaskInferener.unaryFactorInference(example, wgtArr, true, false).output; // gold best

    	(new UiucOutput(goldBestOutput));
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
    
    override def clone() = {
    	val cpInferencer = new MultiTaskInferencer(beamSize, searcher, pruner, featIndexer);
    	cpInferencer;
    }
  }

  def constructProblem(exs: ArrayBuffer[AceJointTaskExample]): SLProblem = {
    val sp: SLProblem = new SLProblem();
    for (ex <- exs) {
      val goldOutput = new UiucOutput(new Array[Int](ex.totalSize)); // a randome output ...
      sp.addExample(ex, goldOutput);
    }
    sp;
  }
  
  def uiucStructLearning(trainExs: ArrayBuffer[AceJointTaskExample], 
                         featIndexer: Indexer[String],
                         testExs: ArrayBuffer[AceJointTaskExample],
                         beamSize: Int,
                         restart: Int,
                         searcher: SearchBasedLearner,
                         unaryPruner: StaticDomainPruner): Array[Double] = {
    
		val trainTimer = new MyTimeCounter("Training time");
	  trainTimer.start();
    
    val slcfgPath = "config/uiuc-sl-config/myDCD-search.config";
    val model = new UiucSvmModel();
    
    val spTrain = constructProblem(trainExs);

    val featGenr = new MultiTaskFeatGener();
    val latentInfr = new MultiTaskInferencer(beamSize, searcher, unaryPruner, featIndexer);
    
    // 
    val para = new SLParameters();
    para.loadConfigFile(slcfgPath);
    para.TOTAL_NUMBER_FEATURE = featIndexer.size();

    val baseLearner: Learner = LearnerFactory.getLearner(latentInfr, featGenr, para);
    
    //// about latent settings
    val latentPara = new SLParameters();
    latentPara.TOTAL_NUMBER_FEATURE = featIndexer.size();
    latentPara.MAX_NUM_ITER = 1;
    
    //val latentLearner = new LatentLearner(baseLearner, featGenr, latentPara, latentInfr);
    val latentLearner = new MyLatentSvmLearner(baseLearner, featGenr, latentPara, latentInfr);
    model.infSolver = latentInfr;
    //model.wv = latentLearner.train(spTrain, new WeightVector(latentPara.TOTAL_NUMBER_FEATURE));
    model.wv = latentLearner.train(spTrain, new WeightVector(latentPara.TOTAL_NUMBER_FEATURE), searcher, testExs, beamSize, restart, unaryPruner);
    WeightVector.printSparsity(model.wv);
    
    latentInfr.printCount();
    println("C = " + baseLearner.getParameters.C_FOR_STRUCTURE);
    println("Iterations = " + latentPara.MAX_NUM_ITER);
    
    // training time count~
    trainTimer.end();
    trainTimer.printSecond("Training time");
    
    printLossWeight();

    // save the model
    //model.saveModel("models/mymodel/mysvm-ace05.txt");
    //getDoubleWeightVector(model.wv);
    convertDoubleWeightArray(model.wv);
  }
  
  def printLossWeight() {
    println("Loss Weight ==> (" + JointTaskStructTesting.CorefErrorWeight + ", " + JointTaskStructTesting.NerErrorWeight + ", " + JointTaskStructTesting.WikiErrorWeight + ")");
  }
  
  def svmLoadModel(): Array[Double] =  {
    val model = SLModel.loadModel("models/mymodel/mysvm-ace05.txt");
    //getDoubleWeightVector(model.wv);
    convertDoubleWeightArray(model.wv);
  }
  
  def convertDoubleWeightArray(wv: WeightVector): Array[Double] = {
		val farr = wv.getWeightArray; // (0) is bias
    var darr = new Array[Double](farr.length);
	  for (i <- 0 until darr.length) {
		  darr(i) = farr(i).toDouble;
	  }
	  darr;
  }
  
}


  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////

object SingleTaskSVMLearning {
  

	class SingleOutput(var output: Array[Int]) extends IStructure {
		def setOutput(newOutput: Array[Int]) {
			if (newOutput.length != output.length) {
				throw new RuntimeException("Wrong length for output: " + newOutput.length);
			}
			output = newOutput;
		}
		override def equals(o1: Any): Boolean = {
				val vec = o1.asInstanceOf[SingleOutput];
				if (vec.output.length != output.length) {
					throw new RuntimeException("Wrong length for output: " + vec.output.length);
				}
				output.corresponds(vec.output){_ == _};
		}
	}
  
  class SingleTaskFeatGener[T] extends AbstractFeatureGenerator {
    def getFeatureVector(xi: IInstance, yhati: IStructure): IFeatureVector = {
      val x = xi.asInstanceOf[AceSingleTaskStructExample[T]];
      val yhat = yhati.asInstanceOf[SingleOutput];
      
      val fb = new FeatureVectorBuffer();
      val valMap: HashMap[Int,Double] = x.featurize(yhat.output);
      for (idx <- valMap.keySet) {
        //fb.addFeature(idx + 1, valMap.get(idx).get);
        fb.addFeature(idx, valMap.get(idx).get);
      }
      fb.toFeatureVector();
    }
    override def getFeatureVectorDiff(x: IInstance, y1: IStructure, y2: IStructure): IFeatureVector = {
    		val f1 = getFeatureVector(x, y1);
    		val f2 = getFeatureVector(x, y2);		
    		return f1.difference(f2);
    }
  }
  
  class SingleTaskInferencer[T](val lossScale: Double = 1.0) extends AbstractInferenceSolver {
    
    def getBestStructure(wi: WeightVector, xi: IInstance): IStructure = {
      getLossAugmentedBestStructure(wi, xi, null);
    }

    def getLoss(xi: IInstance, ystari: IStructure, yhati: IStructure): Float = {
      val x = xi.asInstanceOf[AceSingleTaskStructExample[T]];
      val yhat = yhati.asInstanceOf[SingleOutput];
      x.getZeroOneError(yhat.output).toFloat; // non-normalized loss
    }
    
    def getLossAugmentedBestStructure(weight: WeightVector, xi: IInstance, ystari: IStructure): IStructure = {

    		val mi = xi.asInstanceOf[AceSingleTaskStructExample[T]];;
    		val gmi = ystari.asInstanceOf[SingleOutput];

    		val bestOutput = new Array[Int](mi.getVarNumber);

    		for (j <- 0 until mi.getVarNumber) {

    			var bestScore: Float = -Float.MaxValue;
          val values = mi.variables(j).values;

          for (i <- 0 until values.length) {
    			  var score = computeFeatDotWeight(values(i).feature, weight);//weight.dotProduct(featureGener.getFeatureVector(mi, currentl));
    			  // loss augmented!
            if (ystari != null) {
            	if (!(values(i).isCorrect)) {
            		score += lossScale.toFloat;
            	}
            }

    			  if (score > bestScore){
    			  	bestOutput(j) = i;
    			  	bestScore = score;
    			  }
    	  	}
    		}

    		(new SingleOutput(bestOutput));   
    }
    
    def computeFeatDotWeight(fv: Array[Int], wv: WeightVector): Float = {
      var result: Float = 0;
      if (fv == null) {
        return Float.NegativeInfinity;//throw new RuntimeException("null fv!");
      }
      for (idx <- fv) {
        result += wv.get(idx);//wv.get(idx + 1);
      }
      //result += wv.get(0); // add bias?
      result;
    }
  }

  class SingleTaskProblem() extends SLProblem {
    
	  def loadExample(exs: ArrayBuffer[AceMultiTaskExample]) {
		  loadNerExample(exs);
	  }
    
    def loadCorefExample(exs: ArrayBuffer[AceMultiTaskExample]) {
      for (ex <- exs) {
       instanceList.add(ex.corefOutput);
       val goldOut = getGold[Int](ex.corefOutput);//Array.fill(ex.corefOutput.getVarNumber)(-1);
       goldStructureList.add(new SingleOutput(goldOut));
      }
    }

    def loadNerExample(exs: ArrayBuffer[AceMultiTaskExample]) {
      for (ex <- exs) {
       instanceList.add(ex.nerOutput);
       val goldOut = getGold[String](ex.nerOutput);//Array.fill(ex.nerOutput.getVarNumber)(-1);
       goldStructureList.add(new SingleOutput(goldOut));
      }
    }

    def loadWikiExample(exs: ArrayBuffer[AceMultiTaskExample]) {

    }
    
    
    ////////////////////
    def getGold[T](singleEx: AceSingleTaskStructExample[T]) : Array[Int] = {
      val vars = singleEx.variables;
      val output = Array.fill(singleEx.getVarNumber)(-1);
      for (i <- 0 until vars.length) {
        val corrIdxs = new ArrayBuffer[Int]();
        for (j <- 0 until vars(i).values.length) {
          if (vars(i).values(j).isCorrect) { corrIdxs += j; }
        }
        output(i) = corrIdxs(0);
      }
      output;
    }
    
  }
  
  def NerSingleStructLearning(trainExs: ArrayBuffer[AceMultiTaskExample], 
		                          featIndexer: Indexer[String],
		                          testExs: ArrayBuffer[AceMultiTaskExample]): Array[Double] = {


		  val model = new SLModel();
		  model.lm = new Lexiconer();

		  val sp = new SingleTaskProblem(); 
		  sp.loadNerExample(trainExs);

		  // Disallow the creation of new features
		  model.lm.setAllowNewFeatures(false);

		  // initialize the inference solver
		  val sovler = new SingleTaskInferencer();
		  model.infSolver = sovler;

		  val fg = new SingleTaskFeatGener();
		  val para = new SLParameters();
      val slcfgPath = "config/uiuc-sl-config/myDCD-ner-struct.config";
      //val slcfgPath = "config/uiuc-sl-config/myStructuredPerceptron-ner-struct.config";
		  para.loadConfigFile(slcfgPath);
		  para.TOTAL_NUMBER_FEATURE = featIndexer.size();
      println("Feature count: " + featIndexer.size());

		  val learner: Learner = LearnerFactory.getLearner(model.infSolver, fg, para);
		  model.wv = learner.train(sp);
		  WeightVector.printSparsity(model.wv);

		  // save the model
		  //model.saveModel(modelPath);
		  getDoubleWeightVector(model.wv);
  }
  
  def CorefSingleStructLearning(trainExs: ArrayBuffer[AceMultiTaskExample], 
                               featIndexer: Indexer[String],
                               testExs: ArrayBuffer[AceMultiTaskExample]): Array[Double] = {


      val model = new SLModel();
      model.lm = new Lexiconer();

      val sp = new SingleTaskProblem(); 
      sp.loadCorefExample(trainExs);

      // Disallow the creation of new features
      model.lm.setAllowNewFeatures(false);

      // initialize the inference solver
      val sovler = new SingleTaskInferencer();
      model.infSolver = sovler;

      val fg = new SingleTaskFeatGener();
      val para = new SLParameters();
      //val slcfgPath = "config/uiuc-sl-config/myDCD-coref-struct.config";
      val slcfgPath = "config/uiuc-sl-config/myStructuredPerceptron-coref-struct.config";
      para.loadConfigFile(slcfgPath);
      para.TOTAL_NUMBER_FEATURE = featIndexer.size();

      val learner: Learner = LearnerFactory.getLearner(model.infSolver, fg, para);
      model.wv = learner.train(sp);
      WeightVector.printSparsity(model.wv);

      // save the model
      //model.saveModel(modelPath);
      getDoubleWeightVector(model.wv);
  }
  
  
  def getDoubleWeightVector(wv: WeightVector): Array[Double] = {
    val farr = wv.getWeightArray; // (0) is bias
    var darr = new Array[Double](farr.length);
    for (i <- 0 until darr.length) {
      darr(i) = farr(i).toDouble;
    }
    darr;
  }

}