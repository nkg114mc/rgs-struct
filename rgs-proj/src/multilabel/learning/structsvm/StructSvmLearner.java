package multilabel.learning.structsvm;

import multilabel.instance.Example;
import multilabel.instance.Featurizer;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
 
import multilabel.learning.inferencer.SearchInferencer;
import multilabel.pruner.PrunerLearning;
import multilabel.data.Dataset;
import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.SLModel;
import edu.illinois.cs.cogcomp.sl.core.SLParameters;
import edu.illinois.cs.cogcomp.sl.core.SLProblem;
import edu.illinois.cs.cogcomp.sl.learner.Learner;
import edu.illinois.cs.cogcomp.sl.learner.LearnerFactory;
import multilabel.evaluation.MultiLabelEvaluator;

public class StructSvmLearner {
	
	public static void main(String[] args) {

		String name = "medical";
		System.out.println("Name: " + name);
		
		// read dataset
		Dataset ds = PrunerLearning.readArffFileTest(name);
		
		// train!
		SLModel mdl = structSvmLearning(ds);
		System.out.println("Done SVM Training.");
		
		//
		testDataset(ds, mdl);
		System.out.println("Done SVM Testing.");
	}
	
	public static void testDataset(Dataset ds, SLModel model) {

		try {
			PrintWriter writer = new PrintWriter("test_output.log");

			//ILPinferencer inferencer = new ILPinferencer();
			SearchInferencer inferencer = new SearchInferencer(50, 3);
			ArrayList<Example> testExs = ds.getTestExamples();
			for (int i = 0; i < testExs.size(); i++) {
				Example ex = testExs.get(i);
				ex.predictOutput = inferencer.inference(ex, model.wv, null, false);
				writer.println(ex.predictOutput.toString());
			}
			writer.close();

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		// scoring!
		MultiLabelEvaluator evaluator = new MultiLabelEvaluator();
		evaluator.evaluationDataSet(ds.name, ds.getTestExamples());
	}
	
	public static SLModel structSvmLearning(Dataset ds) {
		
		SLModel model = null;
		ArrayList<Example> trainExs = ds.getTrainExamples();
		
		try {
			
			//String slcfgPath = "uiuc-sl-config/myDCD.config";
			String slcfgPath = "uiuc-sl-config/myDCD-search.config";
			model = new SLModel();

			SLProblem sp = new SLProblem();
			ArrayList<StrucSvmInstance> trainInsts = new ArrayList<StrucSvmInstance>();
			for (Example ex : trainExs) {
				StrucSvmInstance inst = new StrucSvmInstance(ex);
				trainInsts.add(inst);
				sp.addExample(inst, inst.example.getGroundTruthOutput());
			}

			// initialize the inference solver
			StrucSvmInferencer solver = new StrucSvmInferencer(50,3);
			model.infSolver = solver;

			StructSvmFeatGen fg = new StructSvmFeatGen();
			SLParameters para = new SLParameters();

			para.loadConfigFile(slcfgPath);
			para.TOTAL_NUMBER_FEATURE = Featurizer.getFeatureDimension(ds.getFeatureDimension(), ds.getLabelDimension());//featIndexer.size();

			Learner learner = LearnerFactory.getLearner(model.infSolver, fg, para);
			model.wv = learner.train(sp);
			edu.illinois.cs.cogcomp.sl.util.WeightVector.printSparsity(model.wv);
			getDoubleWeightVector(para.TOTAL_NUMBER_FEATURE, model.wv);

			//if(learner instanceof L2LossSSVMLearner)
			//  System.out.println("Primal objective:" + ((L2LossSSVMLearner)learner).getPrimalObjective(sp, model.wv, model.infSolver, para.C_FOR_STRUCTURE));

			// save the model
			//model.saveModel(modelPath);


			//getDoubleWeightVector(model.wv);

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return model;
	}
	
	public static void getDoubleWeightVector(int featSize, edu.illinois.cs.cogcomp.sl.util.WeightVector wv) {
		float[] farr = wv.getWeightArray();
		for (int i = 0; i < featSize; i++) {
			System.out.println("weight(" + i + ") = "+ farr[i + 1]);
		}
	}

}
