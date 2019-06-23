package elearning;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearning.einfer.LinearSearchStateScorer;
import elearning.einfer.SearchStateScoringFunction;
import init.SeqSamplingRndGenerator;
import weka.classifiers.functions.SGD;
import weka.core.Instances;
import weka.core.SelectedTag;

public class SGDRegressionLearner extends AbstractRegressionLearner {
	
	public static final String ARFF_DUMP_FOLDER = "./ArffDump";
	
	private String name;
	private SGD sgdRegressor;
	
	public SGDRegressionLearner(String nm) {
		name = nm;
		sgdRegressor = null;//new SGD();
		/*
		sgdRegressor.setLossFunction(new SelectedTag(SGD.SQUAREDLOSS, SGD.TAGS_SELECTION)); // for regression!
		
		String[] opts = sgdRegressor.getOptions();
		for (String s : opts) {
			System.out.println(s);
		}*/
	}


	@Override
	public SearchStateScoringFunction regressionTrain(ArrayList<RegressionInstance> regrDataIter, int featLen, int iterNum) {
		try {
			// dump file
			SeqSamplingRndGenerator.checkArffFolder(ARFF_DUMP_FOLDER);			
			String fn =  ARFF_DUMP_FOLDER + "/" + name + "_evaluation_iter" + String.valueOf(iterNum) + ".arff";;
			dumpArff(regrDataIter, featLen, fn);
			
			regrDataIter.clear(); // save memory

			// load from file
			FileReader fdr = new FileReader(fn);
			Instances wkInsts = new Instances(fdr);
			wkInsts.setClassIndex(wkInsts.numAttributes() - 1);


			System.out.println("NumClasses = " + wkInsts.numClasses());
			System.out.println("NumAttrs = " + wkInsts.numAttributes());
			System.out.println("NumInstss = " + wkInsts.numInstances());

			// train
			
			sgdRegressor = new SGD();
			//sgdRegressor.setLossFunction(new SelectedTag(SGD.SQUAREDLOSS, SGD.TAGS_SELECTION)); 
			sgdRegressor.setLossFunction(new SelectedTag(SGD.EPSILON_INSENSITIVE, SGD.TAGS_SELECTION)); 
			
			
			sgdRegressor.buildClassifier(wkInsts);


			double[] coeffs = sgdRegressor.getWeights();
			WeightVector wv = doubleArrtoWeightVec(coeffs);
			return (new LinearSearchStateScorer(wv));

		} catch (Exception e) {
			e.printStackTrace();
		}

		return null;
	}
	
	public static WeightVector doubleArrtoWeightVec(double[] wv) {
		WeightVector wght = new WeightVector(wv.length);
		for (int i = 0; i < wv.length; i++) {
			//assert (!Double.isNaN(wv[i]));
			if (Double.isNaN(wv[i])) {
				System.err.println("wv[" + i + "] = NaN");
				wv[i] = 0;
			}
			wght.setElement(i, (float)(wv[i]));
		}
		return wght;
	}
	
	public static void dumpArff(List<RegressionInstance> segInsts, int featLen, String fn) {

		
		try {
			PrintWriter pw = new PrintWriter(fn);
			String names = null;
			
			for (RegressionInstance dp : segInsts) {
				if (names == null) {
					names = "";
					
					// title
					pw.println("@relation EvaluationLearning");
					pw.println();
					
					// attributes
					for (int j = 0; j < featLen; j++) {
						String attr = "Feat"+String.valueOf(j);
						pw.println("@attribute " + attr + " numeric");
					}
					// label
					pw.println("@attribute Class numeric");
					pw.println();
					pw.println("@data");
				}
				String csvStr = dp.toArffSparseVecStr(featLen);
				pw.println(csvStr);
			}
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
	}

}
