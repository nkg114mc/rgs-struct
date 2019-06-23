package sequence.hw;

import java.util.HashMap;
import java.util.List;
import java.util.Random;

import edu.illinois.cs.cogcomp.sl.core.SLModel;
import edu.illinois.cs.cogcomp.sl.core.SLParameters;
import edu.illinois.cs.cogcomp.sl.core.SLProblem;
import edu.illinois.cs.cogcomp.sl.learner.Learner;
import edu.illinois.cs.cogcomp.sl.learner.LearnerFactory;
import general.FactorGraphBuilder.FactorGraphType;
import init.RandomStateGenerator;
import init.UniformRndGenerator;
import search.GreedySearcher;
import search.ZobristKeys;

public class HandWritingSmallMain {

	/**
	 * Hand-Writing Recognition Problem
	 * 
	 * Chao Ma
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
/*		
		try {
			
			HandWritingMain.runLearning(true);

			String configFilePath = "sl-config/hw-small-search-DCD.config";
			int nrnd = 1;

			HwLabelSet hwLabels = new HwLabelSet();
			// load data
			HwDataReader rder = new HwDataReader();
			List<List<HwInstance>> trtstInsts = rder.readData("../datasets/hw", hwLabels, 0, true);
			System.out.println("List count = " + trtstInsts.size());
			List<SLProblem> slproblems = HwDataReader.convertToSLProblem(trtstInsts);

			//////////////////////////////////////////////////////////////////////
			// train
			SLModel model = new SLModel();
			SLProblem spTrain = slproblems.get(0);

			// initialize the inference solver
			ZobristKeys abkeys = new ZobristKeys(100, hwLabels.getLabels().length);
			HwFeaturizer fg = new HwFeaturizer(hwLabels.getLabels(), HwFeaturizer.HwSingleLetterFeatLen, true, true, true);
			//model.infSolver = new HwInferencer(fg);//HwSearchInferencer(fg);
			
			RandomStateGenerator initStateGener = new UniformRndGenerator(new Random());
			GreedySearcher searcher = new GreedySearcher(FactorGraphType.SequenceGraph, fg, nrnd, initStateGener, abkeys);
			model.infSolver = new HwSearchInferencer(searcher);
			//model.infSolver = new HwViterbiInferencer(fg);
			model.featureGenerator = fg;

			SLParameters para = new SLParameters();
			para.loadConfigFile(configFilePath);
			para.TOTAL_NUMBER_FEATURE = fg.getFeatLen();

			Learner learner = LearnerFactory.getLearner(model.infSolver, fg, para);
			model.wv = learner.train(spTrain);
			model.config =  new HashMap<String, String>();

			// test
			//////////////////////////////////
			SLProblem spTest = slproblems.get(1);
			HandWritingMain.evaluate(spTest, model);
			System.out.println("Done.");

		} catch(Exception e) {
			System.err.println(e);
		}

	
 */

	}
}