package elearning;

import java.util.List;

import edu.illinois.cs.cogcomp.sl.core.SLModel;
import edu.illinois.cs.cogcomp.sl.core.SLParameters;
import edu.illinois.cs.cogcomp.sl.core.SLParameters.LearningModelType;
import edu.illinois.cs.cogcomp.sl.core.SLProblem;
import edu.illinois.cs.cogcomp.sl.learner.Learner;
import edu.illinois.cs.cogcomp.sl.learner.LearnerFactory;
import edu.illinois.cs.cogcomp.sl.learner.l2_loss_svm.L2LossSSVMLearner;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import elearning.LowLevelCostLearning.StopType;
import elearning.einfer.ELinearSearchInferencer;
import elearning.einfer.ESearchInferencer;
import general.AbstractFeaturizer;
import general.AbstractInstance;
import general.AbstractOutput;
import general.FactorGraphBuilder.FactorGraphType;
import init.RandomStateGenerator;
import search.GreedySearcher;
import search.SearchResult;
import search.SearchTrajectory;
import search.ZobristKeys;
import sequence.hw.HwDataReader;
import sequence.hw.HwInstance;
import sequence.hw.HwSearchInferencer;

public class AlterElearning {
	
	public static EInferencer learnEFunction(RandomStateGenerator randomGenr,
			List<HwInstance> instances,
			AbstractFeaturizer efeaturizer,
			FactorGraphType fgt,
			ZobristKeys abkeys,

			WeightVector cost_weight, 
			GreedySearcher gsearcher, 

			AbstractRegressionLearner regressionTrainer,
			int iteration,
			boolean applyInstWght,
			StopType howStop,
			
			SLParameters givenPara) {

		
		GreedySearcher esearcher = new GreedySearcher(fgt, efeaturizer, 1, gsearcher.getActionGener(), randomGenr,  gsearcher.getLossFunc(), abkeys);

		
		WeightVector e_weight = null;
		////////////reference depth ///////////////////////////////////////
		int orginalDepthSum = 0;
		for (AbstractInstance ainst : instances) {
			AbstractOutput y_start = (randomGenr.generateSingleRandomInitState(ainst).structOutput);
			SearchResult result0 = gsearcher.runSearchGivenInitState(cost_weight, ainst, y_start, null, false);
			SearchTrajectory traj = result0.getUniqueTraj();
			orginalDepthSum += traj.getStateList().size();
		}
		System.out.println("Original sum depth: " + orginalDepthSum);
		/////////////////////////////////////////////////////////////////////
		

		SLModel model = new SLModel();
		SLProblem spTrain = HwDataReader.ExampleListToSLProblem(instances);//slproblems.get(0);

		model.infSolver = new HwSearchInferencer(esearcher);
		model.featureGenerator = efeaturizer;

		SLParameters para = new SLParameters();
		if (givenPara != null) {
			System.out.println("Use given param.");
			para = givenPara;
			para.TOTAL_NUMBER_FEATURE = efeaturizer.getFeatLen();
		} else {
			//para.loadConfigFile(configPath);
			initParams(para);
			para.TOTAL_NUMBER_FEATURE = efeaturizer.getFeatLen();
		}


		Learner learner = LearnerFactory.getLearner(model.infSolver, efeaturizer, para);
		WeightVector initwv = new WeightVector(para.TOTAL_NUMBER_FEATURE);
		try {
			model.wv = learner.train(spTrain, initwv);
		} catch (Exception e) {
			e.printStackTrace();
		}

		e_weight = model.wv;
		
		EInferencer einfr = new ELinearSearchInferencer(esearcher, e_weight);
		return einfr;
	}
	
	public static void initParams(SLParameters para) {
		
		para.LEARNING_MODEL = LearningModelType.L2LossSSVM;
		para.L2_LOSS_SSVM_SOLVER_TYPE = L2LossSSVMLearner.SolverType.DCDSolver;

		para.NUMBER_OF_THREADS = 1;
		para.C_FOR_STRUCTURE = 0.01f;
		para.TRAINMINI = false;
		para.TRAINMINI_SIZE = 1000;
		para.STOP_CONDITION = 0.1f;
		para.CHECK_INFERENCE_OPT = false;
		para.MAX_NUM_ITER = 250;
		para.PROGRESS_REPORT_ITER = 10;
		para.INNER_STOP_CONDITION = 0.1f;
		para.MAX_ITER_INNER = 250;
		para.MAX_ITER_INNER_FINAL = 2500;
		para.TOTAL_NUMBER_FEATURE = -1;
		para.CLEAN_CACHE = true;
		para.CLEAN_CACHE_ITER = 5;
		para.DEMIDCD_NUMBER_OF_UPDATES_BEFORE_UPDATE_BUFFER = 100;
		para.DEMIDCD_NUMBER_OF_INF_PARSE_BEFORE_UPDATE_WV = 10;
		para.LEARNING_RATE = 0.01f;
		para.DECAY_LEARNING_RATE = false;
	}

}
