package init;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import experiment.RndLocalSearchExperiment.InitType;
import general.AbstractInstance;
import general.AbstractOutput;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.XGBoost;
import multilabel.data.DatasetReader;
import multilabel.data.HwInstanceDataset;
import multilabel.evaluation.MultiLabelEvaluator;
import multilabel.instance.Example;
import multilabel.instance.Label;
import multilabel.learning.StructOutput;
import multilabel.utils.UtilFunctions;
import search.SearchState;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;
import sequence.hw.HwSegment;

public class MultiLabelXgbRndGenerator extends RandomStateGenerator {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1597974283328828258L;
	private int labelSize;
	private int domainSize;
	private List<Booster> boosters;
	private Random random;
	
	public MultiLabelXgbRndGenerator(int seqsz, int dmsz, List<Booster> logsMds, Random rnd) {
		labelSize = seqsz;
		domainSize = dmsz;
		boosters = logsMds;
		random = rnd;
		if (labelSize != boosters.size()) {
			throw new RuntimeException("Labels and model numbers are not consistent!");
		}
		if (dmsz != 2) {
			throw new RuntimeException("Dimension size != " + dmsz);
		}
		System.out.println("Create multi-label logistic initializer.");
	}
	
	public MultiLabelXgbRndGenerator(int seqsz, int dmsz, List<Booster> logsMds) {
		labelSize = seqsz;
		domainSize = dmsz;
		boosters = logsMds;
		random = new Random();
		if (labelSize != boosters.size()) {
			System.out.println("Labels and model numbers are not consistent!");
		}
		if (dmsz != 2) {
			throw new RuntimeException("Dimension size != " + dmsz);
		}
		System.out.println("Create multi-label logistic initializer.");
	}
	
	public HashSet<SearchState> generateRandomInitState(AbstractInstance inst, int stateNum) {
		double[][] probs = predictOnInstance(inst);
		HashSet<SearchState> genStates = new HashSet<SearchState>();
		for (int i = 0; i < stateNum; i++) {
			AbstractOutput rndout = sampleOneOutput(probs);
			genStates.add(new SearchState(rndout));
		}
		return genStates;
	}
	
	public SearchState generateSingleRandomInitState(AbstractInstance inst) {
		HashSet<SearchState> sset = generateRandomInitState(inst,1);
		SearchState result = null;
		for (SearchState s : sset) {
			result = s;
			break;
		}
		return result;
	}
	
	public AbstractOutput sampleOneOutput(double[][] probilities) {
		
		HwOutput output = new HwOutput(probilities.length, Label.MULTI_LABEL_DOMAIN);
		for (int i = 0; i < output.size(); i++) {
			int sampledIdx = sampleWithProbs(probilities[i]);
			output.setOutput(i, sampledIdx);
		}
		
		return output;
	}
	
	public int sampleWithProbs(double[] probilities) {
		double totalWeight = 0;
		for (int i = 0; i < probilities.length; ++i) {
			totalWeight += probilities[i];
		}
		////////////////////////////////////////////
		int randomIndex = -1;
		double randomProb = random.nextDouble() * totalWeight;
		for (int i = 0; i < probilities.length; ++i) {
			randomProb -= probilities[i];
		    if (randomProb <= 0.0d) {
		        randomIndex = i;
		        break;
		    }
		}
		return randomIndex;
	}
	
	// linear model
	public double[][] predictOnInstance(AbstractInstance abInst) {
		
		double[][] p = new double[labelSize][abInst.domainSize()];

		try {
			HwInstance inst = (HwInstance) abInst;
			for (int i = 0; i < labelSize; i++) {
				Booster booster = boosters.get(i);
				HwSegment seg = inst.letterSegs.get(i);
				p[i] = predictProbWithModel(booster, seg, inst.alphabet);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return p;
	}
	
	private double[] predictProbWithModel(Booster booster, HwSegment seg, String[] alphabet) {
		double[] p = predictScoreWithModel(booster, seg, alphabet);
		return p;
	}
	
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	
	public static double[] getFeatureWithSingleLabel(HwSegment seg, int label) {
		assert ((label == 0) || (label == 1));
		double[] fv = seg.getFeatArr();
		return fv;
	}
	
	private static double[] predictScoreWithModel(Booster booster, HwSegment seg, String[] alphabet) {
		double[] sc = new double[alphabet.length];
		//for (int i = 0; i < alphabet.length; i++) {
		//	double[] fv = getFeatureWithSingleLabel(seg, i);
		//	sc[i] = XgbInitClassifierLearning.scoringGivenBoosterAndFeature(fv, booster);
		//}
		
		double[] fv = getFeatureWithSingleLabel(seg, 1);
		sc[1] = XgbInitClassifierLearning.scoringGivenBoosterAndFeature(fv, booster);
		sc[0] = 1.0 - sc[1];
		return sc;
	}
	
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	
	public static MultiLabelXgbRndGenerator loadGenrIfExist(String path, String datasetName, List<HwInstance> trnInsts, List<HwInstance> tstInsts, String[] alphabet, boolean doDebug) {
		Object obj = UtilFunctions.loadObj(path);
		if (obj == null) {
			// retrain
			MultiLabelXgbRndGenerator trn_g = trainXgbModel(datasetName, trnInsts, tstInsts, alphabet, doDebug);
			UtilFunctions.saveObj(trn_g, path);
			return trn_g;
		} else {
			MultiLabelXgbRndGenerator gnr = (MultiLabelXgbRndGenerator)obj;
			return gnr;
		}
	}
	
	
	//// Training
	
	public static MultiLabelXgbRndGenerator trainXgbModel(String datasetName, List<HwInstance> trnInsts, List<HwInstance> tstInsts, String[] alphabet, boolean doDebug) {

		int labelCnt = trnInsts.get(0).size();
		
		try {
			
			String[] mdfiles = new String[labelCnt];
			List<Booster> logsModels = new ArrayList<Booster>();
			

			for (int l = 0; l < labelCnt; l++) {
				// dump lth label learning
				List<HwSegment> segs = instancesToSegs(trnInsts, l);
				System.out.println("==== Start Logistic Training on Label " + l + "th ====");
				Booster bstr = XgbInitClassifierLearning.trainSingleClassifier(segs, alphabet);
				//logsModels.add(bstr);
				
				// dump md
				String fn = SeqSamplingRndGenerator.ARFF_DUMP_FOLDER + "/" + datasetName + "_xgb_label" + String.valueOf(l) + ".model";
				mdfiles[l] = fn;
				bstr.saveModel(fn);
				
				System.out.println("==== Done Learning ====");
				System.gc();
			}
			
			System.gc();

			for (int l = 0; l < labelCnt; l++) {
				Booster booster2 = XGBoost.loadModel(mdfiles[l]);
				logsModels.add(booster2);
			}
			
			///// quick test
			System.out.println("==== Eval Logistic Model on TrainSet ====");
			testXgbModel(trnInsts, alphabet,  logsModels);
			System.out.println("");
			System.out.println("");
			System.out.println("==== Eval Logistic Model on TestSet ====");
			testXgbModel(tstInsts, alphabet, logsModels);
			
			/////////////////////////////////////
			MultiLabelXgbRndGenerator genr = new MultiLabelXgbRndGenerator(labelCnt, alphabet.length, logsModels);
			return genr;
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return null; // should reach here ...
	}

	public static void testXgbModel(List<HwInstance> trnInsts, String[] alphabet, List<Booster> logistics) {
		
		
		try {
			
			int lcnt = logistics.size();
			ArrayList<Example> mlexs = new ArrayList<Example>();
			
			for (int j = 0; j < trnInsts.size(); j++) {
				
				ArrayList<Label> labels = new ArrayList<Label>();
				StructOutput predictOutput = new StructOutput(lcnt);

				List<HwSegment> segs = trnInsts.get(j).letterSegs;
				for (int l = 0; l < segs.size(); l++) {
					HwSegment seg = segs.get(l);
					seg.goldIndex = getGoldValueIdx(trnInsts.get(j).alphabet, seg.letter);
					//System.out.println("Class = " +ins.classValue());
					int goldv = (int)(seg.goldIndex);
					int predv = -1;
					double maxProb = -1;
					double subProb = 0;
					double[] probs = predictScoreWithModel(logistics.get(l), seg, alphabet);

					for (int k = 0; k < probs.length; k++) {
						//System.out.println("predict = " + probs[k]);
						subProb += probs[k];
						if (probs[k] > maxProb) {
							maxProb = probs[k];
							predv = k;
						}
					}
					labels.add(new Label(l, goldv));
					predictOutput.setValue(l, predv);
				}
				
				Example ex = new Example(j, labels, null);
				ex.predictOutput = predictOutput;
				mlexs.add(ex);
			}
			
			
			MultiLabelEvaluator evaluator = new MultiLabelEvaluator();
			evaluator.evaluationDataSet("some-multi-label", mlexs);
			
		} catch (Exception e) {
			e.printStackTrace();
		}	
	}

	
	public static List<HwSegment> instancesToSegs(List<HwInstance> insts, int whichLabel) {
		ArrayList<HwSegment> segInsts = new ArrayList<HwSegment>();
		for (HwInstance inst : insts) {
			List<HwSegment> segs = inst.letterSegs;
			HwSegment seg = segs.get(whichLabel);
			seg.goldIndex = getGoldValueIdx(inst.alphabet, seg.letter);
			segInsts.add(seg);
		}
		return segInsts;
	}

	public static int getGoldValueIdx(String[] alphabet, String goldlb) {
		int gv = -1;
		for (int i = 0; i < alphabet.length; i++) {
			if (goldlb.equals(alphabet[i])) {
				gv = i;
				break;
			}
		}
		if (gv < 0) {
			throw new RuntimeException("Gold value index = " + gv);
		}
		return gv;
	}
	
	@Override
	public InitType getType() {
		return InitType.LOGISTIC_INIT;
	}

	@Override
	public void testPerformance(List<HwInstance> insts, String[] alphabet) {
		System.out.println("==== Test Logistic Model on TestSet ====");
		testXgbModel(insts, alphabet, boosters);
	}
	
	
	
	///////////////////////////////////
	public static void main(String[] args) {
		String dsName = "bibtex";//"bookmarks-umass";//"yeast";//
		
		HwInstanceDataset ds = DatasetReader.loadHwInstances(dsName);
		
		//String logisticMdPath = "../logistic_models/"+dsName+".logistic"; 
		//MultiLabelXgbRndGenerator initStateGener = loadGenrIfExist(logisticMdPath, dsName, ds.getTrainExamples(), ds.getTestExamples(), Label.MULTI_LABEL_DOMAIN, false);
		
		MultiLabelXgbRndGenerator initStateGener = trainXgbModel(dsName, ds.getTrainExamples(), ds.getTestExamples(), Label.MULTI_LABEL_DOMAIN, false);
		
		System.out.println("=======");
		System.out.println("Done.");
	}
	
}
