package init;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import edu.illinois.cs.cogcomp.sl.core.SLModel;
import edu.illinois.cs.cogcomp.sl.core.SLParameters;
import edu.illinois.cs.cogcomp.sl.core.SLParameters.LearningModelType;
import edu.illinois.cs.cogcomp.sl.core.SLProblem;
import edu.illinois.cs.cogcomp.sl.learner.Learner;
import edu.illinois.cs.cogcomp.sl.learner.LearnerFactory;
import edu.illinois.cs.cogcomp.sl.learner.l2_loss_svm.L2LossSSVMLearner;
import edu.illinois.cs.cogcomp.sl.util.IFeatureVector;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import experiment.RndLocalSearchExperiment.InitType;
import general.AbstractInstance;
import general.AbstractOutput;
import multilabel.utils.UtilFunctions;
import search.SearchState;
import sequence.hw.HwDataReader;
import sequence.hw.HwFeaturizer;
import sequence.hw.HwInferencer;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;
import sequence.hw.HwSegment;

public class HwFastSamplingRndGenerator extends RandomStateGenerator {

	private static final long serialVersionUID = 1156880291653176557L;

	private Random random;
	private WeightVector unaryWeight;
	
	private HwFeaturizer ufeaturizer = null;
	

	public HwFastSamplingRndGenerator(int dmsz, WeightVector wv) {
		unaryWeight = wv;
		random = new Random();
		System.out.println("Create HW linear initializer.");
	}
	
	public HashSet<SearchState> generateRandomInitState(AbstractInstance inst, int stateNum) {
		
		String[] dm = ((HwInstance)inst).alphabet;
		double[][] probs = predictOnInstance(inst);
		
		HashSet<SearchState> genStates = new HashSet<SearchState>();
		for (int i = 0; i < stateNum; i++) {
			AbstractOutput rndout = sampleOneOutput(probs, dm);
			//AbstractOutput rndout = bestOneOutput(probs, dm);
			genStates.add(new SearchState(rndout));
		}
		
		//System.out.println("Generate " + genStates.size() + " initial states.");
		
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
	
	public AbstractOutput sampleOneOutput(double[][] probilities, String[] dm) {
		
		HwOutput output = new HwOutput(probilities.length, dm);
		for (int i = 0; i < output.size(); i++) {
			int sampledIdx = SeqSamplingRndGenerator.sampleWithProbs(probilities[i], random);
			output.setOutput(i, sampledIdx);
		}
		
		return output;
	}
	
	// unit-test only
	private AbstractOutput bestOneOutput(double[][] probilities, String[] dm) {
		
		HwOutput output = new HwOutput(probilities.length, dm);
		for (int i = 0; i < output.size(); i++) {
			int bestIdx = pickLargestIndex(probilities[i]);
			output.setOutput(i, bestIdx);
		}
		
		return output;
	}
	private int pickLargestIndex(double[] probility) {
		double maxw = -Double.MAX_VALUE;
		int maxIdx = -1;
		for (int i = 0; i < probility.length; ++i) {
			if (probility[i] > maxw) {
				maxw = probility[i];
				maxIdx = i;
			}
		}
		return maxIdx;
	}


	// linear model
	public double[][] predictOnInstance(AbstractInstance abInst) {
		
		double[][] p = new double[abInst.size()][abInst.domainSize()];

		try {
			HwInstance inst = (HwInstance) abInst;
			if (ufeaturizer == null) {
				ufeaturizer = new HwFeaturizer(inst.alphabet, HwFeaturizer.HwSingleLetterFeatLen, false, false, false);
			}
			
			List<HwSegment> segs = inst.letterSegs;
			for (int i = 0; i < inst.size(); i++) {
				p[i] = predictWithLinearModel(segs.get(i), inst.alphabet);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return p;
	}
	
	private double[] predictWithLinearModel(HwSegment seg, String[] alphabet) {

		double[] sc = new double[alphabet.length];
		for (int i = 0; i < alphabet.length; i++) {
			IFeatureVector fv = getSegFeature(seg, alphabet, i);
			sc[i] = unaryWeight.dotProduct(fv);
		}
		
		double sumExp = 0;
		double[] p = new double[alphabet.length];
		
		sumExp = 0;
		for (int i = 0; i < sc.length; i++) {
			p[i] = Math.exp(sc[i]);
			sumExp += p[i];
		}
		for (int i = 0; i < p.length; i++) {
			p[i] =  p[i] / sumExp;
		}
		
		return p;
	}
	
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	
	//// Training
	
	public static HwFastSamplingRndGenerator loadGenrIfExist(String path, String datasetName, List<HwInstance> trnInsts, List<HwInstance> tstInsts, String[] alphabet, boolean doDebug, int iterMax) {
		Object obj = UtilFunctions.loadObj(path);
		if (obj == null) {
			// retrain
			//ImageSegSamplingRndGenerator trn_g = trainLogisticModel(datasetName, trnInsts, tstInsts, alphabet, doDebug,iterMax);
			HwFastSamplingRndGenerator trn_g = trainUnaryLinearModel(datasetName, trnInsts, tstInsts, alphabet);
			File fm = new File(path);
			File fd = fm.getParentFile();
			if (fd.exists() && fd.isDirectory()) {
				// ok
			} else {
				fd.mkdir();
			}
			UtilFunctions.saveObj(trn_g, path);
			trn_g.testPerformance(trnInsts, alphabet);
			trn_g.testPerformance(tstInsts, alphabet);
			return trn_g;
		} else {
			HwFastSamplingRndGenerator gnr = (HwFastSamplingRndGenerator)obj;
			gnr.testPerformance(trnInsts, alphabet);
			gnr.testPerformance(tstInsts, alphabet);
			return gnr;
		}
	}
	
	public static void initParams(SLParameters para) {
		
		para.LEARNING_MODEL = LearningModelType.L2LossSSVM;

		para.L2_LOSS_SSVM_SOLVER_TYPE = L2LossSSVMLearner.SolverType.DCDSolver;

		para.NUMBER_OF_THREADS = 1;
		para.C_FOR_STRUCTURE = 0.1f;
		para.TRAINMINI = false;
		para.TRAINMINI_SIZE = 1000;
		para.STOP_CONDITION = 0.01f;
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
	
	public static HwFastSamplingRndGenerator trainUnaryLinearModel(String datasetName, List<HwInstance> trnInsts, List<HwInstance> tstInsts, String[] alphabet) {

		
		try {

			List<HwInstance> instSegs = instancesToSegInstances(trnInsts);
			System.out.println("Segments: " + instSegs.size());
			
			SLProblem spTrain = HwDataReader.ExampleListToSLProblem(instSegs);
			System.out.println(spTrain.size());

			//////////////////////////////////////////////////////////////////////
			// train
			SLModel model = new SLModel();

			// initialize the inference solver
			HwSegFeaturizer fg = new HwSegFeaturizer(alphabet, HwFeaturizer.HwSingleLetterFeatLen, false, false, false);
			model.infSolver = new HwSegInferencer(fg.getFeaturizer());
			model.featureGenerator = fg;

			SLParameters para = new SLParameters();
			initParams(para);
			para.TOTAL_NUMBER_FEATURE = fg.getFeatLen();

			Learner learner = LearnerFactory.getLearner(model.infSolver, fg, para);
			model.wv = learner.train(spTrain);
			
			//System.out.println(model.wv.toString());
			
			/////////////////////////////////////
			HwFastSamplingRndGenerator genr = new HwFastSamplingRndGenerator(alphabet.length, model.wv);
			
			///// quick test
			System.out.println("==== Test Logistic Model on TrainSet ====");
			testLogisticModel(trnInsts, alphabet, genr);
			System.out.println("");
			System.out.println("==== Test Logistic Model on TestSet ====");
			testLogisticModel(tstInsts, alphabet, genr);
			
			return genr;
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return null; // should not reach here ...
	}

	public static void testLogisticModel(List<HwInstance> images, String[] alphabet, HwFastSamplingRndGenerator genr) {
		
		try {

			double total = 0;
			double acc = 0;
			double avgTruAcc = 0;

			for (int i = 0; i < images.size(); i++) {

				HwOutput gold = images.get(i).getGoldOutput();
				HwOutput prediction = new HwOutput(gold.size(), images.get(i).alphabet);
	
				double[][] probsAllSegs = genr.predictOnInstance(images.get(i));
				
				List<HwSegment> segs = images.get(i).letterSegs;
				for (int j = 0; j < segs.size(); j++) {
					int predv = -1;
					double maxProb = -1;
					double subProb = 0;
					double[] probs = probsAllSegs[j];//logistic.distributionForInstance(ins);
					
					//System.out.print("Probs = {");
					for (int k = 0; k < probs.length; k++) {
						subProb += probs[k];
						if (probs[k] > maxProb) {
							maxProb = probs[k];
							predv = k;
						}
						//System.out.print(probs[k] + ", ");
					}
					//System.out.println("}");
					prediction.setOutput(j, predv);
				}

				for (int j = 0; j < prediction.output.length; j++) {
					total += 1.0;
					if (prediction.output[j] == gold.output[j]){
						acc += 1.0;
					}
				}
			}

			avgTruAcc = avgTruAcc / total;
			double accuracy = acc / total;
			System.out.println("CorrCnt = " + acc + " / " + total + " = " + accuracy);

	
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}

	
	
	public static List<HwInstance> instancesToSegInstances(List<HwInstance> insts) {
		ArrayList<HwInstance> segInsts = new ArrayList<HwInstance>();
		for (HwInstance inst : insts) {
			List<HwSegment> segs = inst.letterSegs;
			for (HwSegment seg : segs) {
				seg.goldIndex = SeqSamplingRndGenerator.getGoldValueIdx(inst.alphabet, seg.letter);
				List<HwSegment> singleSeg = new ArrayList<HwSegment>();
				singleSeg.add(seg);
				HwInstance singleInst = new HwInstance(singleSeg, inst.alphabet);
				segInsts.add(singleInst);
			}
		}
		return segInsts;
	}
	

	public IFeatureVector getSegFeature(HwSegment seg, String[] alphabet, int valIdx) {
		HwInstance singlex = HwInferencer.getSingleSegInstance(seg, alphabet);
		HwOutput singley = new HwOutput(1, alphabet);
		singley.setOutput(0, valIdx);
		IFeatureVector fv = ufeaturizer.getFeatureVector(singlex, singley);
		return fv;
	}


	@Override
	public InitType getType() {
		return InitType.LOGISTIC_INIT;
	}
	
	
	public Random getRandom() {
		return random;
	}
	
	public WeightVector getUnaryWght() {
		return unaryWeight;
	}
	public HwFeaturizer getFeaturizer() {
		return ufeaturizer;
	}
	//public int getDomainSize() {
	//	return domainSize;
	//}
	
	/////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////
	
	public static void main(String[] args) {
		
		try {
			
			//evaluator.evaluate(testInsts, slmodel, labels, true);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Override
	public void testPerformance(List<HwInstance> insts, String[] alphabet) {
		System.out.println("==== Test Logistic Model on TestSet ====");
		testLogisticModel(insts, alphabet, this);
		System.out.println("==== Histgram ====");
		computeHistg(insts, alphabet);
	}

	public static class ArrCmp implements Comparator<Double> {
		@Override
		public int compare(Double o1, Double o2) {
			if (o1 > o2) {
				return -1;
			} else if (o1 < o2) {
				return 1;
			}
			return 0;
		}
	}

	public void computeHistg(List<HwInstance> images, String[] alphabet) {
		
		try {

			double total = 0;
			double acc = 0;
			double avgTruAcc = 0;
			
			double[] probHist = new double[alphabet.length];
			Arrays.fill(probHist, 0);

			for (int i = 0; i < images.size(); i++) {

				HwOutput gold = images.get(i).getGoldOutput();
				HwOutput prediction = new HwOutput(gold.size(), images.get(i).alphabet);
	
				double[][] probsAllSegs = predictOnInstance(images.get(i));
				
				List<HwSegment> segs = images.get(i).letterSegs;
				for (int j = 0; j < segs.size(); j++) {
		
					double[] probs = probsAllSegs[j];
					double[] pcopy = Arrays.copyOf(probs, probs.length);
					
					Arrays.sort(pcopy);//, new ArrCmp());
					
					for (int k = 0; k < pcopy.length; k++) {
						int k2 = pcopy.length - 1 - k;
						probHist[k] += pcopy[k2];
					}
				}

				for (int j = 0; j < prediction.output.length; j++) {
					total += 1.0;
				}
			}
			
			for (int j = 0; j < probHist.length; j++) {
				probHist[j] /= total;
			}

			for (int j = 0; j < probHist.length; j++) {
				int rk = j + 1;
				System.out.println(rk + "th: " + probHist[j]);
			}

	
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
	
	

}
