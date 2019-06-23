package init;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import experiment.RndLocalSearchExperiment.InitType;
import general.AbstractInstance;
import general.AbstractOutput;
import multilabel.evaluation.MultiLabelEvaluator;
import multilabel.instance.Example;
import multilabel.instance.Label;
import multilabel.learning.StructOutput;
import multilabel.utils.UtilFunctions;
import search.SearchState;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;
import sequence.hw.HwSegment;
import weka.classifiers.functions.Logistic;
import weka.core.Instance;
import weka.core.Instances;

public class MultiLabelSamplingRndGenerator extends RandomStateGenerator {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3837344407444732340L;
	private int labelSize;
	private int domainSize;
	private Instances dataStruct;
	private List<Logistic> logisticModels;
	private Random random;
	
	public MultiLabelSamplingRndGenerator(int seqsz, int dmsz, Instances dsHeader,  List<Logistic> logsMds, Random rnd) {
		labelSize = seqsz;
		domainSize = dmsz;
		dataStruct = dsHeader;
		logisticModels = logsMds;
		random = rnd;
		if (labelSize != logisticModels.size()) {
			throw new RuntimeException("Labels and model numbers are not consistent!");
		}
		if (dmsz != 2) {
			throw new RuntimeException("Dimension size != " + dmsz);
		}
		System.out.println("Create multi-label logistic initializer.");
	}
	
	public MultiLabelSamplingRndGenerator(int seqsz, int dmsz, Instances dsHeader, List<Logistic> logsMds) {
		labelSize = seqsz;
		domainSize = dmsz;
		dataStruct = dsHeader;
		logisticModels = logsMds;
		random = new Random();
		if (labelSize != logisticModels.size()) {
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
	
	public double[][] predictOnInstance(AbstractInstance abInst) {
		
		double[][] p = new double[abInst.size()][abInst.domainSize()];

		try {
			HwInstance inst = (HwInstance) abInst;
			for (int i = 0; i < inst.size(); i++) {
				Logistic logisticModel = logisticModels.get(i);
				Instance wkInst = SeqSamplingRndGenerator.segToInst(inst.letterSegs.get(i), dataStruct);
				p[i] = logisticModel.distributionForInstance(wkInst);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return p;
	}
	
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	
	public static MultiLabelSamplingRndGenerator loadGenrIfExist(String path, String datasetName, List<HwInstance> trnInsts, List<HwInstance> tstInsts, String[] alphabet, boolean doDebug) {
		Object obj = UtilFunctions.loadObj(path);
		if (obj == null) {
			// retrain
			MultiLabelSamplingRndGenerator trn_g = trainLogisticModel(datasetName, trnInsts, tstInsts, alphabet, doDebug);
			UtilFunctions.saveObj(trn_g, path);
			return trn_g;
		} else {
			MultiLabelSamplingRndGenerator gnr = (MultiLabelSamplingRndGenerator)obj;
			return gnr;
		}
	}
	
	
	//// Training
	
	public static MultiLabelSamplingRndGenerator trainLogisticModel(String datasetName, List<HwInstance> trnInsts, List<HwInstance> tstInsts, String[] alphabet, boolean doDebug) {

		SeqSamplingRndGenerator.checkArffFolder(SeqSamplingRndGenerator.ARFF_DUMP_FOLDER);
		int labelCnt = trnInsts.get(0).size();
		
		
		try {
			
			Instances firstHdr = null;
			List<Instances> headers = new ArrayList<Instances>();
			List<Logistic> logsModels = new ArrayList<Logistic>();
			
			for (int l = 0; l < labelCnt; l++) {
				
				// dump lth label learning
				String fn = SeqSamplingRndGenerator.ARFF_DUMP_FOLDER + "/" + datasetName + "_train_label" + String.valueOf(l) + ".arff";
				List<HwSegment> segs = instancesToSegs(trnInsts, l);
				SeqSamplingRndGenerator.dumpArff(segs, alphabet, fn);
				
				FileReader fdr = new FileReader(fn);
				Instances wkInsts = new Instances(fdr);
				wkInsts.setClassIndex(wkInsts.numAttributes() - 1);
				
				System.out.println("NumClasses = " + wkInsts.numClasses());
				System.out.println("NumAttrs = " + wkInsts.numAttributes());
				System.out.println("NumInstss = " + wkInsts.numInstances());
				
				// actual training
				Logistic logistic = new Logistic();
				logistic.setDebug(doDebug);
				System.out.println("==== Start Logistic Training on Label " + l + "th ====");
				logistic.buildClassifier(wkInsts);
				
				// just keep a header
				wkInsts.delete();
				headers.add(wkInsts);
				if (firstHdr == null) {
					firstHdr = wkInsts;
				}
				
				logsModels.add(logistic);
				
				System.out.println("==== Done Learning ====");
			}
			
			///// quick test
			System.out.println("==== Eval Logistic Model on TrainSet ====");
			testLogisticModel(trnInsts, alphabet, firstHdr, logsModels);
			System.out.println("");
			System.out.println("");
			System.out.println("==== Eval Logistic Model on TestSet ====");
			testLogisticModel(tstInsts, alphabet, firstHdr, logsModels);
			
			/////////////////////////////////////
			MultiLabelSamplingRndGenerator genr = new MultiLabelSamplingRndGenerator(labelCnt, alphabet.length, firstHdr, logsModels);
			return genr;
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return null; // should reach here ...
	}

	public static void testLogisticModel(List<HwInstance> trnInsts, String[] alphabet, Instances header, List<Logistic> logistics) {
		
		
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
	
					Instance ins = SeqSamplingRndGenerator.segToInst(seg, header);
					//System.out.println("Class = " +ins.classValue());
					int goldv = (int)(ins.classValue());
					int predv = -1;
					double maxProb = -1;
					double subProb = 0;
					double[] probs = logistics.get(l).distributionForInstance(ins);

					for (int k = 0; k < probs.length; k++) {
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
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}	
	}
	
/*
	public static Instance segToInst(HwSegment seg, Instances dtst) {
		double[] feats = new double[seg.imgDblArr.length + 1];
		System.arraycopy(seg.imgDblArr, 0, feats, 0, seg.imgDblArr.length);
		feats[seg.imgDblArr.length] = seg.goldIndex;
		Instance inst = new Instance(1, feats);
		inst.setDataset(dtst);
		return inst;
	}
*/
	
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
/*
	public static List<Instance> instancesToWkInsts(List<HwInstance> insts, Instances dsHeader, int whichLabel) {
		ArrayList<Instance> re = new ArrayList<Instance>();
		for (HwInstance inst : insts) {
			mlInstToWkInst(HwInstance inst, Instances dsHeader, int whichLabel)
		}
		return segInsts;
	}
*/
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
	
	public int getLableSize() {
		return labelSize;
	}
	public int getDomainSizne() {
		return domainSize;
	}
	public Instances getWkHeader() {
		return dataStruct;
	}
	public List<Logistic> logisticList() {
		return logisticModels;
	}
	public Random getRandom() {
		return random;
	}
	
	
	@Override
	public InitType getType() {
		return InitType.LOGISTIC_INIT;
	}

	@Override
	public void testPerformance(List<HwInstance> insts, String[] alphabet) {
		System.out.println("==== Test Logistic Model on TestSet ====");
		testLogisticModel(insts, alphabet, dataStruct, logisticModels);
	}
	
}
