package init;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import experiment.RndLocalSearchExperiment.InitType;
import general.AbstractInstance;
import general.AbstractOutput;
import imgseg.ImageSegFeaturizer;
import multilabel.utils.UtilFunctions;
import search.SearchState;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;
import sequence.hw.HwSegment;
import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class SeqSamplingRndGenerator extends RandomStateGenerator {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 8082135159107117138L;

	public static final String ARFF_DUMP_FOLDER = "./ArffDump";

	private int domainSize;
	private Instances dataStruct;
	private Logistic logisticModel;
	private Random random;
	

	public SeqSamplingRndGenerator(int dmsz, Instances dsHeader, Logistic logsMd, Random rnd) {
		domainSize = dmsz;
		dataStruct = dsHeader;
		logisticModel = logsMd;
		random = rnd;
		System.out.println("Create sequence logistic initializer.");
	}
	
	public SeqSamplingRndGenerator(int dmsz, Instances dsHeader, Logistic logsMd) {
		domainSize = dmsz;
		dataStruct = dsHeader;
		logisticModel = logsMd;
		random = new Random();
		System.out.println("Create sequence logistic initializer.");
	}
	
	public Instances getWkInstHeader() {
		return dataStruct;
	}
	
	public Logistic getLogisticModel() {
		return logisticModel;
	}

	public HashSet<SearchState> generateRandomInitState(AbstractInstance inst, int stateNum) {
		
		String[] dm = ((HwInstance)inst).alphabet;
		double[][] probs = predictOnInstance(inst);
		
		HashSet<SearchState> genStates = new HashSet<SearchState>();
		for (int i = 0; i < stateNum; i++) {
			AbstractOutput rndout = sampleOneOutput(probs, dm);
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
	
	public AbstractOutput sampleOneOutput(double[][] probilities, String[] dm) {
		
		//if (probilities.length != seqSize) {
		//	throw new RuntimeException(probilities.length + " != " + seqSize);
		//}
		
		HwOutput output = new HwOutput(probilities.length, dm);
		for (int i = 0; i < output.size(); i++) {
			int sampledIdx = sampleWithProbs(probilities[i], random);
			output.setOutput(i, sampledIdx);
		}
		
		return output;
	}
	
	public static int sampleWithProbs(double[] probilities, Random rndm) {
		double totalWeight = 0;
		for (int i = 0; i < probilities.length; ++i) {
			totalWeight += probilities[i];
		}
		////////////////////////////////////////////
		int randomIndex = -1;
		double randomProb = rndm.nextDouble() * totalWeight;
		for (int i = 0; i < probilities.length; ++i) {
			randomProb -= probilities[i];
		    if (randomProb <= 0.0d) {
		        randomIndex = i;
		        break;
		    }
		}
		return randomIndex;
	}
	
	public static int pickBestWithProbs(double[] probilities){//, Random rndm) {
		double bestSc = -Double.MAX_VALUE;
		int bestIndex = -1;
		for (int i = 0; i < probilities.length; ++i) {
		    if (probilities[i] > bestSc) {
		    	bestSc = probilities[i];
		    	bestIndex = i;
		    }
		}
		return bestIndex;
	}
	
	public double[][] predictOnInstance(AbstractInstance abInst) {
		
		double[][] p = new double[abInst.size()][abInst.domainSize()];

		try {
			HwInstance inst = (HwInstance) abInst;
			for (int i = 0; i < inst.size(); i++) {
				Instance wkInst = segToInst(inst.letterSegs.get(i), dataStruct);
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
	
	public static void checkArffFolder(String fdPath) {
		File fd = new File(fdPath);
		if (!fd.exists()) {
			// create folder
			fd.mkdir();
			System.out.println("Create folder: " + fdPath);
		} else if (fd.exists()) {
			if (fd.isDirectory()) {
				// ok
			} else {
				throw new RuntimeException(fd + ": Folder exists and is not a folder!");
			}
		}
	}
	
	
	
	
	
	
	
	public Random getRandom() {
		return random;
	}
	public int getDomainSize() {
		return domainSize;
	}
	
	//// Training
	
	public static SeqSamplingRndGenerator loadGenrIfExist(String path, String datasetName, List<HwInstance> trnInsts, List<HwInstance> tstInsts, String[] alphabet, boolean doDebug, int iterMax) {
		Object obj = UtilFunctions.loadObj(path);
		if (obj == null) {
			// retrain
			SeqSamplingRndGenerator trn_g = trainLogisticModel(datasetName, trnInsts, tstInsts, alphabet, doDebug,iterMax);
			File fm = new File(path);
			File fd = fm.getParentFile();
			if (fd.exists() && fd.isDirectory()) {
				// ok
			} else {
				fd.mkdir();
			}
			UtilFunctions.saveObj(trn_g, path);
			return trn_g;
		} else {
			SeqSamplingRndGenerator gnr = (SeqSamplingRndGenerator)obj;
			return gnr;
		}
	}
	
	public static SeqSamplingRndGenerator trainLogisticModel(String datasetName, List<HwInstance> trnInsts, List<HwInstance> tstInsts, String[] alphabet, boolean doDebug, int iterMax) {

		checkArffFolder(ARFF_DUMP_FOLDER);
		
		try {
			String fn = ARFF_DUMP_FOLDER + "/" + datasetName + "_train.arff";
			List<HwSegment> segs = instancesToSegs(trnInsts);
			dumpArff(segs, alphabet, fn);
			
			FileReader fdr = new FileReader(fn);
			Instances wkInsts = new Instances(fdr);
			wkInsts.setClassIndex(wkInsts.numAttributes() - 1);
			
			
			System.out.println("NumClasses = " + wkInsts.numClasses());
			System.out.println("NumAttrs = " + wkInsts.numAttributes());
			System.out.println("NumInstss = " + wkInsts.numInstances());
			
			// actual training
			Logistic logistic = new Logistic();
			//String[] options = {  };
			//logistic.setOptions(options);
			logistic.setDebug(doDebug);
			logistic.setMaxIts(iterMax);
			System.out.println("==== Start Logistic Training ====");
			System.out.println(logistic.getTechnicalInformation().toString());
			logistic.buildClassifier(wkInsts);
			
			// just keep a header
			wkInsts.delete();
			
			
			///// quick test
			System.out.println("==== Test Logistic Model on TrainSet ====");
			testLogisticModel(trnInsts, alphabet, wkInsts, logistic);
			System.out.println("");
			System.out.println("");
			System.out.println("==== Test Logistic Model on TestSet ====");
			testLogisticModel(tstInsts, alphabet, wkInsts, logistic);
			
			/////////////////////////////////////
			SeqSamplingRndGenerator genr = new SeqSamplingRndGenerator(alphabet.length, wkInsts, logistic);
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


	public static void testLogisticModel(List<HwInstance> trnInsts, String[] alphabet, Instances header, Logistic logistic) {
		
		
		try {
			
			List<HwSegment> segs = instancesToSegs(trnInsts);
			System.out.println("NumClasses = " + header.numClasses());
			System.out.println("NumAttrs = " + header.numAttributes());
			
			
			double acc = 0;
			double total = 0;

			for (int j = 0; j < segs.size(); j++) {
				Instance ins = segToInst(segs.get(j), header);
				//System.out.println("Class = " +ins.classValue());
				int goldv = (int)(ins.classValue());
				int predv = -1;
				double maxProb = -1;
				double subProb = 0;
				double[] probs = logistic.distributionForInstance(ins);
				
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
				
				total += 1;
				if (predv == goldv) {
					acc += 1;
				}
				
			}
			
			double accu = (acc / total);
			System.out.println("CorrCnt = " + acc + " / " + total + " = " + accu);
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
	
/*
	public static void testLogisticModelTemp(List<HwInstance> trnInsts, String[] alphabet) {
		
		
		String fn = "temp_train.arff";
		List<HwSegment> segs = instancesToSegs(trnInsts);
		//dumpCsv(segs, "temp_train.csv");
		dumpArff(segs, alphabet, fn);
		
		Logistic logistic = loadLogisticModel("/home/mc/workplace/rand_search/random_search_proj/logs1.model");//new Logistic();
		
		try {
			FileReader fdr = new FileReader(fn);
			Instances wkInsts = new Instances(fdr);
			wkInsts.setClassIndex(wkInsts.numAttributes() - 1);
			wkInsts.delete();
			
			System.out.println("NumClasses = " + wkInsts.numClasses());
			System.out.println("NumAttrs = " + wkInsts.numAttributes());
			System.out.println("NumInstss = " + wkInsts.numInstances());
			
			
			double acc = 0;
			double total = 0;
			
			int ninst = wkInsts.numInstances();
			//for (int j = 0; j < ninst; j++) {
			//	Instance ins = wkInsts.instance(j);
			for (int j = 0; j < segs.size(); j++) {
			
				Instance ins = segToInst(segs.get(j), wkInsts);
				//System.out.println("Class = " +ins.classValue());
				int goldv = (int)(ins.classValue());
				int predv = -1;
				double maxProb = -1;
				double subProb = 0;
				double[] probs = logistic.distributionForInstance(ins);
				
				System.out.print("Probs = {");
				for (int k = 0; k < probs.length; k++) {
					subProb += probs[k];
					if (probs[k] > maxProb) {
						maxProb = probs[k];
						predv = k;
					}
					System.out.print(probs[k] + ", ");
				}
				System.out.println("}");
				
				if (predv == goldv) {
					acc += 1;
				}
				
				//System.out.println("ProbSum = " + subProb);
				//System.out.println("ProbLen = " + probs.length);
				
			}
			
			System.out.println("CorrCnt = " + acc);
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
*/
	
	
	public static Instance segToInst(HwSegment seg, Instances dtst) {
		double[] srcfeats = seg.getFeatArr();
		double[] feats = new double[srcfeats.length + 1];
		System.arraycopy(srcfeats, 0, feats, 0, srcfeats.length);
		feats[srcfeats.length] = seg.goldIndex;
		
		Instance inst = new DenseInstance(1, feats);
		inst.setDataset(dtst);
		return inst;
	}
	
	public static List<HwSegment> instancesToSegs(List<HwInstance> insts) {
		
		ArrayList<HwSegment> segInsts = new ArrayList<HwSegment>();
		
		for (HwInstance inst : insts) {
			List<HwSegment> segs = inst.letterSegs;
			for (HwSegment seg : segs) {
				seg.goldIndex = getGoldValueIdx(inst.alphabet, seg.letter);
				segInsts.add(seg);
			}
		}
		
		return segInsts;
	}
	
/*
	public static void dumpCsv(List<HwSegment> segInsts, String fn) {
		
		try {
			PrintWriter pw = new PrintWriter(fn);
			String names = null;
			
			for (HwSegment seg : segInsts) {
				if (names == null) {
					names = "";
					for (int j = 0; j < seg.imgDblArr.length; j++) {
						names += "\"feat"+String.valueOf(j+1)+"\",";
					}
					names += "\"class\"";
					pw.println(names);
				}
				String csvStr = segToCsvStr(seg);
				pw.println(csvStr);
			}
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
	}
*/
	
	public static void dumpArff(List<HwSegment> segInsts, String[] alphabet, String fn) {
		
		//fn = fn.toLowerCase();
		//if (!fn.endsWith(".arff")) {
		//	fn = fn + ".arff";
		//}
		
		try {
			PrintWriter pw = new PrintWriter(fn);
			String names = null;
			
			for (HwSegment seg : segInsts) {
				if (names == null) {
					names = "";
					
					// title
					pw.println("@relation MultiClassLogisticRegression");
					pw.println();
					
					// attributes
					double[] feat = seg.getFeatArr();
					for (int j = 0; j < feat.length; j++) {
						String attr = "Feat"+String.valueOf(j+1);
						pw.println("@attribute " + attr + " numeric");
					}
					// label
					String lbstr = "";
					lbstr += "{";
					for (int j = 0; j < alphabet.length; j++) {
						if (j > 0) {
							lbstr += ",";
						}
						lbstr += String.valueOf(j);
					}
					lbstr += "}";
					pw.println("@attribute Class " + lbstr);
					pw.println();
					pw.println("@data");
				}
				String csvStr = segToCsvStr(seg);
				pw.println(csvStr);
			}
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
	}
	
	public static String segToCsvStr(HwSegment hseg) {
		String str = "";
		// features
		double[] feat = hseg.getFeatArr();
		for (int i = 0; i < feat.length; i++) {
			str += (feat[i] + ",");
		}
		// label
		str += String.valueOf(hseg.goldIndex);
		return str;
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
	
	public static Logistic loadLogisticModel(String mdf) {
		Logistic cls = (Logistic) loadClassifier(mdf);
		return cls;
	}
	
	public static Classifier loadClassifier(String mdf) {
		ObjectInputStream ois;
		Classifier cls = null;
		try {
			ois = new ObjectInputStream(new FileInputStream(mdf));
			cls = (Classifier) ois.readObject();
			ois.close();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		
		return cls;
	}
	
	@Override
	public InitType getType() {
		return InitType.LOGISTIC_INIT;
	}

	@Override
	public void testPerformance(List<HwInstance> insts, String[] alphabet) {
		System.out.println("==== Test Logistic Model on TestSet ====");
		testLogisticModel(insts, alphabet, dataStruct, logisticModel);
	}
	

}
