package init;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import experiment.RndLocalSearchExperiment.InitType;
import general.AbstractInstance;
import general.AbstractOutput;
import init.HwFastAlphaGenerator.ArrayIndex;
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

public class SeqAlphaGenerator extends RandomStateGenerator {

	private static final long serialVersionUID = 4884708678131127291L;

	public static final String ARFF_DUMP_FOLDER = "./ArffDump";

	private int domainSize;
	private Instances dataStruct;
	private Logistic logisticModel;
	private Random random;
	
	private double alpha = -1; // percetage of the variables that fixed as unary-initial values
    // alpha = 0:   Pure random
    // alpha = 1.0: Fixed unary initial
	
	public class ArrayIndex {
		public int index = -1;
		public double sc = 0;
	}
	
	

	public SeqAlphaGenerator(int dmsz, Instances dsHeader, Logistic logsMd, Random rnd, double alp) {
		domainSize = dmsz;
		dataStruct = dsHeader;
		logisticModel = logsMd;
		random = rnd;
		alpha = alp;
		System.out.println("Create sequence logistic initializer.");
	}
	
	public SeqAlphaGenerator(int dmsz, Instances dsHeader, Logistic logsMd, double alp) {
		domainSize = dmsz;
		dataStruct = dsHeader;
		logisticModel = logsMd;
		random = new Random();
		alpha = alp;
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
			//AbstractOutput rndout = sampleOneOutput(probs, dm);
			AbstractOutput rndout = sampleAlphaOutput(probs, dm, alpha);
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
	
	
	
	
	
	
	
	
	
	public AbstractOutput sampleAlphaOutput(double[][] probilities, String[] dm, double alfa) {
		
		HwOutput output = new HwOutput(probilities.length, dm);
		
		
		double len1 = ((double)output.size()) * alfa;
		int lenAlpha = (int)len1;
		int lenRnd = output.size() - lenAlpha;
		
		int[] isFixFlags = new int[output.size()];
		
		/// 1. Pick the percentage of 
		ArrayList<Integer> alphaIdxs = new ArrayList<Integer>();
		
		// randomly order
		ArrayList<ArrayIndex> scIdxs = new ArrayList<ArrayIndex>();
		for (int i = 0; i < output.size(); i++) {
			ArrayIndex ai = new ArrayIndex();
			ai.index = i;
			ai.sc = random.nextDouble();
			scIdxs.add(ai);
		}
		Collections.sort(scIdxs, new Comparator<ArrayIndex>() {
            @Override
            public int compare(ArrayIndex lhs, ArrayIndex rhs) {
                // -1 - less than, 1 - greater than, 0 - equal, all inversed for descending
                return lhs.sc > rhs.sc ? -1 : (lhs.sc < rhs.sc) ? 1 : 0;
            }
        });
		
		// pick alpha
		Arrays.fill(isFixFlags, 0);
		for (int i = 0; i < lenAlpha; i++) {
			int idx = scIdxs.get(i).index;
			alphaIdxs.add(idx);
			isFixFlags[idx] = 1;
		}
		
		/*
		//// have a look at
		System.out.print("(");
		for (int j = 0; j < isFixFlags.length; j++) {
			System.out.print(isFixFlags[j] + ",");
		} System.out.print(")");
		System.out.print(" total = " + output.size());
		System.out.print(" lenAlpha = " + lenAlpha);
		System.out.print(" lenRnd = " + lenRnd);
		System.out.println();
		*/
		
		/// 2. Assign values
		
		
		for (int i = 0; i < output.size(); i++) {
			int assignedValue = -1;
			if (isFixFlags[i] > 0) { // fix
				assignedValue = pickBestWithProbs(probilities[i]);
			} else { // rnd
				assignedValue = UniformRndGenerator.getValueIndexUniformly(dm.length, random);
			}
			output.setOutput(i, assignedValue);
		}
		
		return output;
	}
	
/*
	public AbstractOutput sampleOneOutput(double[][] probilities, String[] dm) {

		HwOutput output = new HwOutput(probilities.length, dm);
		for (int i = 0; i < output.size(); i++) {
			int sampledIdx = sampleWithProbs(probilities[i], random);
			output.setOutput(i, sampledIdx);
		}
		
		return output;
	}
*/
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
	
	public static int pickBestWithProbs(double[] probilities) {
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
				Instance wkInst = SeqSamplingRndGenerator.segToInst(inst.letterSegs.get(i), dataStruct);
				p[i] = logisticModel.distributionForInstance(wkInst);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return p;
	}

	
	
	
	
	@Override
	public InitType getType() {
		return InitType.ALPHA_INIT;
	}

	@Override
	public void testPerformance(List<HwInstance> insts, String[] alphabet) {
		System.out.println("==== Test Logistic Model on TestSet ====");
		//testLogisticModel(insts, alphabet, dataStruct, logisticModel);
	}
	

}
