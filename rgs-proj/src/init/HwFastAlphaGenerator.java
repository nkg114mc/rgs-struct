package init;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import edu.illinois.cs.cogcomp.sl.util.IFeatureVector;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import experiment.RndLocalSearchExperiment.InitType;
import general.AbstractInstance;
import general.AbstractOutput;
import imgseg.ImageSegFeaturizer;
import init.ImageSegAlphaGenerator.ArrayIndex;
import search.SearchState;
import sequence.hw.HwFeaturizer;
import sequence.hw.HwInferencer;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;
import sequence.hw.HwSegment;

public class HwFastAlphaGenerator extends RandomStateGenerator {


	/**
	 * 
	 */
	private static final long serialVersionUID = -7935466659753819960L;
	private Random random;
	private double alpha = -1;
	
	private WeightVector unaryWeight;
	private HwFeaturizer ufeaturizer = null;
	
	
	public class ArrayIndex {
		public int index = -1;
		public double sc = 0;
	}
	

	public HwFastAlphaGenerator(int dmsz, WeightVector wv, double alp) {
		unaryWeight = wv;
		random = new Random();
		alpha = alp;
		System.out.println("Create HW alpha linear initializer.");
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
				assignedValue = pickLargestIndex(probilities[i]);//SeqSamplingRndGenerator.pickBestWithProbs(probilities[i]);
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
			int sampledIdx = SeqSamplingRndGenerator.sampleWithProbs(probilities[i], random);
			output.setOutput(i, sampledIdx);
		}
		
		return output;
	}
*/
	
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

	
	

	public IFeatureVector getSegFeature(HwSegment seg, String[] alphabet, int valIdx) {
		HwInstance singlex = HwInferencer.getSingleSegInstance(seg, alphabet);
		HwOutput singley = new HwOutput(1, alphabet);
		singley.setOutput(0, valIdx);
		IFeatureVector fv = ufeaturizer.getFeatureVector(singlex, singley);
		return fv;
	}


	@Override
	public InitType getType() {
		return InitType.ALPHA_INIT;
	}
	
	/////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////
	

	@Override
	public void testPerformance(List<HwInstance> insts, String[] alphabet) {
		System.out.println("==== Test Logistic Model on TestSet ====");
		//testLogisticModel(insts, alphabet, this);
		System.out.println("==== Histgram ====");
		//computeHistg(insts, alphabet);
	}

}
