package init;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
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
import imgseg.FracScore;
import imgseg.ImageDataReader;
import imgseg.ImageInstance;
import imgseg.ImageSegEvaluator;
import imgseg.ImageSegFeaturizer;
import imgseg.ImageSegLabel;
import imgseg.ImageSegMain;
import imgseg.ImageSuperPixel;
import imgseg.ImageUnaryInferencer;
import multilabel.utils.UtilFunctions;
import search.SearchState;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;
import sequence.hw.HwSegment;
import weka.classifiers.functions.Logistic;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class ImageSegAlphaGenerator extends RandomStateGenerator {

	private static final long serialVersionUID = 4391055535254125452L;

	public static final String ARFF_DUMP_FOLDER = "./ArffDump";

	private int domainSize;
	private Instances dataStruct;
	private Logistic logisticModel;
	private Random random;
	
	private WeightVector unaryWeight;
	private ImageSegFeaturizer unaryFeaturizer;
	
	private double alpha = 0; // percetage of the variables that fixed as unary-initial values
	                          // alpha = 0:   Pure random
	                          // alpha = 1.0: Fixed unary initial
	

	public class ArrayIndex {
		public int index = -1;
		public double sc = 0;
	}
	
	public ImageSegAlphaGenerator(int dmsz, Instances dsHeader, Logistic logsMd, Random rnd, double alp) {
		domainSize = dmsz;
		dataStruct = dsHeader;
		logisticModel = logsMd;
		random = rnd;
		alpha = alp;
		System.out.println("Create image logistic initializer.");
	}
	
	public ImageSegAlphaGenerator(int dmsz, Instances dsHeader, Logistic logsMd, double alp) {
		domainSize = dmsz;
		dataStruct = dsHeader;
		logisticModel = logsMd;
		random = new Random();
		alpha = alp;
		System.out.println("Create image logistic initializer.");
	}
	
	public ImageSegAlphaGenerator(int dmsz, WeightVector wv, double alp) {
		domainSize = dmsz;
		unaryWeight = wv;
		unaryFeaturizer = null;
		random = new Random();
		alpha = alp;
		System.out.println("Create image linear initializer.");
	}
	
	public HashSet<SearchState> generateRandomInitState(AbstractInstance inst, int stateNum) {
		
		String[] dm = ((HwInstance)inst).alphabet;
		double[][] probs = predictOnInstance(inst);
		
		HashSet<SearchState> genStates = new HashSet<SearchState>();
		for (int i = 0; i < stateNum; i++) {
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
	
	
	
	// pick \alpha * |y| variables as fixed initial
	// assign the (1 - \alpha) * |y| variables as random values
	
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
				assignedValue = SeqSamplingRndGenerator.pickBestWithProbs(probilities[i]);
			} else { // rnd
				assignedValue = UniformRndGenerator.getValueIndexUniformly(dm.length, random);
			}
			output.setOutput(i, assignedValue);
		}
		
		return output;
	}

	
	
	
	
	
	// linear model
	public double[][] predictOnInstance(AbstractInstance abInst) {
		
		double[][] p = new double[abInst.size()][abInst.domainSize()];

		try {
			ImageInstance inst = (ImageInstance) abInst;
			if (unaryFeaturizer == null) {
				unaryFeaturizer = new ImageSegFeaturizer(inst.alphabet, false, false);
			}
			for (int i = 0; i < inst.size(); i++) {
				ImageSuperPixel supixel = inst.getSuPix(inst.letterSegs.get(i).index);
				p[i] = predictWithLinearModel(supixel, inst.alphabet);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return p;
	}
	
	private double[] predictWithLinearModel(ImageSuperPixel supixel, String[] alphabet) {

		double[] sc = new double[alphabet.length];
		for (int i = 0; i < alphabet.length; i++) {
			IFeatureVector fv = unaryFeaturizer.getSuPixFeatureVector(supixel, i);
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
	
	public static ImageSegAlphaGenerator loadGenrIfExist(String path, String datasetName, List<ImageInstance> trnInsts, List<ImageInstance> tstInsts, String[] alphabet, boolean doDebug, int iterMax) {
		Object obj = UtilFunctions.loadObj(path);
		if (obj == null) {
			// retrain
			//ImageSegSamplingRndGenerator trn_g = trainLogisticModel(datasetName, trnInsts, tstInsts, alphabet, doDebug,iterMax);
			ImageSegAlphaGenerator trn_g = trainUnaryLinearModel(datasetName, trnInsts, tstInsts, alphabet);
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
			ImageSegAlphaGenerator gnr = (ImageSegAlphaGenerator)obj;
			return gnr;
		}
	}
	
/*
	public static ImageSegSamplingRndGenerator trainLogisticModel(String datasetName, List<ImageInstance> trnInsts, List<ImageInstance> tstInsts, String[] alphabet, boolean doDebug, int iterMax) {

		SeqSamplingRndGenerator.checkArffFolder(ARFF_DUMP_FOLDER);
		
		try {
			String fn = ARFF_DUMP_FOLDER + "/" + datasetName + "_train.arff";
			List<ImageSuperPixel> segs = instancesToSegs(trnInsts);
			dumpImageArff(segs, alphabet, fn);
			
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
			
			
			/////////////////////////////////////
			ImageSegSamplingRndGenerator genr = new ImageSegSamplingRndGenerator(alphabet.length, wkInsts, logistic);
			
			
			///// quick test
			System.out.println("==== Test Logistic Model on TrainSet ====");
			testLogisticModel(trnInsts, alphabet, genr);
			System.out.println("");
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
		
		return null; // should reach here ...
	}
*/
	
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
	
	public static ImageSegAlphaGenerator trainUnaryLinearModel(String datasetName, List<ImageInstance> trnInsts, List<ImageInstance> tstInsts, String[] alphabet) {

		
		try {

			SLProblem spTrain = ImageDataReader.ExampleListToSLProblem(trnInsts);

			//////////////////////////////////////////////////////////////////////
			// train
			SLModel model = new SLModel();

			// initialize the inference solver
			ImageSegFeaturizer fg = new ImageSegFeaturizer(alphabet, false, false);// true, true);
			model.infSolver = new ImageUnaryInferencer(fg);
			model.featureGenerator = fg;

			SLParameters para = new SLParameters();
			initParams(para);//para.loadConfigFile(configFilePath);
			para.TOTAL_NUMBER_FEATURE = fg.getFeatLen();

			Learner learner = LearnerFactory.getLearner(model.infSolver, fg, para);
			model.wv = learner.train(spTrain);
			
			/////////////////////////////////////
			ImageSegAlphaGenerator genr = new ImageSegAlphaGenerator(alphabet.length, model.wv, 1);
			
			
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
		
		return null; // should reach here ...
	}

	//public static void testLogisticModel(List<ImageInstance> images, String[] alphabet, Instances header, Logistic logistic) {
	public static void testLogisticModel(List<ImageInstance> images, String[] alphabet, ImageSegAlphaGenerator genr) {
		
		try {
			
			//System.out.println("NumClasses = " + header.numClasses());
			//System.out.println("NumAttrs = " + header.numAttributes());

			double total = 0;
			double acc = 0;
			double avgTruAcc = 0;

			FracScore[] fscores = new FracScore[21];
			FracScore[] gtscs = new FracScore[21];
			for (int j = 0; j < 21; j++) {
				fscores[j] = new FracScore();
				gtscs[j] = new FracScore();
			}

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


				FracScore[] oneResult = ImageSegEvaluator.evaluateOneImage(images.get(i), prediction, alphabet);
				ImageSegEvaluator.accuFracScore(fscores, oneResult);
				//FracScore[] gtResult = ImageSegEvaluator.evaluateOneImageGtPic(images.get(i), prediction, labels, labelSet);
				//ImageSegEvaluator.accuFracScore(gtscs, gtResult);

				for (int j = 0; j < prediction.output.length; j++) {
					total += 1.0;
					if (prediction.output[j] == gold.output[j]){
						acc += 1.0;
					}
				}

				///if (ifDump) {
				//	evaluator.dumpImage(images.get(i), prediction, labels);
				//}
			}

			avgTruAcc = avgTruAcc / total;
			double accuracy = acc / total;
			System.out.println("CorrCnt = " + acc + " / " + total + " = " + accuracy);

			ImageSegEvaluator.printMSRCscore(fscores, alphabet);
	
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}

	
	public static Instance pixelToInst(ImageSuperPixel seg, Instances dtst) {
		double[] ufeat = extractUnaryFeatureWithClass(seg, true);
		Instance inst = new DenseInstance(1, ufeat);
		inst.setDataset(dtst);
		return inst;
	}
	
	public static List<ImageSuperPixel> instancesToSegs(List<ImageInstance> insts) {
		
		ArrayList<ImageSuperPixel> segInsts = new ArrayList<ImageSuperPixel>();
		
		for (ImageInstance inst : insts) {
			List<HwSegment> segs = inst.letterSegs;
			for (HwSegment seg : segs) {
				ImageSuperPixel supixel = inst.getSuPix(seg.index);//seg.goldIndex = SeqSamplingRndGenerator.getGoldValueIdx(inst.alphabet, seg.letter);
				segInsts.add(supixel);
			}
		}
		
		return segInsts;
	}
	
	public static void dumpImageArff(List<ImageSuperPixel> segInsts, String[] alphabet, String fn) {

		try {
			PrintWriter pw = new PrintWriter(fn);
			String names = null;
			
			for (ImageSuperPixel seg : segInsts) {
				if (names == null) {
					names = "";
					
					// title
					pw.println("@relation ImageSegLogisticRegression");
					pw.println();
					
					// attributes
					double[] feat = extractUnaryFeatureWithClass(seg, false);
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
				double[] feat2 = extractUnaryFeatureWithClass(seg, false);
				int labelIndex = seg.getLabel();
				String csvStr = arrToCsvStr(feat2, labelIndex);
				pw.println(csvStr);
			}
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

	}
	
	public static String arrToCsvStr(double[] feat, int lbIdx) {
		String str = "";
		// features
		for (int i = 0; i < feat.length; i++) {
			str += (feat[i] + ",");
		}
		// label
		str += String.valueOf(lbIdx);
		return str;
	}
	
	public static double[] extractUnaryFeatureWithClass(ImageSuperPixel pixel,  boolean includeLabel) {
		
		int flen = pixel.features[0].length * pixel.features.length;
		
		double[] result = null;
		if (includeLabel) {
			result = new double[flen + 1];
			for (int i = 0; i < pixel.features.length; i++) {
				for (int j = 0; j < pixel.features[i].length; j++) {
					result[i * pixel.features[i].length + j] = pixel.features[i][j];
				}
			}
			result[result.length - 1] = pixel.getLabel();
		} else {
			result = new double[flen];
			for (int i = 0; i < pixel.features.length; i++) {
				for (int j = 0; j < pixel.features[i].length; j++) {
					result[i * pixel.features[i].length + j] = pixel.features[i][j];
				}
			}
		}
		
		return result;
	}

	@Override
	public InitType getType() {
		return InitType.LOGISTIC_INIT;
	}
	
	/////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////
	
	public static void main(String[] args) {
		
		try {
			ImageSegLabel[] labels = ImageSegLabel.loadLabelFromFile("../msrc/imageseg_label_color_map.txt");
			String[] labelNames = ImageSegLabel.getStrLabelArr(labels, false);
			String[] labelNamesFull = ImageSegLabel.getStrLabelArr(labels, true);
			
			ImageSegEvaluator.initRgbToLabel(labels);

			ImageDataReader reader = new ImageDataReader("../msrc");
			ImageSegEvaluator evaluator = new ImageSegEvaluator(reader.getDebugFolder());

			//List<ImageInstance> trainInsts = ImageSegMain.loadFromListFile(reader, "../msrc/Train3.txt", labelNames, true);
			List<ImageInstance> trainInsts = ImageSegMain.loadFromListFile(reader, "../msrc/Train-small.txt", labelNames, true);
			//List<ImageInstance> trainInsts = ImageSegMain.loadFromListFile(reader, "../msrc/TrainValidation.txt", labelNames, true);
			//List<ImageInstance> trainInsts = ImageSegMain.loadFromListFile(reader, "../msrc/Train.txt", labelNames, true);


			//String svmCfgFile = "sl-config/msrc21-search-DCD.config";
			//String modelLogsFn = "../logistic_models/msrc21.logistic";
			//String modelSvFn = "../logistic_models/msrc21.ssvm";
			//SLModel slmodel = runLearning(trainInsts, labelNames, InitType.UNIFORM_INIT, 1, svmCfgFile, modelLogsFn, modelSvFn);

			List<ImageInstance> testInsts = ImageSegMain.loadFromListFile(reader, "../msrc/Test.txt", labelNamesFull, true);
			
			//ImageSegSamplingRndGenerator genr = trainLogisticModel("msrc21", trainInsts, testInsts, labelNames, false, -1);
			//ImageSegSamplingRndGenerator genr = trainUnaryLinearModel("msrc21", trainInsts, testInsts, labelNames);
			ImageSegAlphaGenerator genr = trainUnaryLinearModel("msrc21", testInsts, testInsts, labelNames);

			//evaluateUnary(trainInsts, slmodel, labelNames, labels, evaluator, true);
			//evaluateUnary(testInsts, slmodel, labelNames, labels, evaluator, true);
			
			//evaluator.evaluate(testInsts, slmodel, labels, true);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Override
	public void testPerformance(List<HwInstance> insts, String[] alphabet) {
		List<ImageInstance> images = new ArrayList<ImageInstance>();
		for (HwInstance ins : insts) {
			images.add((ImageInstance)ins);
		}

		System.out.println("==== Test Logistic Model on TestSet ====");
		testLogisticModel(images, alphabet, this);
	}



}
