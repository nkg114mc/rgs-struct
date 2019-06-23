package berkeleyentity.ranking;

//import java.util.HashMap;

import ciir.umass.edu.eval.Evaluator;
import ciir.umass.edu.learning.DataPoint;
import ciir.umass.edu.learning.RANKER_TYPE;
import ciir.umass.edu.learning.Ranker;
import ciir.umass.edu.learning.RankerFactory;
import ciir.umass.edu.metric.METRIC;
import ciir.umass.edu.metric.MetricScorerFactory;

public  class UMassRankLib {
	
	String[] rType = new String[]{"MART", "RankNet", "RankBoost", "AdaRank", "Coordinate Ascent", "LambdaRank", "LambdaMART", "ListNet", "Random Forests"};
	RANKER_TYPE[] rType2 = new RANKER_TYPE[]{RANKER_TYPE.MART, RANKER_TYPE.RANKNET, RANKER_TYPE.RANKBOOST, RANKER_TYPE.ADARANK, RANKER_TYPE.COOR_ASCENT, RANKER_TYPE.LAMBDARANK, RANKER_TYPE.LAMBDAMART, RANKER_TYPE.LISTNET, RANKER_TYPE.RANDOM_FOREST};

	int rankerType = 4;
	String trainMetric = "ERR@10";
	String testMetric = "ERR@10";
	boolean printIndividual = false;
	private String modelFilePath = "";
	
	/** RankLib main class */
	//private Evaluator eval = null;
	/** Ranker */
	private Ranker ranker = null;
	
	/** to load the ranker */
 	private RankerFactory rFact = new RankerFactory();
	private MetricScorerFactory mFact = new MetricScorerFactory();
	
	public UMassRankLib()
	{
		initialize(); // init
	}
	
	private void initialize()
	{
		rFact = new RankerFactory();
		mFact = new MetricScorerFactory();
		ranker = rFact.createRanker(RANKER_TYPE.MART);
	}
	
	private void setEvaluator(RANKER_TYPE rType, METRIC trainMetric, METRIC testMetric)
	{
		Evaluator.normalize = false;
	}
	
	/** model file */
	public void setModelPath(String path)
	{
		modelFilePath = path;
	}
	public String getModelPath()
	{
		return modelFilePath;
	}
	
	public void loadModelFile(String modelPath)
	{
		System.out.println("Loading ranklib model file: " + modelPath);
		ranker = rFact.loadRankerFromFile(modelPath);
		modelFilePath = modelPath;
	}
	
	public double getRankerScore(DataPoint dp)
	{
		double score = -1;
		if (ranker == null) {
			throw new RuntimeException("ranker has not be initialized yet!");
		}
		score = ranker.eval(dp);
		return score;
	}
	
	/*
	public RankList constructRankList(int rankLength, double[][] featVecs)
	{
		ArrayList<DataPoint> dplist = new ArrayList<DataPoint>();
		
		// input all feature vectors 
		for (int i = 0; i < rankLength; i++) {
			String sampleStr = rankSampleToStr(featVecs[i], 0);
			DataPoint rankSample = new DenseDataPoint(sampleStr);
			dplist.add(rankSample);
		}
		
		RankList rl = new RankList(dplist);
		return rl;
	}*/
	
	public String rankSampleToStr(double[] feat, int label)
	{
		int qid = 1;
		String str = new String("");
		String lstr = new String(Integer.toString(label));
		String qstr = new String("qid:"+qid);
		
		String fstr = new String("");
		for (int i = 0; i < feat.length; i++) {
			fstr = (fstr + " " + (i+1) + ":" + feat[i]);
		}
		
		str = lstr + " " + qstr + fstr;
		return str;
	}

}
