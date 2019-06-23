package feature;

import sequence.hw.HwFeaturizer;

public class UnaryFactor extends Factor  {
	
	int varibleIndex;
	
	String[] alphabet;
	int domainSize = 0;
	double[] scoreTable = null;
	double[] features = null;
	
	
	public UnaryFactor(int varIndex, String albt[], double[] feats) {
		varibleIndex = varIndex;
		domainSize = albt.length;
		alphabet = albt;
		scoreTable = new double[domainSize];
		features = feats;
	}
	
	public void computeScoreTable(double[] weights) {
		/*
		for (int i = 0; i < domainSize; i++) {
			scoreTable[i] = 0;
			for (int j = 0; j < features.length; j++) {
				String unaryfn = HwFeaturizer.getUnaryFeatName(alphabet, varibleIndex, i);
				int idx = HwFeaturizer.getIndex(unaryfn) ;
				sparseValues.put(idx, unaryFeats[i][j]);
			}
		}
		
		

		
		throw new RuntimeException("Score table is not defined!");*/
	}

}
