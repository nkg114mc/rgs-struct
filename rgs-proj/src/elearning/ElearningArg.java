package elearning;

public class ElearningArg {
	
	// switch
	public boolean runEvalLearn;
	
	// about featurizer
	public boolean useFeat2;
	public boolean useFeat3;
	public boolean useFeat4;
	
	public float assignSvmC;
	
	// about evulation learning
	public int elearningIter;
	public boolean considerInstWght;
	
	// about testing
	
	public boolean doEvalTest;
	public int restartNumTest;
	
	
	// multi-time eval
	public int multiRunTesting;
	

	public ElearningArg() {
		
		assignSvmC = -1;
		
		runEvalLearn = true;
		elearningIter = 20;
		considerInstWght = false;
		doEvalTest = true;
		restartNumTest = 20;
		
		useFeat2 = false;//true;
		useFeat3 = false;
		useFeat4 = false;
		
		multiRunTesting = 10;
	}
	
	

}
