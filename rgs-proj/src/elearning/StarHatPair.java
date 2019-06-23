package elearning;

import java.util.HashMap;

public class StarHatPair {
	
	public HashMap<Integer, Double> phi_end;
	public HashMap<Integer, Double> phi_real;
	
	public StarHatPair(HashMap<Integer, Double> f_end, HashMap<Integer, Double> f_real) {
		phi_end = f_end;
		phi_real = f_real;
	}

}
