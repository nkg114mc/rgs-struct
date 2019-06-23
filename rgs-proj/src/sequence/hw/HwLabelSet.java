package sequence.hw;

import java.util.HashMap;

import general.AbstractLabelSet;

public class HwLabelSet extends AbstractLabelSet {
	
	//private static HashMap<String, Integer> LetterToIntMap = null;//initMap();
	private static final String[] ALPHABET = {"a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"};

	public String[] getLabels() {
		return ALPHABET;
	}
	
	public static HashMap<String, Integer> initMap(String[] abt) {
		HashMap<String, Integer> ltiMap = new HashMap<String, Integer>();
		for (int i = 0; i < abt.length; i++) {
			ltiMap.put(abt[i], i);
		}
		return ltiMap;
	}

}
