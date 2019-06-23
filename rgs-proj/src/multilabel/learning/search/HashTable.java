package multilabel.learning.search;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

public class HashTable {

	int[][] zobristHashKeys = null;
	HashMap<Integer, ArrayList<OldSearchState>> indexedTable = null;
	HashSet<OldSearchState> allStates = null;
	int totalSize = 0;

	public int size() {
		return allStates.size();//totalSize;
	}
	
	private int[][] initialZobrise(int maxNment, int maxNval) {
		Random rnd = new Random();
		int[][] zbKeys = new int[maxNment][maxNval];
		for (int i = 0; i < maxNment; i++) {
			for (int j = 0; j < maxNval; j++) {
				zbKeys[i][j] = rnd.nextInt(Integer.MAX_VALUE);
			}
		}
		return zbKeys;
	}

	public HashTable(int maxNmention, int maxNvalue) {
		zobristHashKeys = initialZobrise(maxNmention, maxNvalue);
		indexedTable = new HashMap<Integer, ArrayList<OldSearchState>>();
		allStates = new HashSet<OldSearchState>();
		totalSize = 0;
	}
	
	public int computeIndex(OldSearchState state) {
		int result = 0;
		for (int i = 0; i < state.size(); i++) {
			int j = state.getOutput().getValue(i);
			result = (result ^ zobristHashKeys[i][j]);
		}
		return result;
	}
		  
	public void insert(OldSearchState state) {
		int indexValue = computeIndex(state);
		ArrayList<OldSearchState> valueList = indexedTable.get(indexValue);
		if (valueList == null) {
			// insert new state
			ArrayList<OldSearchState> newList = new ArrayList<OldSearchState>();
			newList.add(state);
			indexedTable.put(indexValue, newList);
		} else {
			valueList.add(state);
			indexedTable.put(indexValue, valueList);
		}
		totalSize += 1;
		allStates.add(state);
	}
	
	public void insertAllNoRepeat(HashSet<OldSearchState> states) {
		for (OldSearchState s : states) {
			if (!this.probeExistence(s)) {
				insert(s);
			}
		}
	}
	public void insertAllNoRepeat(ArrayList<OldSearchState> states) {
		for (OldSearchState s : states) {
			if (!this.probeExistence(s)) {
				insert(s);
			}
		}
	}
	
	// return true or false
	public boolean probeExistence(OldSearchState state) {
		int indexValue = computeIndex(state);
		ArrayList<OldSearchState> valueList = indexedTable.get(indexValue);

		boolean isExist = false;
		if (valueList == null) {
			isExist = false;
		} else {
			ArrayList<OldSearchState> existList = valueList;
			for (OldSearchState everyEntry : existList) {
				if (everyEntry.getOutput().isEqual(state.getOutput())) {
					isExist = true;
					break;
				}
			}
		}
		return isExist;
	}

	
	public HashSet<OldSearchState> getAllElementsHashSet() {
		if (totalSize != allStates.size()) {
			throw new RuntimeException("Inequal size: " + totalSize + " != " + allStates.size());
		}
		return allStates;
	}
}
