package search;

import java.util.ArrayList;
import java.util.HashMap;

public class SearchHashTable {

	int totalSize = 0;
	int[][] zobristHashKeys;

	// main table
	HashMap<Integer, ArrayList<SearchState>> indexedTable;



	public SearchHashTable(ZobristKeys zbkeys) {
		zobristHashKeys = zbkeys.getZbKeys(); // initialize hashkeys
		indexedTable = new HashMap<Integer, ArrayList<SearchState>>();
	}

	public int computeIndex(SearchState state) {
		int result = 0;
		for (int i = 0; i < state.structOutput.size(); i++) {
			int j = state.structOutput.getOutput(i);
			result = (result ^ zobristHashKeys[i][j]);
		}
		return result;
	}

	public int computeIndexIncreamental(int oldStateIndex, int slotIdx, int oldValIdx, int newValIdx) {
		int result = oldStateIndex;
		result = (result ^ zobristHashKeys[slotIdx][oldValIdx]);
		result = (result ^ zobristHashKeys[slotIdx][newValIdx]);
		return result;
	}


	public void insertNewInstance(SearchState state) {
		int indexValue = computeIndex(state);
		//ArrayList<SearchState> valueList = indexedTable.get(indexValue);
		if (!indexedTable.containsKey(indexValue)) {
			// insert new state
			ArrayList<SearchState> newList = new ArrayList<SearchState>();
			newList.add(state);
			indexedTable.put(indexValue, newList);
		} else {
			ArrayList<SearchState> existList = indexedTable.get(indexValue);
			existList.add(state);
			//indexedTable.put(indexValue, existList);
		}
		totalSize += 1;
	}

	// return true or false
	public boolean probeExistence(SearchState state) {
		int indexValue = computeIndex(state);
		ArrayList<SearchState> valueList = indexedTable.get(indexValue);

		boolean isExist = false;
		if (valueList == null) {
			isExist = false;
		} else {
			for (SearchState everyEntry : valueList) {
				if (everyEntry.isEqualOutput(state)) {
					isExist = true;
					break;
				}
			}
		}
		return isExist;
	}

	public boolean probeExistenceWithAction(SearchState state, int oldIndex, SearchAction action) {
		int indexValue = computeIndexIncreamental(oldIndex, action.getSlotIdx(), action.getOldVal(), action.getNewVal());
		ArrayList<SearchState> valueList = indexedTable.get(indexValue);

		boolean isExist = false;
		if (valueList == null) {
			isExist = false;
		} else {
			for (SearchState everyEntry : valueList) {
				if (everyEntry.isEqualOutput(state)) {
					isExist = true;
					break;
				}
			}
		}
		return isExist;
	}
	
	public int getSize() {
		return totalSize;
	}
}

