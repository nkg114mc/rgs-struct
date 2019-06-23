package multilabel.learning.search;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

public class SearchBeam {
	
	int beamSize = -1; // Uninitialized
	ArrayList<OldSearchState> queue = null;
	
	
	public SearchBeam(int size) {
		beamSize = size;
		queue = new ArrayList<OldSearchState>();
	}
	
	public ArrayList<OldSearchState> getQueueStates() {
		return queue;
	}
	
	public void clearBeam() {
		queue.clear();
		queue = new ArrayList<OldSearchState>();
	}
	
	public void insert(OldSearchState s) {
		queue.add(s);
	}
	
	public void insertAll(ArrayList<OldSearchState> slist) {
		queue.addAll(slist);
	}

	// predict score sorter
	static final Comparator<OldSearchState> PREDICT_ORDER = new Comparator<OldSearchState>() {
		public int compare(OldSearchState s1, OldSearchState s2) {
			if (s1.predScore > s2.predScore) {
				return -1;
			} else if (s1.predScore < s2.predScore) {
				return 1;
			}
			return 0;
		}
	};
	// predict score sorter
	static final Comparator<OldSearchState> TRUESCORE_ORDER = new Comparator<OldSearchState>() {
		public int compare(OldSearchState s1, OldSearchState s2) {
			if (s1.trueAccuracy > s2.trueAccuracy) {
				return -1;
			} else if (s1.trueAccuracy < s2.trueAccuracy) {
				return 1;
			}
			return 0;
		}
	};
	
	private void sortQueue(boolean useTrueLoss) {
		if (useTrueLoss) {
			Collections.sort(queue, TRUESCORE_ORDER);
		} else {
			Collections.sort(queue, PREDICT_ORDER);
		}
	}
	
	// just return the best state
	public OldSearchState pickBest(boolean useTrueLoss) {
		double bestScore = -Double.MAX_VALUE;
		int bestIdx = -1;
		for (int i = 0; i < queue.size(); i++) {
			double score = queue.get(i).predScore;
			if (useTrueLoss) score = queue.get(i).trueAccuracy;
			if (score > bestScore) {
				bestScore = score;
				bestIdx = i;
			}
		}
		
		OldSearchState best = queue.get(bestIdx).getSelfCopy();
		return best;
	}
	
	// return the best, and delete it from beam
	public OldSearchState popBest(boolean useTrueLoss) {
		double bestScore = -Double.MAX_VALUE;
		int bestIdx = -1;
		for (int i = 0; i < queue.size(); i++) {
			double score = queue.get(i).predScore;
			if (useTrueLoss) score = queue.get(i).trueAccuracy;
			
			//System.out.println("Beam(" + i + ").score = " + score);
			
			if (score > bestScore) {
				bestScore = score;
				bestIdx = i;
			}
		}
		
		OldSearchState top = queue.get(bestIdx).getSelfCopy();
		// remove
		queue.remove(bestIdx);
		return top;
	}
	
	public void dropTail(boolean useTrueLoss) {
		
		if (queue.size() > beamSize) {
			// sort all elements in the beam
			sortQueue(useTrueLoss); 
			// remove the elements that ranks lower than beamSize
			int originSize = queue.size();
			for (int i = (originSize - 1); i >= beamSize; i--) {
				queue.remove(i);
			}
			
			//System.out.println(queue.size() + " == " + beamSize);
		}
		
	}
	
	public ArrayList<OldSearchState> getTopK(int topk, boolean useTrueLoss) {
		ArrayList<OldSearchState> results = new ArrayList<OldSearchState>();
		if (queue.size() > beamSize) {
			// sort all elements in the beam
			sortQueue(useTrueLoss); 
			for (int i = 0; i < topk; i++) {
				results.add(queue.get(i));
			}
			//System.out.println(queue.size() + " == " + beamSize);
		} else {
			results.addAll(queue);
		}
		return results;
	}

	public ArrayList<OldSearchState> getAll() {
		ArrayList<OldSearchState> results = new ArrayList<OldSearchState>();
		results.addAll(queue);
		return results;
	}
	
	public ArrayList<OldSearchState> popAll() {
		ArrayList<OldSearchState> results = new ArrayList<OldSearchState>();
		results.addAll(queue);
		
		// clear queue
		clearBeam();
		
		return results;
	}
	

}
