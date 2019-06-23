package search;

import java.util.ArrayList;
import java.util.List;

public class SearchResult {
	
	public SearchState predState;
	public double predScore;
	public double accuarcy;
	
	public List<SearchTrajectory> trajectories; // c-search trajectory (to get y_real)
	public List<Double> restartPredScores;
	
	public List<SearchTrajectory> e_trajs; // e-search trajectory (to find y_end)
	
	public List<Long> iterTime = new ArrayList<Long>();
	public List<Long> iterAccumTime = new ArrayList<Long>();
	
	public int bestRank = -1;
	
	public void addTraj(SearchTrajectory traj) {
		if (trajectories == null) {
			trajectories = new ArrayList<SearchTrajectory>();
		}
		trajectories.add(traj);
	}
	
	public void addPredScore(double sc) {
		if (restartPredScores == null) {
			restartPredScores = new ArrayList<Double>();
		}
		restartPredScores.add(sc);
	}
	
	public SearchTrajectory getUniqueTraj() {
		if (trajectories.size() == 1) {
			return trajectories.get(0);
		} else {
			throw new RuntimeException("Can not return unique traj since it is not unique...");
		}
	}

}
