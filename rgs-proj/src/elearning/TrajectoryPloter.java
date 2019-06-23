package elearning;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import general.AbstractLossFunction;
import init.SeqSamplingRndGenerator;
import search.SearchResult;
import search.SearchState;
import search.SearchTrajectory;
import search.loss.LossScore;

public class TrajectoryPloter {
	
	public static final String PLOT_FOLDER = "./../PlotCsv";
	
	private AbstractLossFunction lossFunc;
	
	private int restartNum;
	private int maxTrajLen;
	private int recordedMaxTrajLen;
	private int recordedRestartLen;

	private int stateCurveCnt;
	private int restartCurveCnt;
	
	public double[] bestCostAcc_vs_step;
	public double[] bestCost_vs_step;
	
	public double[] bestCostAcc_vs_restart;
	public double[] bestCost_vs_restart;
	
	private List<List<LossScore>> storedStepAccFracs;
	private List<List<LossScore>> storedRestartAccFracs;
	
	private int totalStep;
	private int totalEstep;
	private int totalCstep;
	
	
	public long[] iterTime_vs_restart;
	public long[] iterAccumTime_vs_restart;
	
	public TrajectoryPloter(int restart, int maxDepth, AbstractLossFunction lossf) {
		
		bestCostAcc_vs_step = new double[maxDepth];
		Arrays.fill(bestCostAcc_vs_step, 0);
		bestCost_vs_step = new double[maxDepth];
		Arrays.fill(bestCost_vs_step, 0);
		
		bestCostAcc_vs_restart = new double[restart];
		Arrays.fill(bestCostAcc_vs_restart, 0);
		bestCost_vs_restart = new double[restart];
		Arrays.fill(bestCost_vs_restart, 0);
		
		restartNum = restart;
		maxTrajLen = maxDepth;
		lossFunc = lossf;
		
		recordedMaxTrajLen = -1; // max length of recorded trajectory
		recordedRestartLen = -1;
		
		
		storedStepAccFracs = new ArrayList<List<LossScore>>();
		storedRestartAccFracs = new ArrayList<List<LossScore>>();
		
		totalStep = 0;
		totalEstep = 0;
		totalCstep = 0;
		

		// about time
		iterTime_vs_restart = new long[restart];
		Arrays.fill(iterTime_vs_restart, 0);
		iterAccumTime_vs_restart = new long[restart];
		Arrays.fill(iterAccumTime_vs_restart, 0);
		
		SeqSamplingRndGenerator.checkArffFolder(PLOT_FOLDER);
	}
	
	public void addOneResult(SearchResult sresult) {
		
		List<SearchTrajectory> c_trajs = sresult.trajectories;
		List<SearchTrajectory> e_trajs = sresult.e_trajs;
		
		//double[] trajAccs = new double[c_trajs.size()];
		//double[] trajCosts = new double[c_trajs.size()];
		
		// for steps
		ArrayList<Double> trajCosts = new ArrayList<Double>();
		ArrayList<LossScore> trajAccFracs = new ArrayList<LossScore>();
		
		// for restarts
		ArrayList<Double> restartCosts = new ArrayList<Double>();
		ArrayList<LossScore> restartAccFracs = new ArrayList<LossScore>();
		
		int eLen = 0;
		int clen = 0;
		int stepLen = 0;
		for (int i = 0; i < c_trajs.size(); i++) {

			// state curve
			double lastCost = -Double.MAX_VALUE;
			LossScore lastAccFrac = null;
			
			// meta-search

			if (e_trajs != null) {
				if (e_trajs.size() > 0) {
					SearchTrajectory e_traj = e_trajs.get(i);
					List<SearchState> e_states = e_traj.getStateList();
					
					if (i == 0) { // first
						e_states = aseSkywalkerww(e_states);
					}
					
					for (int d0 = 0; d0 < e_states.size(); d0++) { // record the same thing as base-search
						
						e_states.get(d0).trueAccFrac.info = "e";
						trajAccFracs.add(e_states.get(d0).trueAccFrac);
						lastAccFrac = e_states.get(d0).trueAccFrac; // remember the last
						assert (lastAccFrac != null);
						
						trajCosts.add((double)e_states.get(d0).score);
						lastCost = e_states.get(d0).score;
						
						//System.out.println("Step(" + d + ") Accuracy = " + states.get(d).trueAcc);
						stepLen++;
						eLen++;
					}
				}
			}


			// base-search

			SearchTrajectory traj = c_trajs.get(i);
			List<SearchState> states = traj.getStateList();
			for (int d = 0; d < states.size(); d++) {
				
				states.get(d).trueAccFrac.info = "c";
				trajAccFracs.add(states.get(d).trueAccFrac);
				lastAccFrac = states.get(d).trueAccFrac; // remember the last
				assert (states.get(d).trueAccFrac != null);
				
				trajCosts.add((double)states.get(d).score);
				lastCost = states.get(d).score;
				
				//System.out.println("Step(" + d + ") Accuracy = " + states.get(d).trueAcc);
				stepLen++;
				clen++;
			}

			restartCosts.add(lastCost);
			restartAccFracs.add(lastAccFrac);
					
			// increase count
			stateCurveCnt++;
		}
		
		if (stepLen > recordedMaxTrajLen) {
			recordedMaxTrajLen = stepLen;
		}
		
		////////////////////////////////////////////////////
		////////////////////////////////////////////////////
		
		
		// for steps
		ArrayList<Double> trajBestCosts = new ArrayList<Double>();
		ArrayList<LossScore> trajBestAccFracs = new ArrayList<LossScore>();
		
		// step curve - pick best
		double maxCostStep = -Double.MAX_VALUE;
		LossScore maxCostAccFracStep = null;
		double e_maxCostStep = -Double.MAX_VALUE;
		LossScore e_maxCostAccFracStep = null;
		for (int i = 0; i < trajCosts.size(); i++) {
			if (inBaseSearchTraj(trajAccFracs.get(i).info)) { // is c
				if (maxCostStep < trajCosts.get(i)) {
					maxCostStep = trajCosts.get(i);
					maxCostAccFracStep = trajAccFracs.get(i);
				}
			} else { // is e
				if (e_maxCostStep < trajCosts.get(i)) {
					e_maxCostStep = trajCosts.get(i);
					e_maxCostAccFracStep = trajAccFracs.get(i);
				}
			}
			
			if (maxCostAccFracStep != null) {
				trajBestCosts.add(maxCostStep);
				trajBestAccFracs.add(maxCostAccFracStep.getSelfCopy());
			} else { // put e_max
				trajBestCosts.add(e_maxCostStep);
				trajBestAccFracs.add(e_maxCostAccFracStep.getSelfCopy());
			}
			
		}
		
		
		// for restarts
		ArrayList<Double> restartBestCosts = new ArrayList<Double>();
		ArrayList<LossScore> restartBestAccFracs = new ArrayList<LossScore>();
		
		// restart curve - pick best
		double maxCostRestart = -Double.MAX_VALUE;
		LossScore maxCostAccFracRestart = null;
		for (int j = 0; j < restartCosts.size(); j++) {
			
			if (inBaseSearchTraj(restartAccFracs.get(j).info)) { // usually true
				if (maxCostRestart < restartCosts.get(j)) {
					maxCostRestart = restartCosts.get(j);
					maxCostAccFracRestart = restartAccFracs.get(j).getSelfCopy();
				}
			} else {
				throw new RuntimeException("meta-search does not has following base-search traj!");
			}
			
			restartBestCosts.add(maxCostRestart);
			restartBestAccFracs.add(maxCostAccFracRestart);
		}

		assert (trajCosts.size() == trajBestCosts.size());
		assert (restartCosts.size() == restartBestCosts.size());
		if (stepLen > maxTrajLen) {
			throw new RuntimeException("Step Len is larger than max: " + stepLen + " > " + maxTrajLen);
		}
		
		// store cost
		//////////////////////////
		double lastCost2 = -Double.NEGATIVE_INFINITY;
		for (int i = 0; i < trajBestCosts.size(); i++) {
			bestCost_vs_step[i] += trajBestCosts.get(i).doubleValue();
			lastCost2 = trajBestCosts.get(i).doubleValue();
		}
		for (int i = trajBestCosts.size(); i < maxTrajLen; i++) {
			bestCost_vs_step[i] += lastCost2;
		}
		//////////////////////////
		for (int j = 0; j < restartBestCosts.size(); j++) {
			bestCost_vs_restart[j] += restartBestCosts.get(j).doubleValue();
		}
		
		totalStep += stepLen;
		totalEstep += eLen;
		totalCstep += clen;
		
		// cache true accuracy
		storedStepAccFracs.add(trajBestAccFracs);
		storedRestartAccFracs.add(restartBestAccFracs);
		
		if (c_trajs.size() > recordedRestartLen) {
			recordedRestartLen = c_trajs.size();
		}
		
		// store time
		//////////////////////////////
		for (int i = 0; i < sresult.iterTime.size(); i++) {
			iterTime_vs_restart[i] = sresult.iterTime.get(i).longValue();
			iterAccumTime_vs_restart[i] = sresult.iterAccumTime.get(i).longValue();
		}
		
		
		restartCurveCnt++;
	}

	public static boolean inBaseSearchTraj(String info) {
		return (info.equals("c"));
	}
	
/*
	public void plotToFile(String fnprefix) {
		
		System.out.println("StepTrajCnt = " + stateCurveCnt);
		System.out.println("TrajMaxLength = " + recordedMaxTrajLen);
		System.out.println("======================");
		System.out.println("InstanceCnt = " + restartCurveCnt);
		System.out.println("RestartCnt = " + recordedRestartLen);
		
		// step Curve
		double[] stepAccCurve = copyAndAvg(bestCostAcc_vs_step, (double)stateCurveCnt);
		double[] stepCostCurve = copyAndAvg(bestCost_vs_step, (double)stateCurveCnt);
		String fn1 = TrajectoryPloter.PLOT_FOLDER + "/" + fnprefix + "_perf_vs_step.csv";
		outputCsvFile(fn1, stepAccCurve, stepCostCurve, recordedMaxTrajLen);
		
		// restart curve
		double[] restartAccCurve = copyAndAvg(bestCostAcc_vs_restart, (double)restartCurveCnt);
		double[] restartCostCurve = copyAndAvg(bestCost_vs_restart, (double)restartCurveCnt);
		String fn2 = TrajectoryPloter.PLOT_FOLDER + "/" + fnprefix + "_perf_vs_restart.csv";
		outputCsvFile(fn2, restartAccCurve, restartCostCurve, recordedRestartLen);
		
	}
*/
	
	public void plotToFile(String fnprefix) {
		
		System.out.println("StepTrajCnt = " + stateCurveCnt);
		System.out.println("TrajMaxLength = " + recordedMaxTrajLen);
		System.out.println("======================");
		System.out.println("InstanceCnt = " + restartCurveCnt);
		System.out.println("RestartCnt = " + recordedRestartLen);
		System.out.println("======================");
		double instCnt = restartCurveCnt;
		double avgTstep = totalStep;
		double avgEstep = totalEstep;
		double avgCstep = totalCstep;
		avgTstep /= instCnt;
		avgEstep /= instCnt;
		avgCstep /= instCnt;
		System.out.println("AvgTotalSteps = " + avgTstep);
		System.out.println("AvgESteps = " + avgEstep);
		System.out.println("AvgCSteps = " + avgCstep);
		
		
		computeMacroAcc();
		
		// step Curve
		double[] stepAccCurve = bestCostAcc_vs_step;
		double[] stepCostCurve = copyAndAvg(bestCost_vs_step, (double)restartCurveCnt);
		String fn1 = TrajectoryPloter.PLOT_FOLDER + "/" + fnprefix + "_perf_vs_step.csv";
		outputCsvFile(fn1, stepAccCurve, stepCostCurve, recordedMaxTrajLen);
		
		// restart curve
		double[] restartAccCurve = bestCostAcc_vs_restart;
		double[] restartCostCurve = copyAndAvg(bestCost_vs_restart, (double)restartCurveCnt);
		String fn2 = TrajectoryPloter.PLOT_FOLDER + "/" + fnprefix + "_perf_vs_restart.csv";
		//outputCsvFile(fn2, restartAccCurve, restartCostCurve, recordedRestartLen);
		outputWithTimeCsvFile(fn2, restartAccCurve, restartCostCurve, iterTime_vs_restart, iterAccumTime_vs_restart, recordedRestartLen);
		
	}
	
	private void computeMacroAcc() {
		
		Arrays.fill(bestCostAcc_vs_step, 0);
		Arrays.fill(bestCostAcc_vs_restart, 0);
		
		//// step accuracy
		
		for (int i = 0; i < recordedMaxTrajLen; i++) {

			List<LossScore> stpAccFracs = new ArrayList<LossScore>();
			for (int j = 0; j < storedStepAccFracs.size(); j++) { // jth instance
				List<LossScore> instAti = storedStepAccFracs.get(j);
				
				LossScore scStepj = null;
				if (i >= instAti.size()) {
					scStepj = instAti.get(instAti.size() - 1); // last
				} else { // i < instAti.size()
					scStepj = instAti.get(i); // ith
				}
				
				stpAccFracs.add(scStepj);
			}
			
			double macroAccStp = lossFunc.computeMacro(stpAccFracs);
			bestCostAcc_vs_step[i] = macroAccStp;
		}

		
		//// restart accuracy
		
		for (int i = 0; i < recordedRestartLen; i++) {

			List<LossScore> restartAccFracs = new ArrayList<LossScore>();
			for (int j = 0; j < storedRestartAccFracs.size(); j++) { // jth instance
				List<LossScore> instAti = storedRestartAccFracs.get(j);
				restartAccFracs.add(instAti.get(i));
			}
			
			double macroAccRestart = lossFunc.computeMacro(restartAccFracs);
			bestCostAcc_vs_restart[i] = macroAccRestart;
		}

	}
	
	private double[] copyAndAvg(double[] curv, double den) {
		double[] cparr = new double[curv.length];
		for (int i = 0; i < curv.length; i++) {
			cparr[i] = curv[i] / den;
		}
		return cparr;
	}
	
	public static void outputCsvFile(String fn, double[] curveAcc, double[] curveCost, int maxLen) {
		try {
			PrintWriter pw = new PrintWriter(fn);
			for (int i = 0; i < maxLen; i++) {
				pw.println(i + "," + curveAcc[i] + "," + curveCost[i]);
				System.out.println(i + "," + curveAcc[i] + "," + curveCost[i]);
			}
			pw.close();
			
			System.out.println("Dump curve to file:" + fn);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	public static void outputWithTimeCsvFile(String fn,
			                                 double[] curveAcc, double[] curveCost, 
			                                 long[] iterTime, long[] iterAccumTime, int maxLen) {
		try {
			PrintWriter pw = new PrintWriter(fn);
			for (int i = 0; i < maxLen; i++) {
				pw.println(i + "," + curveAcc[i] + "," + curveCost[i] + "," + iterTime[i] + "," + iterAccumTime[i]);
				System.out.println(i + "," + curveAcc[i] + "," + curveCost[i] + "," + iterTime[i] + "," + iterAccumTime[i]);
			}
			pw.close();
			
			System.out.println("Dump curve to file:" + fn);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	private List<SearchState> aseSkywalkerww(List<SearchState> e_states) {
		List<SearchState> e_s2 = new ArrayList<SearchState>();
		if (e_states.size() >= 4) {
			for (int i = 0; i < e_states.size(); i += 2) {
				if (i < e_states.size()) {
					e_s2.add(e_states.get(i));
				}
			}
		} else {
			e_s2.addAll(e_states);
		}
		return e_s2;
	}
	
	public static void main(String[] args) {
		
		

	}
	
}
