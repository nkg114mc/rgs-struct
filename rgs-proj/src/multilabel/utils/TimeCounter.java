package multilabel.utils;

public class TimeCounter {
	
	long tStart;
	
	public TimeCounter() {
		tStart = -1;
	}
	
	public void startTimer() {
		tStart = System.currentTimeMillis();
	}
	
	public static double computeElpseTime(long tStart, long tEnd) {
		long tDelta = tEnd - tStart;
		double elapsedSeconds = tDelta / 1000.0;
		return elapsedSeconds;
	}
	
	public void printTimeCount() {
		double t = getElpseTime();
		System.out.println("Time consume: " + t + " s.");
	}
	
	public double getElpseTime() {
		if (tStart != -1) {
			long tEnd = System.currentTimeMillis();
			long tDelta = tEnd - tStart;
			double elapsedSeconds = tDelta / 1000.0;
			return elapsedSeconds;
		} else {
			System.err.println("Stop watch has not been started!");
			return 0;
		}
	}

}
