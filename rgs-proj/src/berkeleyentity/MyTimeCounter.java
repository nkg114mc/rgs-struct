package berkeleyentity;

import java.time.Duration;
import java.time.Instant;

public class MyTimeCounter {
	
	String name;
	Instant t1, t2;
	
	public MyTimeCounter(String nm) {
		name = nm;
		t1 = null;
		t2 = null;
	}
	
	public void start() {
		t1 = Instant.now();
	}
	
	public void end() {
		t2 = Instant.now();
	}
	
	public void printSecond(String str) {
		
		String label = str;
		if (str == null) {
			label = name;
		}
		
		if (t1 == null) {
			System.err.println("  ["+label+"] Time has not been started yet!");
			return;
		}
		if (t2 == null) {
			System.err.println("  ["+label+"] Time has not been ended yet!");
			return;
		}
		
		Duration dura = Duration.between(t1, t2);
		long secnds = dura.getSeconds();
		long mints = dura.toMinutes();
		System.out.println("  ["+label+"] Time elapsed " + mints + " minutes.");
		System.out.println("  ["+label+"] Time elapsed " + secnds + " seconds.");
	}
	
	public long getSeconds() {
		if (t1 == null) {
			System.err.println(" Time has not been started yet!");
			return -1;
		}
		if (t2 == null) {
			System.err.println(" Time has not been ended yet!");
			return -1;
		}
		
		Duration dura = Duration.between(t1, t2);
		long secnds = dura.getSeconds();
		return secnds;
	}
	
	public long getSecondsSnapshot() {
		if (t1 == null) {
			System.err.println(" Time has not been started yet!");
			return -1;
		}
		Instant curT2 = Instant.now();
		
		Duration dura = Duration.between(t1, curT2);
		long secnds = dura.getSeconds();
		return secnds;
	}
	
	public long getMilSecondSnapShot() {
		if (t1 == null) {
			System.err.println(" Time has not been started yet!");
			return -1;
		}
		Instant curT2 = Instant.now();
		
		Duration dura = Duration.between(t1, curT2);
		long msecs = dura.toMillis();
		return msecs;
	}
	
	public long getMinutes() {
		if (t1 == null) {
			System.err.println(" Time has not been started yet!");
			return -1;
		}
		if (t2 == null) {
			System.err.println(" Time has not been ended yet!");
			return -1;
		}
		
		Duration dura = Duration.between(t1, t2);
		long mints = dura.toMinutes();
		return mints;
	}
	
	public static void main(String[] args) {
		MyTimeCounter mtc = new MyTimeCounter("Train");
		mtc.start();
		
		int c = 0;
		for (long i = 0; i < 30000000000l; i++) {
			int a = 1;
			int b = 2;
			c = a + b;
		}
		mtc.end();
		
		mtc.printSecond(null);
	}
}

