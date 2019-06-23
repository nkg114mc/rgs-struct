package multilabel.utils;

import java.lang.instrument.Instrumentation;

public class ObjectSizeFetcher {

	private static Instrumentation instrumentation;

	public static void premain(String args, Instrumentation inst) {
		instrumentation = inst;
	}

	public static long getObjectSize(Object o) {
		return instrumentation.getObjectSize(o);
	}

	public static long getObjectSizeGB(Object o) {
		return instrumentation.getObjectSize(o);
	}
	
	//////////////
	
	public static int getArrSize(int[][] arr) {
		int total = 0;
		for (int i = 0; i < arr.length; i++) {
			total += arr[i].length;
		}
		return total;
	}
	
	public static int getArrSize(double[][] arr) {
		int total = 0;
		for (int i = 0; i < arr.length; i++) {
			total += arr[i].length;
		}
		return total;
	}
	
	public static int getArrSize(float[][] arr) {
		int total = 0;
		for (int i = 0; i < arr.length; i++) {
			total += arr[i].length;
		}
		return total;
	}
}
