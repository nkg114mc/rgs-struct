package multilabel.utils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.text.NumberFormat;
import java.util.ArrayList;

import multilabel.instance.Example;
import multilabel.instance.Label;
import multilabel.learning.StructOutput;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class UtilFunctions {
	
	public static void saveArff(Instances trInsts, String fn) {
		ArffSaver saver1 = new ArffSaver();
		saver1.setInstances(trInsts);
		try {
			saver1.setFile(new File(fn));
			saver1.setDestination(new File(fn));
			saver1.writeBatch();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	
	// ground truth
	public static StructOutput getGoldOutputFromExmp(Example example) {
		StructOutput goldOutput = new StructOutput(example.labelDim());
		ArrayList<Label> labels = example.getLabel();
		for (int i = 0; i < example.labelDim(); i++) {
			goldOutput.setValue(labels.get(i).originIndex, labels.get(i).value);
		}
		return goldOutput;
	}
	
	
	public static void printMemoryUsage() {
		Runtime runtime = Runtime.getRuntime();
		NumberFormat frmt = NumberFormat.getInstance();

		double onegb = (1024 * 1024 * 1024);
		
		StringBuilder sb = new StringBuilder();
		double maxMemory = (double)runtime.maxMemory();
		double allocatedMemory = (double)runtime.totalMemory();
		double freeMemory = (double)runtime.freeMemory();

		sb.append("       max memory: " + frmt.format(maxMemory / onegb) + " GB");
		//sb.append("     free memory: " + frmt.format(freeMemory / onegb) + " GB\n");
		sb.append(" allocated memory: " + frmt.format(allocatedMemory / onegb) + " GB");
		//sb.append("  all free memory: " + frmt.format((freeMemory + (maxMemory - allocatedMemory)) / onegb) + " GB");
		
		System.out.println(sb.toString());
	}
	

	public static void saveObj(Object obj, String path) {
		if (!new File(path).getParentFile().canWrite()) {
			throw new RuntimeException("Can't write to " + path); 
		}
		//if (path.endsWith(".gz")) saveGz(obj, path) else saveNonGz(obj, path);
		try {
			FileOutputStream fileOut = new FileOutputStream(path);
			ObjectOutputStream out = new ObjectOutputStream(fileOut);
			out.writeObject(obj);
			System.out.println("Wrote to " + path);
			out.close();
			fileOut.close();
		} catch(Exception e ) {
			throw new RuntimeException(e);
		}
	}
		  
	public static Object loadObj(String path) {
		if (!new File(path).canRead()) {
			//throw new RuntimeException(); 
			System.err.println("Can't read from " + path);
			return null;
		}
		//if (path.endsWith(".gz")) loadGz(path) else loadNonGz(path);
		Object obj = null;
		try {
			FileInputStream fileIn = new FileInputStream(path);
			ObjectInputStream in = new ObjectInputStream(fileIn);
			obj = in.readObject();
			System.out.println("Object read from " + path);
			in.close();
			fileIn.close();
		} catch(Exception e ) {
			throw new RuntimeException(e);
		}
		return obj;
	}

	
}