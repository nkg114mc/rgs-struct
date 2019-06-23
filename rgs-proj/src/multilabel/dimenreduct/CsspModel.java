package multilabel.dimenreduct;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class CsspModel {
	
	int lowDimen = -1;
	int fullDimen = -1;
	
	// model
	int[] pickIdx; // the index starts from 1
	double[][] Vm;
	
	public CsspModel(String vmPath, String pickIdxPath) {
		loadModel(vmPath, pickIdxPath);
	}
	
	public int getLowDim() {
		return lowDimen;
	}
	
	public int getFullDim() {
		return fullDimen;
	}
	
	public int[] getPickIdx() {
		return pickIdx;
	}
	
	public double[][] getVm() {
		return Vm;
	}
	
	public void loadModel(String vmPath, String pickIdxPath) {
	
		// load Vm
		Vm = loadMatrix(vmPath);
		lowDimen = Vm[0].length;
		fullDimen = Vm.length;

		// load pickIdx
		double[][] pickTmp = loadMatrix(pickIdxPath);
		double[] pickTmpFirstLine = pickTmp[0];
		pickIdx = new int[pickTmpFirstLine.length];
		for (int i = 0; i < pickTmpFirstLine.length; i++) {
			pickIdx[i] = (int)(pickTmpFirstLine[i]);
		}
		
		if (lowDimen != pickIdx.length) { // an error!
			throw new RuntimeException("picked length and lowDimen are not consist: " + lowDimen + " != " + pickIdx.length);
		}
		
		System.out.println("Load Vm from " + vmPath);
		System.out.println("Load pickIdx from " + pickIdxPath);
		System.out.println("Reducer reduces dimension from " + fullDimen + " to " + lowDimen);
		//System.out.println();
	}
	
	public double[][] loadMatrix(String fn) {
		
		BufferedReader bufferedReader = null;
		ArrayList<ArrayList<Double>> matrix = new ArrayList<ArrayList<Double>>();

		int lineCnt = 0;
		int elementCnt = 0;
		try {

			String line;
			bufferedReader = new BufferedReader(new FileReader(fn));
			
			lineCnt = 0;
			while ((line = bufferedReader.readLine()) != null) {
				lineCnt++;
				ArrayList<Double> matrixLine = new ArrayList<Double>();
				
				line = line.trim();
				String[] tks = line.split("\\s+");
				for (int i = 0; i < tks.length; i++) {
					matrixLine.add(Double.parseDouble(tks[i]));
				}
				elementCnt = tks.length;
				
				matrix.add(matrixLine);
			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
		// convert
		int m = lineCnt;
		int n = elementCnt;
		
		double[][] result = new double[m][n];
		for (int i = 0; i < matrix.size(); i++) {
			ArrayList<Double> mline = matrix.get(i);
			for (int j = 0; j < mline.size(); j++) {
				result[i][j] = mline.get(j).doubleValue();
			}
		}
		
		System.out.println("Loaded matrix: " + m + "-by-" + n);
		return result;
	}

	public void print() {
		
		System.out.println("Vm = ");
		for (int i = 0; i < Vm.length; i++) {
			double[] mline = Vm[i];
			for (int j = 0; j < mline.length; j++) {
				System.out.print(mline[j] + ",");
			}
			System.out.println();
		}
		
		System.out.println("pickIdx = ");
		for (int i = 0; i < pickIdx.length; i++) {
			System.out.print(pickIdx[i] + ",");
		}
		System.out.println();
	}
	
	public static String[] getDefaultModelPath(String mlcFolder, String name, int k) {
		String[] modelPaths = new String[2];
		modelPaths[0] = mlcFolder + "/" + name + "/V" + String.valueOf(k); // Vm
		modelPaths[1] = mlcFolder + "/" + name + "/pickIdx" + String.valueOf(k); // picked
		return modelPaths;
	}
	
	public static void main(String[] args) {
		
		// fn1 = [DataSet '/V' num2str(M)];
		// save(fn1, 'Vm', '-ascii');
		// fn2 = [DataSet '/pickIdx' num2str(M)];
		// save(fn2, 'picked', '-ascii');

		CsspModel cm = new CsspModel("/home/mc/workplace/large_multi-label/mlc_lsdr-master/medical/V30", "/home/mc/workplace/large_multi-label/mlc_lsdr-master/medical/pickIdx30");
		cm.print();
	}
}
