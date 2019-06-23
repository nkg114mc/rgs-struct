package berkeleyentity.nersearch;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class Conll2003DataReader {
	
	public static void main(String[] args) {
		loadRead("/home/mc/workplace/rand_search/ner2003/ner/eng.train");
		loadRead("/home/mc/workplace/rand_search/ner2003/ner/eng.testa");
		loadRead("/home/mc/workplace/rand_search/ner2003/ner/eng.testb");
		
		outputAResultExample("/home/mc/workplace/rand_search/ner2003/ner/eng.testb");
	}
	
	
	public static void loadRead(String filePath) {
		
		int[] lengthCount = new int[1000];
		Arrays.fill(lengthCount, 0);
		
		int totalCnt = 0;
		int maxLen = 0;
		int docStartCnt = 0;
		
		int cnt1 = 0;
		int cnt2 = 0;
		
		ArrayList<String> sentCache = new ArrayList<String>();
		
		HashMap<String, Integer> tagTypCnt = new HashMap<String, Integer>();
		
		boolean docStart = false;
		
		BufferedReader br;
		try {
			br = new BufferedReader(new FileReader(filePath));
		    String line = br.readLine();
		    while (line != null) {

		    	if (!line.equals("")) {
		    		sentCache.add(line);
		    		if (line.contains("-DOCSTART-")) {
		    			docStartCnt++;
		    			docStart = true;
		    		} else {
		    			
		    			String[] tks = line.split("\\s+");
		    			assert (tks.length == 4);
		    			
		    			String tg = tks[3];
		    			incCount(tagTypCnt, tg);
		    			
		    		}
		    	} else {
		    		
		    		if (!docStart) {
			    		// done one sentance
			    		totalCnt++;
			    		lengthCount[sentCache.size()]++;
			    		
			    		if (sentCache.size() <= 10) {
			    			cnt1++;
			    		}
			    		if (sentCache.size() >= 20) {
			    			cnt2++;
			    		}

			    		if (maxLen < sentCache.size()) {
			    			maxLen = sentCache.size();
			    		}
		    		}

		    		// new sentence
		    		sentCache.clear();
		    		docStart = false;
		    	}
		    	
		    	line = br.readLine();
		    }
		    br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		for (String s : tagTypCnt.keySet()) {
			System.out.println(s + ": " + tagTypCnt.get(s));
		}
		

		System.out.println("MaxLen = " + maxLen);
		System.out.println("Instan = " + totalCnt);
		System.out.println("DocCnt = " + docStartCnt);
		
		double r1 = ((double)cnt1) / ((double)totalCnt);
		System.out.println("Cnt<=10 = " + r1);
		double r2 = ((double)cnt2) / ((double)totalCnt);
		System.out.println("Cnt>=20 = " + r2);
		for (int j = 1; j <= maxLen; j++) {
			System.out.println("len["+j+"] = " + lengthCount[j]);
		}
		
		    
	}
	
	public static void printOneLine(PrintWriter pw, String line) {
		
		String[] tks = line.split("\\s+");
		if (tks.length == 4) {
			line = tks[0] + " " + tks[1] + " " + tks[3] + " " + tks[3];
		}
		pw.println(line);
	}
	
	public static void outputAResultExample(String filePath) {

		int totalCnt = 0;
		int maxLen = 0;
		int docStartCnt = 0;

		int cnt1 = 0;
		int cnt2 = 0;

		PrintWriter pw;
		try {
			pw = new PrintWriter("example-output.txt");

			ArrayList<String> sentCache = new ArrayList<String>();
			HashMap<String, Integer> tagTypCnt = new HashMap<String, Integer>();

			boolean docStart = false;

			BufferedReader br;
			try {
				br = new BufferedReader(new FileReader(filePath));
				String line = br.readLine();
				while (line != null) {

					printOneLine(pw, line);

					if (!line.equals("")) {
						sentCache.add(line);
						if (line.contains("-DOCSTART-")) {
							docStartCnt++;
							docStart = true;
						} else {

							String[] tks = line.split("\\s+");
							assert (tks.length == 4);

							String tg = tks[3];
							incCount(tagTypCnt, tg);

						}
					} else {

						if (!docStart) {
							// done one sentance
							totalCnt++;

							if (sentCache.size() <= 10) {
								cnt1++;
							}
							if (sentCache.size() >= 20) {
								cnt2++;
							}

							if (maxLen < sentCache.size()) {
								maxLen = sentCache.size();
							}
						}


						// new sentence
						sentCache.clear();
						docStart = false;
					}

					line = br.readLine();
				}
				br.close();
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
			pw.close();

		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
		    
	}

	
	public static void incCount(HashMap<String, Integer> tagTypCnt, String tag) {
		if (tagTypCnt.containsKey(tag)) {
			Integer c = tagTypCnt.get(tag);
			tagTypCnt.put(tag, c.intValue() + 1);
		} else {
			tagTypCnt.put(tag, 1);
		}
	}
	
}