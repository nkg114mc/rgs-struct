package imgseg;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

public class GlobalFeatureFileDump {
	
	public static void main(String[] args) {
		
		String fldr = "/home/mc/workplace/imgseg/dissolve-struct/data/generated/msrc/All";
		File fd = new File(fldr);
		
		String outfd = "/home/mc/workplace/imgseg/dissolve-struct/data/generated/msrc/global_features";
		for (File f : fd.listFiles()) {
			String nm = f.getName();
			if (nm.endsWith(".bmp")) {
				
				String conNm = getConName(nm);
				String outfn = conNm + ".csv";
				
				// output
				File outf = new File(outfd, outfn);
				if (!outf.exists()) {
					try {
						PrintWriter pw;
						pw = new PrintWriter(outf);
						pw.close();
					} catch (FileNotFoundException e) {
						e.printStackTrace();
					}
				}
			}
			
			
			
		}
		
	}
	
	private static String getConName(String nm) {
		String[] arr = nm.split("\\.");
		return arr[0];
	}

}
