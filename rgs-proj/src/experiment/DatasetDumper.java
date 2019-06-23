package experiment;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.List;

import sequence.hw.HwInstance;
import sequence.hw.HwSegment;

public class DatasetDumper {

	public static void main(String[] args) {
		
		CommonDatasetLoader commonDsLdr = new CommonDatasetLoader(false, CommonDatasetLoader.DEFAULT_TRAIN_SPLIT_RATE);
		
		List<List<HwInstance>> ds = commonDsLdr.getHwDs(false, 0);
		List<HwInstance> trnSet = ds.get(0);
		List<HwInstance> tstSet = ds.get(1);

		
		PrintWriter dumpFile;
		try {
			dumpFile = new PrintWriter("/home/mc/workplace/rand_search/lstm-baselines/tf-seq2seq-master/mydata/hw0-large/hw0large-train.txt");
			for (HwInstance ins : trnSet) {
				List<HwSegment> segs = ins.letterSegs;
				for (HwSegment seg : segs) {
					String l = buildLine(seg);
					dumpFile.println(l);
				}
			}
			dumpFile.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

	}
	
	public static String buildLine(HwSegment seg) {
		StringBuilder sb = new StringBuilder();
		
		double[] feat = seg.getFeatArr();
		for (int i = 0; i < feat.length; i++) {
			if (i > 0) {
				sb.append(",");
			}
			sb.append(feat[i]);
		}
		
		sb.append(" ");
		sb.append(seg.goldIndex);
		
		return sb.toString();
	}

}
