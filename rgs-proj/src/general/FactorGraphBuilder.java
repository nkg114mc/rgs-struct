package general;

import berkeleyentity.mentions.SingletonFactorGraph;
import berkeleyentity.randsearch.AceCorefFactorGraph;
import imgseg.ImageSegFactorGraph;
import multilabel.MultiLabelFactorGraph;
import sequence.hw.HwFactorGraph;

public class FactorGraphBuilder {
	
	public static enum FactorGraphType { NoTypeGraph, SequenceGraph, MultiLabelGraph, ImageSegGraph, HotCorefGraph, Ace05CorefGraph, SingletonGraph };
	
	public static AbstractFactorGraph getFactorGraph(FactorGraphType fgt, AbstractInstance ins, AbstractFeaturizer fzr) {
		
		if (fgt == FactorGraphType.SequenceGraph) {
			return (new HwFactorGraph(ins, fzr));
		} else if (fgt == FactorGraphType.MultiLabelGraph) {
			return (new MultiLabelFactorGraph(ins, fzr));
		} else if (fgt == FactorGraphType.ImageSegGraph) {
			return (new ImageSegFactorGraph(ins, fzr));
		//} else if (fgt == FactorGraphType.HotCorefGraph) {
		//	return (new HotCorefFactorGraph(ins, fzr));
		} else if (fgt == FactorGraphType.Ace05CorefGraph) {
			return (new AceCorefFactorGraph(ins, fzr));
		} else if (fgt == FactorGraphType.SingletonGraph) {
			return (new SingletonFactorGraph(ins, fzr));
		} else {
			return null; // error
		}
		
	}

}
