package berkeleyentity.randsearch;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import berkeleyentity.ConllDoc;
import berkeleyentity.coref.CorefDoc;
import berkeleyentity.coref.DocumentGraph;
import berkeleyentity.coref.Mention;
import berkeleyentity.coref.PairwiseIndexingFeaturizer;
import berkeleyentity.oregonstate.IndepVariable;
import berkeleyentity.oregonstate.VarValue;
import sequence.hw.HwInstance;
import sequence.hw.HwOutput;
import sequence.hw.HwSegment;

public class AceCorefInstance extends HwInstance {
	
	DocumentGraph docGraph;

	CorefDoc corefDoc;
	ConllDoc rawDoc;
	String  docName;
	boolean addToIdxer;

	List<IndepVariable<Integer>> cdecisions = null;
	
	public HwOutput predictOutput = null;
	

	public AceCorefInstance(DocumentGraph dg, PairwiseIndexingFeaturizer pairwiseIndexingFeaturizer) {
		super(null, dummyDomain(dg.size()));
		docGraph = dg;
		corefDoc = docGraph.corefDoc();
		rawDoc = corefDoc.rawDoc();
		docName = rawDoc.docID();
		addToIdxer = docGraph.addToFeaturizer();

		cdecisions = extractVariables(dg, pairwiseIndexingFeaturizer);
	}
	
	public static String[] dummyDomain(int n) {
		String[] strs = new String[n];
		for (int i = 0; i < n; i++) {
			strs[i] = " (" + i + ")";
		}
		return strs;
	}
	
	public static List<IndepVariable<Integer>> extractVariables(DocumentGraph dg, PairwiseIndexingFeaturizer pairwiseIndexingFeaturizer) {
		////// Coref Coref Coref Coref Coref Coref Coref Coref Coref
		List<IndepVariable<Integer>> docCorefVars = new ArrayList<IndepVariable<Integer>>();

		// featurizing!
		dg.featurizeIndexNonPrunedUseCache(pairwiseIndexingFeaturizer); 

		for (int i = 0; i < dg.size(); i++) {
			Mention ment = dg.getMention(i);
			ArrayList<VarValue<Integer>> corefValArr = new ArrayList<VarValue<Integer>>();
			ArrayList<VarValue<Integer>> corefGoldArr = new ArrayList<VarValue<Integer>>();

			int valueCnt = 0;

			int[] prunedDomain = dg.getPrunedDomain(i, false);
			HashSet<Integer> goldPrunedAntecedents = new HashSet<Integer>(Ace05CorefInterf.docGraphgetGoldAntecedentsUnderCurrentPruning(dg,i));
			for (Integer jj : prunedDomain) {
				int j = jj.intValue();
				boolean correct = goldPrunedAntecedents.contains(j);
				VarValue<Integer> anteValue = new VarValue<Integer>(valueCnt, j, dg.cachedFeats()[i][j], correct);
				valueCnt += 1;

				corefValArr.add(anteValue);
				if (correct) {
					corefGoldArr.add(anteValue);
				}
			}
			//docCorefVars.add(new IndepVariable<Integer>(corefValArr.toArray, corefGoldArr.toArray, 0));
			docCorefVars.add(new IndepVariable<Integer>(Ace05CorefInterf.convertVarValueListtoArr(corefValArr), Ace05CorefInterf.convertVarValueListtoArr(corefGoldArr), corefValArr.get(0)));
		}

		return docCorefVars;
	}
	
	public boolean isCorrectOutput(HwOutput output) {
		boolean isCorrect = true;
		for (int i = 0; i < output.size(); i++) {
			if (!cdecisions.get(i).noCorrect()) {
				if (!cdecisions.get(i).values()[output.getOutput(i)].isCorrect()) {
					isCorrect = false;
					break;
				}
			}
		}
		return isCorrect;
	}
	
	
	
	public int[] getMentPairFeature(int mIdx, int jidx) {
		IndepVariable<Integer> vrbl = cdecisions.get(mIdx);
		int[] feats = vrbl.values()[jidx].feature();
		return feats;
	}

	
	public int getAnteIndex(int mIdx, int jidx) {
		IndepVariable<Integer> vrbl = cdecisions.get(mIdx);
		int ante = vrbl.values()[jidx].value().intValue();
		return ante;
	}
	
	public boolean isCorrectAnteIndex(int mIdx, int jidx) {
		IndepVariable<Integer> vrbl = cdecisions.get(mIdx);
		boolean crr = vrbl.values()[jidx].isCorrect();
		return crr;
	}
	
	public List<IndepVariable<Integer>> getVarList(int mIdx) {
		return cdecisions;
	}

	public int[] getDomainGivenIndex(int mIdx) {
		IndepVariable<Integer> vrbl = cdecisions.get(mIdx);
		int[] domainIdxs = new int[vrbl.values().length];
		for (int i = 0; i < vrbl.values().length; i++) {
			domainIdxs[i] = i;
		}
		return domainIdxs;
	}
	
	public int[] getGoldDomainGivenIndex(int mIdx) {
		IndepVariable<Integer> vrbl = cdecisions.get(mIdx);
		ArrayList<Integer> crrIdxs = new ArrayList<Integer>();
		for (int j = 0; j < vrbl.values().length; j++) {
			if (vrbl.values()[j].isCorrect()) {
				crrIdxs.add(j);
			}
		}
		/////////////////////////
		int[] domainIdxs = new int[crrIdxs.size()];
		for (int i = 0; i < domainIdxs.length; i++) {
			domainIdxs[i] = crrIdxs.get(i);
		}
		return domainIdxs;
	}

	public int size() {
		return cdecisions.size();
	}
	
	public AceCorefInstance(List<HwSegment> segs, String[] albt) {
		super(segs, albt);
		throw (new RuntimeException("Not implemented!"));
	}


}
