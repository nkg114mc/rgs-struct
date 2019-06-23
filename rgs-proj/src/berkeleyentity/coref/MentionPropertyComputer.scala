package berkeleyentity.coref

import berkeleyentity.oregonstate.AceMentionTypePredictor
import berkeleyentity.DepConstTree
import berkeleyentity.lang.EnglishCorefLanguagePack
import berkeleyentity.Driver



class MentionPropertyComputer(val maybeNumGendComputer: Option[NumberGenderComputer],
                              val maybeMentTypePredictor: Option[AceMentionTypePredictor] = None) {
  
  // so far for English only
  val langPack = new EnglishCorefLanguagePack();

  def predictAceMentionType(m: Mention): MentionType = {
    val rawDoc = m.rawDoc;
    predictAceMentionType(rawDoc.words(m.sentIdx), rawDoc.pos(m.sentIdx), rawDoc.trees(m.sentIdx), m.startIdx, m.headIdx, m.endIdx,m.nerString);
  }

  def predictAceMentionType(words: Seq[String],
                            poss: Seq[String],
                            tree: DepConstTree,
                            startIdx: Int,
                            headIdx: Int,
                            endIdx: Int,
                            nerStr: String): MentionType = {
    
    
    
    val tp = if (maybeMentTypePredictor != None) {
      val tpredictor = maybeMentTypePredictor.get;
      val goldMtp = Mention.getGoldMEntType(nerStr);
      val tempMentEx = AceMentionTypePredictor.mentionInfoToTypeEx(words, poss,  tree, startIdx, headIdx, endIdx, goldMtp);
      val predTp = tpredictor.predictType(tempMentEx);
      predTp;
    } else {
      //throw new RuntimeException("MTypeUkn...");
      //// simple rules to predict mention type
      val ruleTp = if (endIdx - startIdx == 1 && PronounDictionary.isDemonstrative(words(headIdx))) {
    		MentionType.DEMONSTRATIVE;
    	} else if (endIdx - startIdx == 1 && (PronounDictionary.isPronLc(words(headIdx)) || langPack.getPronominalTags.contains(poss(headIdx)))) {
    		MentionType.PRONOMINAL;
    	} else if (langPack.getProperTags.contains(poss(headIdx)) || (Driver.setProperMentionsFromNER && nerStr != "O")) {
    		MentionType.PROPER;
    	} else {
    		MentionType.NOMINAL;
    	}
      ruleTp;
    }
    
    tp;
  }

}