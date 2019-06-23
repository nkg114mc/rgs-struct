package berkeleyentity.coref

import berkeleyentity.Driver
import berkeleyentity.ConllDocReader
import berkeleyentity.lang.EnglishCorefLanguagePack
import berkeleyentity.mentions.MentionClassifier

object CorefMentionPredictionTest {
  
  def main(args: Array[String]) {
    runMentionPredictionCheck();
  }

  def runMentionPredictionCheck() {
    
    val path = "/home/mc/workplace/rand_search/coref/berkfiles/data/ontonotes5/test";
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));

    val rawDocs = ConllDocReader.loadRawConllDocsWithSuffix(path, -1, "v9_auto_conll");
    val assembler = new CorefDocAssemblerCopy(new EnglishCorefLanguagePack(), false);
    val corefDocs = rawDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    
    CorefDocAssemblerCopy.checkGoldMentionRecall(corefDocs);
    
    
    val tstExs = MentionClassifier.extractMentInstanncesNoFeature(corefDocs)
    val predMentFile = "tmp1.txt";
    MentionClassifier.dumpMentionPredictionIgnoreLabel(tstExs, predMentFile)
    
  }

  
}