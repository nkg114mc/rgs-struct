package berkeleyentity.wiki

import scala.collection.mutable.HashMap
import berkeleyentity.ConllDocReader
import berkeleyentity.coref.CorefDocAssembler
import berkeleyentity.GUtil
import berkeleyentity.coref.MentionPropertyComputer
import berkeleyentity.lang.Language
import edu.berkeley.nlp.futile.LightRunner
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.futile.fig.basic.Indexer
import scala.collection.mutable.ArrayBuffer
import berkeleyentity.Chunk

object ACETester {
  
  // Command line options
  val dataPath = "data/ace05/ace05-all-conll"
  val wikiDBPath = "models/wiki-db-ace.ser.gz"
  val wikiPath = "data/ace05/ace05-all-conll-wiki"
  val useFancyQueryChooser = false;
  
  def printGoldWiki(goldwiki : HashMap[String, DocWikiAnnots]) {
    val allAnno = goldwiki.values;
    for (docwiki <- allAnno) {
      println("===========");
      for (sentwiki <- docwiki.values) {
        for (chuck <- sentwiki) {
          println(chuck.label);
        }
      }
      println("-----------");
    }
  }
    
  def main(args: Array[String]) {
    LightRunner.initializeOutput(ACETester.getClass());
    LightRunner.populateScala(ACETester.getClass(), args);
    val docs = ConllDocReader.loadRawConllDocsWithSuffix(dataPath, -1, "", Language.ENGLISH);
//    val goldWikification = GUtil.load(wikiAnnotsPath).asInstanceOf[CorpusWikiAnnots];
    
    val goldWikification = WikiAnnotReaderWriter.readStandoffAnnotsAsCorpusAnnots(wikiPath);
    printGoldWiki(goldWikification);
    println("wikiPaht = " + wikiPath);
    
    // Detect mentions, which depend on the NER coarse pass
    val assembler = CorefDocAssembler(Language.ENGLISH, true);
    val corefDocs = docs.map(doc => assembler.createCorefDoc(doc, new MentionPropertyComputer(None)));

//     This does super, super well but is probably cheating
//    val wikiDB = GUtil.load(wikiDBPath).asInstanceOf[WikipediaInterface];
//    val trainDataPath = "data/ace05/train";
//    val trainDocs = ConllDocReader.loadRawConllDocsWithSuffix(trainDataPath, -1, "", Language.ENGLISH);
//    val trainCorefDocs = trainDocs.map(doc => assembler.createCorefDoc(doc, new MentionPropertyComputer(None)));
//    val wikifier = new BasicWikifier(wikiDB, Some(trainCorefDocs), Some(goldWikification));
    
    val queryChooser = if (useFancyQueryChooser) {
      GUtil.load("models/querychooser.ser.gz").asInstanceOf[QueryChooser]
    } else {
      val fi = new Indexer[String];
      fi.getIndex("FirstNonempty=true");
      fi.getIndex("FirstNonempty=false");
      new QueryChooser(fi, Array(1F, -1F))
    }
    
    
    val wikiDB = GUtil.load(wikiDBPath).asInstanceOf[WikipediaInterface];
    val wikifier = new BasicWikifier(wikiDB, Some(queryChooser));
    
//    val wikiDB = GUtil.load(wikiDBPath).asInstanceOf[WikipediaInterface];
//    val aceHeads = ACEMunger.mungeACEToGetHeads("data/ace05/ace05-all-copy");
//    val wikifier = new BasicWikifier(wikiDB, None, None, Some(aceHeads));
//    val wikifier: Wikifier = FahrniWikifier.readFahrniWikifier("data/wikipedia/lex.anchor.lowAmbiguity-resolved",
//                                                               "data/wikipedia/simTerms");
    
    
    var recalled = 0;
    for (corefDoc <- corefDocs) {
      val docName = corefDoc.rawDoc.docID
      for (i <- 0 until corefDoc.predMentions.size) {
        val ment = corefDoc.predMentions(i);
        val goldLabel = getGoldWikification(goldWikification(docName), ment)
        val myTitles = if (goldLabel.size >= 1 && goldLabel(0) != NilToken) {
          wikifier.oracleWikify(docName, ment, goldLabel);
          val myTitles2 = wikifier.wikifyGetTitleSet(docName, ment);
          if (containsCorrect(goldLabel, myTitles2)) {
            recalled += 1;
          }
          myTitles2;
          //wikifier
        } else if (goldLabel.size == 1 && goldLabel(0) == NilToken) {
          wikifier.oracleWikifyNil(docName, ment);
        }
        val glstr = if (goldLabel.size > 0) {
          goldLabel(0);
        } else {
          "###non-link###";
        }
        println(ment.spanToString + " => " + glstr);
      }
    }
    Logger.logss("Recalled: " + recalled);
    wikifier.printDiagnostics();
    LightRunner.finalizeOutput();
  }
}
