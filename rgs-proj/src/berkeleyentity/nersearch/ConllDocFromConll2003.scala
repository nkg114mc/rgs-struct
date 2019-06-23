package berkeleyentity.nersearch

import java.io.File
import java.util.IdentityHashMap
import java.util.ArrayList
import java.io.BufferedReader
import java.io.FileReader
import scala.collection.JavaConverters.asScalaBufferConverter
import scala.collection.JavaConverters.bufferAsJavaListConverter
import scala.collection.JavaConverters.seqAsJavaListConverter
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import berkeleyentity.lang.Language
import berkeleyentity.lang.ModArabicHeadFinder
import berkeleyentity.lang.ModCollinsHeadFinder
import edu.berkeley.nlp.futile.ling.BikelChineseHeadFinder
import edu.berkeley.nlp.futile.syntax.Trees.PennTreeRenderer
import edu.berkeley.nlp.futile.syntax.Tree
import edu.berkeley.nlp.futile.util.Logger
import edu.berkeley.nlp.futile.fig.basic.IOUtils
import edu.berkeley.nlp.futile.ling.AbstractCollinsHeadFinder
import edu.berkeley.nlp.futile.syntax.Trees.PennTreeReader
import berkeleyentity.ner.NerExample
import berkeleyentity.ner.NerFeaturizer

class Conll2003TokenLine {
  var token: String = "-Invalid-";
  var pos: String = "-Invalid-";
  var parse: String = "-Invalid-";
  var ner: String = "-Invalid-";
  
  var normalNer: String = "-Invalid-";
  
  def isDocStart(): Boolean = {
    false
  }
  
  def getOffsetTag() = {
    if (ner.contains("-")) {
      val arr = ner.split("-");
      arr(0);
    } else {
      ner;
    }
  }
  
  def getTypeTag() = {
		if (ner.contains("-")) {
		  val arr = ner.split("-");
		  arr(1);
	  } else {
		  ner;
	  }
  }
}

class Conll2003Sentence(val words: List[Conll2003TokenLine]) {
  
	val OUTSIDE_TAG = "O";
	val INSIDE_TAG = "I"; 
	val BEGIN_TAG = "B"; 
	
	//val END_TAG = "E"; 
	//val SINGLETON_TAG = "S"; 
      
  private def getStart() = {
		  val start = new Conll2003TokenLine();
		  start.token = "<<-START->>";
		  start.pos = "-";
		  start.parse = "-";
		  start.ner = OUTSIDE_TAG;
		  start.normalNer = OUTSIDE_TAG;
		  start;
  }
  private def getEnd() = {
		  val end = new Conll2003TokenLine();
		  end.token = "<<-END->>";
		  end.pos = "-";
		  end.parse = "-";
		  end.ner = OUTSIDE_TAG;
		  end.normalNer = OUTSIDE_TAG;
		  end;
  }
  
  def getBefore(idx: Int) = {
    if (idx > 0) {
      words(idx - 1)
    } else {
      getStart()
    }
  }
  
  def getAfter(idx: Int) = {
    if (idx < (words.size - 1)) {
      words(idx + 1)
    } else {
      getEnd()
    }
  }
  
  def normalizeTags() {
    for (i <- 0 until words.size) {
      val w = words(i)
      val infront = getBefore(i)
      
      val wofftag = w.getOffsetTag()
      val wtyptag = w.getTypeTag()
      val beforeofftag = infront.getOffsetTag()
      val beforetyptag = infront.getTypeTag()
      
      if ( (wofftag.equals(INSIDE_TAG) && beforeofftag.equals(OUTSIDE_TAG)) ||
           (wofftag.equals(INSIDE_TAG) && !beforetyptag.equals(wtyptag)) ) {
          val newNer = BEGIN_TAG + "-" + wtyptag
          w.normalNer = newNer
          //println(infront.ner + "," + w.ner + " ---> " + newNer)
      } else {
        w.normalNer = w.ner
      }
    }
    
/*
    // BIO to BIOES
    for (i <- 0 until words.size) {
       words(i).ner = words(i).normalNer
    }
    for (i <- 0 until words.size) {
      val w = words(i)
      val infront = getBefore(i)
      val behind = getAfter(i)
      
      val wofftag = w.getOffsetTag()
      val wtyptag = w.getTypeTag()
      val beforeofftag = infront.getOffsetTag()
      val beforetyptag = infront.getTypeTag()
      val afterofftag = behind.getOffsetTag()
      val aftertyptag = behind.getTypeTag()
      
      if ( (wofftag.equals(INSIDE_TAG) && !afterofftag.equals(INSIDE_TAG)) ) {
          val newNer = END_TAG + "-" + wtyptag
          w.normalNer = newNer
          //println(w.ner + "," + behind.ner + " ---> " + newNer)
      } else if ( wofftag.equals(BEGIN_TAG) && !afterofftag.equals(INSIDE_TAG) ) { // singleton
         val newNer = SINGLETON_TAG + "-" + wtyptag
          w.normalNer = newNer
          //println(w.ner + "," + behind.ner + " ---> " + newNer)
      } else {
        w.normalNer = w.ner
      }

    }
*/
  }
  
}

class Conll2003Doc(val sentences: List[Conll2003Sentence]) {
  
}


object Conll2003NerInstanceLoader {

  def main(args: Array[String]) {
    Conll2003NerInstanceLoader.loadNerExamples("/home/mc/workplace/rand_search/ner2003/ner/eng.train");
    Conll2003NerInstanceLoader.loadNerExamples("/home/mc/workplace/rand_search/ner2003/ner/eng.testa");
    Conll2003NerInstanceLoader.loadNerExamples("/home/mc/workplace/rand_search/ner2003/ner/eng.testb");
  }
  
  def convertConll2003SentToNerExample(conllSent: Conll2003Sentence) = {
    val wds = conllSent.words.map { x => x.token }.toSeq
    val poss = conllSent.words.map { x => x.pos }.toSeq
    val goldlbs = conllSent.words.map { x => x.normalNer }.toSeq
    new NerExample(wds, poss, goldlbs)
  }
  
  def loadSentExamples(filePath: String) = {
    val docs = loadDocsFromFile(filePath)
    val exs = docs.flatMap { d => d.sentences };
    println("Example count = " + exs.size)
    exs
  }
  
  def loadNerExamples(filePath: String) = {
    val docs = loadDocsFromFile(filePath)
    val exs = docs.flatMap { d => d.sentences };
    val nerExs = exs.map { x => convertConll2003SentToNerExample(x) }
    nerExs.map { x => checkTransition(x) }
    println("Ner Example count = " + nerExs.size)
    nerExs
  }
  
  def checkTransition(nerEx: NerExample) {
    val sz = nerEx.goldLabels.size
    for (i <- 0 until (sz - 1)) {
      val curr = nerEx.goldLabels(i)
      val after = nerEx.goldLabels(i + 1)
      if (!NerFeaturizer.isLegalTransition(curr, after)) {
        throw new RuntimeException("Invalid transition:  " + curr + " ~~> " + after )
      }
    }
  }
  
  
  def loadDocsFromFile(filePath: String) = {
		
		var totalCnt = 0;
		var docStartCnt = 0;
		var docStart: Boolean = false;
		val sentList = new ArrayList[ArrayList[String]](); 
		var sentCache = new ArrayList[String]();
		
		
		
		val br = new BufferedReader(new FileReader(filePath));
		var line: String = br.readLine();
		while (line != null) {

			if (!line.equals("")) {
				sentCache.add(line);
				if (line.contains("-DOCSTART-")) {
					docStartCnt += 1;
					docStart = true;
				} else {

				}
			} else {

				if (!docStart) {
					// done one sentance
					totalCnt += 1;
				}
				// new sentence
				sentList.add(sentCache);
				sentCache = new ArrayList[String]();
				docStart = false;
			}

			line = br.readLine();
		}
		br.close();

		println("Instan = " + totalCnt);
		println("DocCnt = " + docStartCnt);
		val n2 = totalCnt + docStartCnt;
		println("SentList = " + sentList.size() + "==" + n2);
		
		
		//// construct documents
		val docs = new ArrayBuffer[Conll2003Doc]();
		var sentExCached = new ArrayBuffer[Conll2003Sentence]();
		for (sent <- sentList.asScala) {
		  if (isDocStartLine(sent)) {
		    if (sentExCached.size > 0) {
		      // construct a new doc
		      val d = new Conll2003Doc(sentExCached.toList);
		      docs += d;
		      // clear cache
		      sentExCached = new ArrayBuffer[Conll2003Sentence]();
		    }
		  } else {
		    val s = buildConll2003Sentence(sent)
		    s.normalizeTags()
		    sentExCached += s;
		  }
		}
		
		// last doc
		if (sentExCached.size > 0) {
			// construct a new doc
			val d = new Conll2003Doc(sentExCached.toList);
			docs += d;
		}
		
		println("construct doc = " + docs.size);
		docs.toSeq;
	}
  
  def isDocStartLine(sent: ArrayList[String]): Boolean = {
    require(sent.size() > 0)
    if ((sent.size() == 1) && (sent.get(0).contains("-DOCSTART-"))) {
      true;
    } else {
      false
    }
  }
  
  def buildConll2003Sentence(sent: ArrayList[String]): Conll2003Sentence = {
    val tks = sent.asScala.map{ x => buildTokenLine(x) }.toList
    new Conll2003Sentence(tks);
  }
  
  def buildTokenLine(l: String) = {
		  val ss = l.split("\\s+")
			val tkline = new Conll2003TokenLine();
		  require (ss.length == 4)
		  tkline.token = ss(0);
		  tkline.pos = ss(1);
		  tkline.parse = ss(2);
		  tkline.ner = ss(3);

		  tkline.normalNer = "-Invalid-";
		  tkline
  }
  
  def outputResultFile() {
    
  }
  
}
