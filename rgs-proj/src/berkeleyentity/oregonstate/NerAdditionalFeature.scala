package berkeleyentity.oregonstate

import berkeleyentity.coref.MentionType
import edu.berkeley.nlp.futile.util.Counter
import berkeleyentity.ner.MCNerFeaturizer
import scala.collection.JavaConverters._
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import berkeleyentity.ner.MCNerExample
import berkeleyentity.ner.NerExample
import edu.berkeley.nlp.futile.fig.basic.Indexer
import scala.collection.mutable.ArrayBuffer
import berkeleyentity.coref.Mention
import berkeleyentity.ner.MCNerFeaturizerBase

class MCNerAddonFeaturizer(val orignalNerFeaturizer: MCNerFeaturizer,
                           val nerAddiFeatr: NerAdditionalFeature)  extends MCNerFeaturizerBase {

   def featurize(ex: MCNerExample, addToIndexer: Boolean): Array[Array[Int]] = {
     orignalNerFeaturizer.featurizeAdd(ex, addToIndexer, nerAddiFeatr);
   }
}

class NerAdditionalFeature(val featureIndexer: Indexer[String],
                           val labelIndexer: Indexer[String],
                           val dicts: Array[Counter[String]]) extends Serializable {

  def featurizeForLabel(ex: MCNerExample, labelIndex: Int, addToIndexer: Boolean): Array[Int] = { //label[feats[]]
    val feats = new ArrayBuffer[String]();
    
    //// single example features
    for (i <- 0 until dicts.length) {
      val dict = dicts(i);
      val words = dict.keySet();
      val mentSpan = NerAdditionalFeature.getMentionSpan(ex.ment);
      if (words.contains(mentSpan)) {
        feats += ("NerDict-" + String.valueOf(i));
      }
    }
    
    /////
    val labelStr = labelIndexer.getObject(labelIndex);
    // combine feats with label
    val featIdxs = new ArrayBuffer[Int]();
    for (feat <- feats) {
       val labeledFeat = feat + ":" + labelStr;
       if (addToIndexer || featureIndexer.contains(labeledFeat)) featIdxs += featureIndexer.getIndex(labeledFeat)
    }
    featIdxs.toArray;
  }
}

object NerAdditionalFeature {
  
  def getMentionSpan(ment: Mention) = {
    ment.spanToStringLc;
    //ment.spanToString.toLowerCase();
  }
  
  def collectNerDictionary(trainExs: Seq[tmpExample], testExs: Seq[tmpExample]) {
    
    val tagIndexer = MCNerFeaturizer.StdLabelIndexer;
    
    val prnCntsTrn = getUnaryCounter(trainExs);
    val prnCntsTst = getUnaryCounter(testExs);
    
    var totalALl = 0;
    var appearAll = 0;
    for (i <- 0 until tagIndexer.size) {
    	var total = 0;
    	var appeared = 0;
      for (tstItem <- prnCntsTst(i).keySet.asScala) {
        total += 1
        totalALl += 1
        if (prnCntsTrn(i).keySet().contains(tstItem)) {
          appeared += 1;
          appearAll += 1;
        }
      }
      
      /*
      println("== " + tagIndexer.getObject(i) + " ========================= " + appeared + " / " + total);
      val lst = prnCntsTst(i).getEntrySet.asScala.toList.sortWith(_.getValue() > _.getValue());
      for (etr <- lst) {
        val wd = etr.getKey();
        val appr = if (prnCntsTrn(i).keySet().contains(wd)) {
          "=================> APPEARED!";
        } else {
          "";
        }
        println(etr + " " + appr);
      }
      */
    }
    
    println("Hit: " + appearAll + " / " + totalALl);
  }
  
  def getUnaryCounter(exs: Seq[tmpExample]) = {
    val tagIndexer = MCNerFeaturizer.StdLabelIndexer;
    
    val prnCnts = new Array[Counter[String]](tagIndexer.size);
    for (i <- 0 until tagIndexer.size) prnCnts(i) = new Counter[String]();
    
    for (ex <- exs) {
      val ment = ex.exmp.ment;
      val mentTyp = ment.mentionType;
      val goldNerTag = NerTesting.getGoldNerTag(ment.nerString);
      val goldTagIdx = tagIndexer.getIndex(goldNerTag);
      val wds = ment.words;
      if (mentTyp == MentionType.PROPER) {

      } else if (mentTyp == MentionType.NOMINAL) {

      } else if (mentTyp == MentionType.PRONOMINAL) {
        for (wd <- wds) {
          prnCnts(goldTagIdx).incrementCount(wd.toLowerCase(), 1.0);
        }
      } else if (mentTyp == MentionType.DEMONSTRATIVE) {  
      
      }
    }
    
    prnCnts;
  }
  
  def lengthCount(exs: Seq[tmpExample]) {
    val tagIndexer = MCNerFeaturizer.StdLabelIndexer;
    
    val lenCnts = new Counter[Int]();
    
    for (ex <- exs) {
      val ment = ex.exmp.ment;
      val mentTyp = ment.mentionType;
      val goldNerTag = NerTesting.getGoldNerTag(ment.nerString);
      val goldTagIdx = tagIndexer.getIndex(goldNerTag);
      val wds = ment.words;
      val wdLen = ment.headIdx - ment.startIdx + 1;//wds.length;
      if (mentTyp == MentionType.PROPER) {
        lenCnts.incrementCount(wdLen, 1.0);
      } else if (mentTyp == MentionType.NOMINAL) {
        //lenCnts.incrementCount(wdLen, 1.0);
      } else if (mentTyp == MentionType.PRONOMINAL) {
        //lenCnts.incrementCount(wdLen, 1.0);
      } else if (mentTyp == MentionType.DEMONSTRATIVE) {  
      
      }
    }
    
    for (etr <- lenCnts.entrySet().asScala) {
        println(etr);
    }
  }
  
  def tokenTagCnt(exs: Seq[tmpExample]) {
    
    val tagIndexer = MCNerFeaturizer.StdLabelIndexer;
    
    val tagCnt = new HashMap[String, HashSet[Int]]();
    
    for (ex <- exs) {
      val ment = ex.exmp.ment;
      val mentTyp = ment.mentionType;
      val goldNerTag = NerTesting.getGoldNerTag(ment.nerString);
      val goldTagIdx = tagIndexer.getIndex(goldNerTag);
      val wds = ment.words;
      val wdLen = ment.headIdx - ment.startIdx + 1;//wds.length;
      if (mentTyp == MentionType.PROPER) {
        //lenCnts.incrementCount(wdLen, 1.0);
        val wstr = ment.spanToStringLc;
        if (tagCnt.contains(wstr)) {
          val newc = tagCnt(wstr) + goldTagIdx;
          tagCnt += (wstr -> newc)
        } else {
            val hc = new HashSet[Int]();
            hc += goldTagIdx;
            tagCnt += (wstr -> hc)
        }
      } else if (mentTyp == MentionType.NOMINAL) {
        //lenCnts.incrementCount(wdLen, 1.0);
      } else if (mentTyp == MentionType.PRONOMINAL) {
        //lenCnts.incrementCount(wdLen, 1.0);
      } else if (mentTyp == MentionType.DEMONSTRATIVE) {  
      
      }
    }
    
    val lenCnts = new Counter[Int]();
    for ((k,v)<- tagCnt) {
      //lenCnts.incrementCount(v.size, 1.0);
      if (v.size == 1) {
        println(k);
      }
    }
    for (etr <- lenCnts.entrySet().asScala) {
        println(etr);
    }
  }
  
  
  def wordTagCnt(exs: Seq[tmpExample]) = {
    
    val tagIndexer = MCNerFeaturizer.StdLabelIndexer;
    
    val dicts = new Array[Counter[String]](tagIndexer.size);
    for (j <- 0 until tagIndexer.size) dicts(j) = new Counter[String]();
    
    
    //val accSet = Set("VEH", "WEA", "GPE", "LOC");
    for (ex <- exs) {
      val ment = ex.exmp.ment;
      val mentTyp = ment.mentionType;
      val goldNerTag = NerTesting.getGoldNerTag(ment.nerString);
      val goldTagIdx = tagIndexer.getIndex(goldNerTag);
      val wds = ment.words;
      //val wdLen = ment.headIdx - ment.startIdx + 1;//wds.length;
      val wdLen = wds.length;
      
      if (wdLen <= 3) {
    	  if (mentTyp == MentionType.PROPER) {
    		  //lenCnts.incrementCount(wdLen, 1.0);
    		  val wstr = ment.spanToStringLc;
    		  dicts(goldTagIdx).incrementCount(wstr, 1.0); 
    	  }
      }

      
    }
    
    var dictCnt = 0;
    for (tag <- tagIndexer.getMap.keySet.asScala) {
      //println("==# " +  tag + " #============");
      val wset = dicts(tagIndexer.getIndex(tag)).keySet().asScala;
      for (etr <- wset) {
        //println(etr);
        dictCnt += 1;
      }
    }
    println("Dict Size = " + dictCnt);
    
    dicts;
  }
  
}