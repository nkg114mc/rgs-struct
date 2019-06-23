package berkeleyentity.coref
import scala.collection.JavaConverters.asScalaBufferConverter
import berkeleyentity.Chunk
import berkeleyentity.Driver;

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashSet
import berkeleyentity.lang.CorefLanguagePack
import berkeleyentity.lang.Language
import edu.berkeley.nlp.futile.util.Logger
import berkeleyentity.lang.EnglishCorefLanguagePack
import berkeleyentity.lang.ChineseCorefLanguagePack
import berkeleyentity.lang.ArabicCorefLanguagePack
import edu.berkeley.nlp.futile.util.Counter
import edu.berkeley.nlp.futile.syntax.Trees.PennTreeRenderer
import berkeleyentity.ConllDoc
import java.io.BufferedReader
import java.io.FileNotFoundException
import java.io.FileReader
import java.util.{HashMap => JHashMap}
import java.util.ArrayList



class MentSpan {
		var docName: String = "";
		var sentIdx: Int = -1;
		var startIdx: Int = -1;
		var endIdx: Int = -1;
		var headIdx: Int = -1;
}

class CorefDocAssemblerFile(val langPack: CorefLanguagePack,
                            val useGoldMentions: Boolean) {
  
  val constiCnt = new Counter[String]();
  val docPredSpanCached = new JHashMap[String, ArrayList[MentSpan]]();
  //loadFile(docPredSpanCached, "/home/mc/workplace/coref2017/HOT-coref/ims-hotcoref-2014-06-06/hotmentions-train.txt");
  //loadFile(docPredSpanCached, "/home/mc/workplace/coref2017/HOT-coref/ims-hotcoref-2014-06-06/hotmentions-test.txt");
  loadFile(docPredSpanCached, "/home/mc/workplace/rand_search/random_search_proj/mentDumpHot0.1.txt");
  
  def createCorefDoc(rawDoc: ConllDoc, propertyComputer: MentionPropertyComputer): CorefDoc = {
    val (goldMentions, goldClustering) = extractGoldMentions(rawDoc, propertyComputer);
    if (goldMentions.size == 0) {
      Logger.logss("WARNING: no gold mentions on document " + rawDoc.printableDocName);
    }
    val predMentions = if (useGoldMentions) goldMentions else extractPredMentions(rawDoc, propertyComputer, goldMentions);
    new CorefDoc(rawDoc, goldMentions, goldClustering, predMentions)
  }
  
  def extractGoldMentions(rawDoc: ConllDoc, propertyComputer: MentionPropertyComputer): (Seq[Mention], OrderedClustering) = {
    CorefDocAssembler.extractGoldMentions(rawDoc, propertyComputer, langPack);
  }
  
  def extractPredMentions(rawDoc: ConllDoc, propertyComputer: MentionPropertyComputer, gms: Seq[Mention]): Seq[Mention] = {
    val protoMentionsSorted = getProtoMentionsSorted(rawDoc, gms);
    val finalMentions = new ArrayBuffer[Mention]();
    for (sentProtoMents <- protoMentionsSorted; protoMent <- sentProtoMents) {
      finalMentions += Mention.createMentionComputeProperties(rawDoc, finalMentions.size, protoMent.sentIdx, protoMent.startIdx, protoMent.endIdx, protoMent.headIdx, Seq(protoMent.headIdx), false, propertyComputer, langPack)
    }
    finalMentions;
  }
  
  private def getProtoMentionsSorted(rawDoc: ConllDoc, gms: Seq[Mention]): Seq[Seq[ProtoMention]] = {
    val docID = rawDoc.getDocNameWithPart()
    val spans = docPredSpanCached.get(docID).asScala.toList
    val mentionExtents = (0 until rawDoc.numSents).map(i => new HashSet[ProtoMention]);
    for (mspan <- spans) {
			val sentSent = mentionExtents(mspan.sentIdx);
			val pment = new ProtoMention(mspan.sentIdx, mspan.startIdx, mspan.endIdx, mspan.headIdx);
			sentSent += pment;
		}
    ////
    val sortedMentList = mentionExtents.map(protoMents => CorefDocAssembler.sortProtoMentionsLinear(new ArrayBuffer[ProtoMention] ++ protoMents));
    sortedMentList;
  }
  
  /*
  def sortProtoMentionsLinear(protoMentions: Seq[ProtoMention]): Seq[ProtoMention] = {
    //protoMentions.sortBy(ment => (ment.sentIdx, ment.headIdx, ment.endIdx, ment.startIdx));
    protoMentions.sortBy(ment => (ment.sentIdx, ment.endIdx, ment.startIdx));
  }
  */
  
  def sortFunc(ment1: ProtoMention, ment2: ProtoMention) = {
    if (ment1.sentIdx < ment2.sentIdx) {
      true;
    } else if (ment1.sentIdx > ment2.sentIdx) {
      false;
    } else if (ment1.sentIdx == ment2.sentIdx) {
      
    }
  }
  
  def loadFile(docPredSpans: JHashMap[String, ArrayList[MentSpan]], inputFile: String) = {
		
		/////////////////////////

		val reader = new FileReader(inputFile);
		val br = new BufferedReader(reader);

		var partCnt: Int = 0;
		var lineCnt: Int = 0;

		var line: String = br.readLine();
		while (line != null) {
			if (!line.equals("")) {
				val ms = new MentSpan();
				val tks = line.split("\\s+");
				ms.docName = tks(0);
				ms.sentIdx = Integer.parseInt(tks(1));
				ms.startIdx = Integer.parseInt(tks(2));
				ms.endIdx = Integer.parseInt(tks(3));
				ms.headIdx = Integer.parseInt(tks(4));
				lineCnt += 1;
				addMentSpan(docPredSpans, ms);
			}
			line = br.readLine();
		}

		br.close();
		reader.close();

		println("Total line = " + lineCnt);
		println("Loaded docs = " + docPredSpans.size());
			
		docPredSpans
	}
	
	def addMentSpan(spanMap: JHashMap[String, ArrayList[MentSpan]], mspan: MentSpan) {
		if (spanMap.containsKey(mspan.docName)) {
			val list = spanMap.get(mspan.docName);
			list.add(mspan);
		} else {
			val list = new ArrayList[MentSpan]();
			list.add(mspan);
			spanMap.put(mspan.docName, list);
		}
	}

}

