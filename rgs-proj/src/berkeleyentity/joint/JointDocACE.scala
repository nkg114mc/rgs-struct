package berkeleyentity.joint

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap

import berkeleyentity.Chunk
import berkeleyentity.ConllDoc
import berkeleyentity.coref.DocumentGraph
import berkeleyentity.coref.Mention
import berkeleyentity.wiki._
import edu.berkeley.nlp.futile.util.Logger

class JointDocACE(val rawDoc: ConllDoc,
                  val docGraph: DocumentGraph,
                  val goldWikiChunks: Seq[Seq[Chunk[Seq[String]]]]) {
  
  val goldChunks = (0 until docGraph.corefDoc.rawDoc.numSents).map(sentIdx => {
    // Only take the part that's actually the type from each one, "GPE", "LOC", etc.
    docGraph.corefDoc.goldMentions.filter(_.sentIdx == sentIdx).map(ment => new Chunk[String](ment.startIdx, ment.endIdx, getGoldLabel(ment)));
  });
  
  def getGoldLabel(ment: Mention) = {
    if (ment.nerString.size >= 3) {
      ment.nerString.substring(0, 3)
    } else {
      "O"  // SHouldn't happen during training
    };
  }
  
  def getGoldWikLabels(ment: Mention): Seq[String] = {
    val matchingChunk = goldWikiChunks(ment.sentIdx).filter(chunk => chunk.start == ment.startIdx && chunk.end == ment.endIdx);
    //if (matchingChunk.size > 0) matchingChunk.head.label else Seq(ExcludeToken);
    if (matchingChunk.size == 1) matchingChunk.head.label else Seq(ExcludeToken);
  }
}

object JointDocACE {
  
  def apply(rawDoc: ConllDoc,
            docGraph: DocumentGraph,
            maybeGoldWikiChunks: Option[Seq[Seq[Chunk[Seq[String]]]]]) = {
    val goldWikiChunks = if (maybeGoldWikiChunks.isDefined) {
      maybeGoldWikiChunks.get
    } else {
      (0 until rawDoc.numSents).map(i => Seq[Chunk[Seq[String]]]());
    }
    new JointDocACE(rawDoc, docGraph, goldWikiChunks);
  }
  
  def assembleJointDocs(docGraphs: Seq[DocumentGraph],
                        goldWikification: CorpusWikiAnnots) = {
    docGraphs.map(docGraph => {
      val rawDoc = docGraph.corefDoc.rawDoc;
      val goldWiki = if (goldWikification.contains(rawDoc.docID)) {
        Some((0 until rawDoc.numSents).map(goldWikification(rawDoc.docID)(_)));
      } else {
        if (!goldWikification.isEmpty) {
          Logger.logss("WARNING: Some Wikification doc entries found, but none for " + rawDoc.docID);
        }
        None;
      }
      JointDocACE(rawDoc, docGraph, goldWiki);
    });
  }
}
