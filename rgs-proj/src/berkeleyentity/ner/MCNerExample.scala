package berkeleyentity.ner

import berkeleyentity.DepConstTree
import berkeleyentity.coref.DocumentGraph
import berkeleyentity.coref.Mention

case class MCNerExample(val words: Seq[String],
                        val poss: Seq[String],
                        val tree: DepConstTree,
                        val startIdx: Int,
                        val headIdx: Int,
                        val endIdx: Int,
                        val goldLabel: String)  {
  var ment: Mention = null;
  var docID: Int = -1;
  def wordAt(i: Int) = NerExample.wordAt(words, i); 
  def posAt(i: Int) = NerExample.posAt(poss, i);
}

object MCNerExample {
  def apply(docGraph: DocumentGraph, idx: Int) = {
    val pm = docGraph.getMention(idx);
    val rawDoc = docGraph.corefDoc.rawDoc;
    new MCNerExample(rawDoc.words(pm.sentIdx), rawDoc.pos(pm.sentIdx), rawDoc.trees(pm.sentIdx), pm.startIdx, pm.headIdx, pm.endIdx, pm.nerString);
  }
}
