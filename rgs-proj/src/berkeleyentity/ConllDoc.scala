package berkeleyentity
import edu.berkeley.nlp.futile.syntax.Tree
import java.io.File

case class ConllDocJustWords(val docID: String,
                             val docPartNo: Int,
                             val words: Seq[Seq[String]]) {
  def wordsArrs = words.map(_.toArray).toArray;
}

// rawText should only be used to save trouble when outputting the document
// for scoring; never at any other time!
case class ConllDoc(val docID: String,
                    val docPartNo: Int,
                    val words: Seq[Seq[String]],
                    val pos: Seq[Seq[String]],
                    val trees: Seq[DepConstTree],
                    val nerChunks: Seq[Seq[Chunk[String]]],
                    val corefChunks: Seq[Seq[Chunk[Int]]],
                    val speakers: Seq[Seq[String]]) {
  
  val numSents = words.size;
  
  def uid = docID -> docPartNo;
  
  def fileName = {
    if (docID.contains("/")) {
      docID.substring(docID.lastIndexOf("/") + 1);
    } else {
      docID;
    }
  }
  
  def getDocNameWithPart(): String = {
		val cononical = new File(docID); // to remove extension
		//return new String(cononical.getName() + "-docpart" + docPartNo);docID
		return new String(docID + "-docpart" + docPartNo);
	}
  
  def printableDocName = docID + " (part " + docPartNo + ")";
  
  def isConversation = docID.startsWith("bc") || docID.startsWith("wb");
  
  def getCorrespondingNERChunk(sentIdx: Int, headIdx: Int): Option[Chunk[String]] = ConllDoc.getCorrespondingNERChunk(nerChunks(sentIdx), headIdx);
}

object ConllDoc {
  
  def getCorrespondingNERChunk(nerChunks: Seq[Chunk[String]], headIdx: Int): Option[Chunk[String]] = {
    val maybeChunk = nerChunks.filter(chunk => chunk.start <= headIdx && headIdx < chunk.end);
    if (maybeChunk.size >= 1) Some(maybeChunk.head) else None;
  }
}
