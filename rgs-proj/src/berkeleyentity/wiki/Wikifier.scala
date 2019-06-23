package berkeleyentity.wiki

import berkeleyentity.coref.Mention
import edu.berkeley.nlp.futile.util.Counter

trait Wikifier {

  def wikify(docName: String, ment: Mention): String;
  
  def wikifyGetTitleSet(docName: String, ment: Mention): Seq[String];
  
  def wikifyGetPriorForJointModel(docName: String, ment: Mention): Counter[String];
  
  def oracleWikifyNil(docName: String, ment: Mention);
  
  def oracleWikify(docName: String, ment: Mention, goldTitles: Seq[String]);
  
  def printDiagnostics();
}
