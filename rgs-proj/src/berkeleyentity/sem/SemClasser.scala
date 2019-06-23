package berkeleyentity.sem

import berkeleyentity.coref.Mention
import berkeleyentity.WordNetInterfacer
import berkeleyentity.coref.CorefDoc

trait SemClasser extends Serializable {
  // We only bother to define these for NOMINAL and PROPER mentions; it shouldn't be
  // called for anything else
  def getSemClass(ment: Mention, wni: WordNetInterfacer): String;
}
