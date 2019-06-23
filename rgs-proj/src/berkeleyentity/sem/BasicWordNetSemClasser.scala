package berkeleyentity.sem

import berkeleyentity.WordNetInterfacer
import berkeleyentity.coref.Mention

@SerialVersionUID(1L)
class BasicWordNetSemClasser extends SemClasser {
  def getSemClass(ment: Mention, wni: WordNetInterfacer): String = {
    SemClass.getSemClassNoNer(ment.headStringLc, wni).toString
  }
}
