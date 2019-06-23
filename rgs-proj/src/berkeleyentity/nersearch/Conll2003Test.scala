package berkeleyentity.nersearch

import scala.util.Random
import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._
import java.io.FileInputStream
import java.io.ObjectInputStream
import java.io.File
import java.io.FileOutputStream
import java.io.ObjectOutputStream
import edu.berkeley.nlp.futile.fig.basic.Indexer
import berkeleyentity.ConllDoc
import edu.berkeley.nlp.futile.classify.GeneralLogisticRegression
import berkeleyentity.coref.CorefSystem
import edu.berkeley.nlp.futile.util.Logger
import berkeleyentity.GUtil
import edu.berkeley.nlp.futile.classify.SequenceExample
import edu.berkeley.nlp.futile.fig.basic.IOUtils
import edu.berkeley.nlp.futile.util.Counter
import berkeleyentity.Chunk
import scala.collection.mutable.HashMap
import berkeleyentity.ConllDocReader
import berkeleyentity.lang.Language
import berkeleyentity.ConllDocWriter
import edu.berkeley.nlp.math.SloppyMath
import berkeleyentity.wiki.WikipediaInterface
import berkeleyentity.coref.UID
import berkeleyentity.sem.BrownClusterInterface
import berkeleyentity.ner.NerFeaturizer
import berkeleyentity.ner.NerExample
import berkeleyentity.ner.NEEvaluator
import berkeleyentity.ner.NerSystemLabeled
import berkeleyentity.ner.NerDriver

object Conll2003Test {

  def main(args: Array[String]) {
     val trainPath = "/home/mc/workplace/rand_search/ner2003/ner/eng.train" 
     val testPath = "/home/mc/workplace/rand_search/ner2003/ner/eng.testb"
     val sysPath = ""
     
     val syst = loadConll2003System(sysPath)
     //val testExamples = Conll2003NerInstanceLoader.loadNerExamples(testPath)
     Conll2003NerSystem.evaluateNerSystem(syst, testPath);
  }

  
  def loadConll2003System(modelPath: String) = {
    var nerSystem: NerSystemConll2003 = null;
    try {
      val fileIn = new FileInputStream(new File(modelPath));
      val in = new ObjectInputStream(fileIn);
      nerSystem = in.readObject().asInstanceOf[NerSystemConll2003]
      Logger.logss("Model read from " + modelPath);
      in.close();
      fileIn.close();
    } catch {
      case e: Exception => throw new RuntimeException(e);
    }
    nerSystem;
  }
}
