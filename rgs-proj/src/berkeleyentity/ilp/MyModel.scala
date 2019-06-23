package berkeleyentity.ilp

import edu.berkeley.nlp.futile.fig.basic.Indexer
import berkeleyentity.GUtil

import java.io.PrintWriter

class MyModel(val featIndexer: Indexer[String],
              val weights: Array[Double]) extends Serializable  {

}

object MyModel {
  def saveMyModel(m: MyModel, path: String) {
    GUtil.save(m, path);
    // =======
    val printer = new PrintWriter(path+".weights.log");
    val idxer = m.featIndexer;
    for (i <- 0 until idxer.size()) {
      val featContent = idxer.get(i);
      val wval = m.weights(i);
      printer.println(i + " " + featContent + " " + wval);
    }
    printer.close();
  }
  
  def loadModel(path: String): MyModel = {
    GUtil.load(path).asInstanceOf[MyModel];
  }
}
