package berkeleyentity.oregonstate

import java.util.Comparator
import java.util.PriorityQueue

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer
import scala.util.Random
import java.util.Collections

object SearchBeam {
  
  
  def main(args: Array[String]) {
    unitTest1();
  }
  
  def unitTest1() {
    val rnd = new Random();
    //val b1 = new SearchBeam(3, new StatePredComparator());
    val b1 = new SearchBeam(3, new StateGoldComparator());
    
    println("======");
    for (j <- 0 until 20) {
      val s = new SearchState(new Array[Int](1));
      s.cachedPredScore = rnd.nextDouble();
      s.cachedTrueLoss = rnd.nextDouble();
      b1.insert(s);
      println("  State.score = " + s.cachedPredScore + ", " + s.cachedTrueLoss);
    }
    println("------");
   
    println("size = " + b1.size());
    b1.keepTopKOnly();
    println("size = " + b1.size());
    
    val (ss1) = b1.getBestInBeam();
    println("Best = " + ss1.cachedPredScore + ", " + ss1.cachedTrueLoss);
    val (ss2) = b1.getWorstInBeam()
    println("Worst = " + ss2.cachedPredScore + ", " + ss2.cachedTrueLoss);
    
    println("======");
    val alls = b1.getAll();
    for (i <- 0 until alls.size) {
    	println("  Btate.score = " + alls(i).cachedPredScore + ", " + alls(i).cachedTrueLoss);
    }
    println("------");
    
    val (s1) = b1.getBestInBeam();
    println("Best = " + s1.cachedPredScore + ", " + s1.cachedTrueLoss);
    val (s2) = b1.getWorstInBeam()
    println("Worst = " + s2.cachedPredScore + ", " + s2.cachedTrueLoss);
    val s3 = b1.popBest();
    println("Pop = " + s3.cachedPredScore + ", " + s3.cachedTrueLoss);
    println("size after pop = " + b1.size());
    val s4 = b1.getBestInBeam();
    println("best after pop = " + s4.cachedPredScore + ", " + s4.cachedTrueLoss);
  }
  
}

  // c action according to predict score
class StatePredComparator extends Comparator[SearchState] {
  def compare(s1: SearchState, s2: SearchState): Int = {
    if (s1.cachedPredScore > s2.cachedPredScore) {
      -1;
    } else if (s1.cachedPredScore < s2.cachedPredScore) {
      1;
    } else {
      0;
    }
  }
}

class StateGoldComparator extends Comparator[SearchState] {
  def compare(s1: SearchState, s2: SearchState): Int = {
    if (s1.cachedTrueLoss > s2.cachedTrueLoss) {
      -1;
    } else if (s1.cachedTrueLoss < s2.cachedTrueLoss) {
      1;
    } else {
      // use the predict score to break tie...
      if (s1.cachedPredScore > s2.cachedPredScore) {
        -1;
      } else if (s1.cachedPredScore < s2.cachedPredScore) {
        1;
      } else {
        0;
      }
    }
  }
}

/*
class SearchBeam(val beamSize: Int) {

  assert (beamSize > 0);
	val beam = new PriorityQueue[SearchState](beamSize, new StateInversedPredComparator());

	def insertAll(stateList: Seq[SearchState]) {
		beam.addAll(stateList);
	}

	def insert(state: SearchState) {
		beam.add(state);
	}
	
	def size(): Int = {
	  beam.size;
	}
	
	def getAll() = {
	  beam.asScala.toList;
	}

  def getBestInBeam() = {
    getBestChildState(beam);
  }
  
  def getWorstInBeam() = {
    getWorstChildState(beam);
  }
  

  
	def getBestChildState(children: ListBuffer[SearchState]) = {
		var best: SearchState = null; 
	  var bestScore: Double = -Double.MaxValue;
	  for (s <- children) {
	  	if (s.cachedPredScore > bestScore) {
		  	bestScore = s.cachedPredScore;
	  		best = s;
	  	}
	  }
	  (best, bestScore);
	}
	
  def getWorstChildState(children: ListBuffer[SearchState]) = {
		var worst: SearchState = beam.peek();
	  var worstScore: Double = worst.cachedPredScore;
	  //for (s <- children) {
	   // 	if (s.cachedPredScore < worstScore) {
		//  	worstScore = s.cachedPredScore;
	  //		worst = s;
	  //	}
	  //}
	  (worst,worstScore);
	}

  
	//def keepTopKOnlyWithCorrect() {
  //  dropTail(beamSize); // throw the states that rank lower than beamsize
  //}
  
  def keepTopKOnly() {
		dropTail(beamSize); // throw the states that rank lower than beamsize
	}
  
  def popBest(): SearchState = {
    findAndRemoveBest(beam);
  }
  
  def findAndRemoveBest(stateList: ListBuffer[SearchState]): SearchState = {
    var bestIdx: Int = -1;
    var bestState: SearchState = null;
	  var bestScore: Double = -Double.MaxValue;
	  for (i <- 0 until stateList.size) {
	  	if (stateList(i).cachedPredScore > bestScore) {
		  	bestScore = stateList(i).cachedPredScore;
	  		bestState = stateList(i);
	  		bestIdx = i;
	  	}
	  }
	  
	  if (bestIdx == -1) {
	    throw new RuntimeException("No best in beam!!!");
	  }
	  
	  stateList.remove(bestIdx);
	  bestState;
  }

  def dropTail(topK: Int) {
	  if (beam.size > topK) {
	    for () {
	      
	    }
	    
		  // sort state!
		  beam.sortWith(_.cachedPredScore > _.cachedPredScore);
		  // drop tail
		  beam.take(topK);

	  }
  }

}  
*/
  


/*
class SearchBeam(val beamSize: Int) {

  assert (beamSize > 0);
	var beam = new ListBuffer[SearchState]();

	def insertAll(stateList: Seq[SearchState]) {
		beam ++= (stateList);
	}

	def insert(state: SearchState) {
		beam += (state);
	}
	
	def size(): Int = {
	  beam.size;
	}
	
	def getAll() = {
	  beam;
	}

  def getBestInBeam() = {
    getBestChildState(beam);
  }
  
  def getWorstInBeam() = {
    getWorstChildState(beam);
  }
  

  
	def getBestChildState(children: ListBuffer[SearchState]) = {
		if (beam.size == 0) {
			(null, -Double.MaxValue);
		} else {
			var best: SearchState = null; 
		  var bestScore: Double = -Double.MaxValue;
		  for (s <- children) {
		  	if (s.cachedPredScore > bestScore) {
		  		bestScore = s.cachedPredScore;
		  		best = s;
		  	}
		  }
	  	(best, bestScore);
		}
	}
	
  def getWorstChildState(children: ListBuffer[SearchState]) = {
		if (beam.size == 0) {
			(null, -Double.MaxValue);
		} else {
			var worst: SearchState = null; 
	  	var worstScore: Double = Double.MaxValue;
		  for (s <- children) {
		  	if (s.cachedPredScore < worstScore) {
		  		worstScore = s.cachedPredScore;
		  		worst = s;
		  	}
		  }
		  (worst,worstScore);
		}
	}
  
  def keepTopKOnly() {
		dropTail(beamSize); // throw the states that rank lower than beamsize
	}
  
  def popBest(): SearchState = {
    findAndRemoveBest(beam);
  }
  
  def findAndRemoveBest(stateList: ListBuffer[SearchState]): SearchState = {
    var bestIdx: Int = -1;
    var bestState: SearchState = null;
	  var bestScore: Double = -Double.MaxValue;
	  for (i <- 0 until stateList.size) {
	  	if (stateList(i).cachedPredScore > bestScore) {
		  	bestScore = stateList(i).cachedPredScore;
	  		bestState = stateList(i);
	  		bestIdx = i;
	  	}
	  }
	  
	  if (bestIdx == -1) {
	    throw new RuntimeException("No best in beam!!!");
	  }
	  
	  stateList.remove(bestIdx);
	  bestState;
  }
  
  def justSort() {
    beam.sortWith(_.cachedPredScore > _.cachedPredScore);
  }

  def dropTail(topK: Int) {
	  if (beam.size > 0) {
		  // sort state!
		  val newBeam = beam.sortWith(_.cachedPredScore > _.cachedPredScore).take(topK);;
		  // drop tail
		  beam = newBeam;

	  }
  }

}
*/

class SearchBeam(val beamSize: Int, val cmptr: Comparator[SearchState]) {

  assert (beamSize > 0);
	var beam = new ListBuffer[SearchState]();

	def insertAll(stateList: Seq[SearchState]) {
		beam ++= (stateList);
	}

	def insert(state: SearchState) {
		beam += (state);
	}
	
	def size(): Int = {
	  beam.size;
	}
	
	def getAll() = {
	  beam;
	}

  def getBestInBeam() = {
    getBestChildState(beam);
  }
  
  def getWorstInBeam() = {
    getWorstChildState(beam);
  }
  
	def getBestChildState(children: ListBuffer[SearchState]) = {
		if (beam.size == 0) {
			//(null, -Double.MaxValue);
		  null;
		} else {
		  justSort();
		  /*
			var best: SearchState = null; 
		  var bestScore: Double = -Double.MaxValue;
		  for (s <- children) {
		  	if (s.cachedPredScore > bestScore) {
		  		bestScore = s.cachedPredScore;
		  		best = s;
		  	}
		  }*/
		  val best = children(0);
		  //val bestScore = children(0).
	  	//(best, bestScore);
		  best;
		}
	}
	
  def getWorstChildState(children: ListBuffer[SearchState]) = {
		if (beam.size == 0) {
			//(null, -Double.MaxValue);
		  null;
		} else {
		  /*
			var worst: SearchState = null; 
	  	var worstScore: Double = Double.MaxValue;
		  for (s <- children) {
		  	if (s.cachedPredScore < worstScore) {
		  		worstScore = s.cachedPredScore;
		  		worst = s;
		  	}
		  }*/
		  //(worst,worstScore);
		  justSort();
		  val worst = children(children.size - 1);
		  //val bestScore = children(0).
		  worst;
		}
	}
  
  def keepTopKOnly() {
		dropTail(beamSize); // throw the states that rank lower than beamsize
	}
  
  def popBest(): SearchState = {
    justSort();
    val bestState = beam(0);
    beam.remove(0);
	  bestState;
  }

  
  def justSort() {
    //beam.sortWith(_.cachedPredScore > _.cachedPredScore);
    Collections.sort(beam.asJava, cmptr);
  }
/*
  def dropTail(topK: Int) {
	  if (beam.size > 0) {
		  // sort state!
		  val newBeam = beam.sortWith(_.cachedPredScore > _.cachedPredScore).take(topK);;
		  // drop tail
		  beam = newBeam;

	  }
  }
*/
  def dropTail(topK: Int) {
	  if (beam.size > 0) {
	    Collections.sort(beam.asJava, cmptr);
		  val newBeam = beam.take(topK);
		  beam = newBeam;
	  }
  }
}

class ActionBeam(val topkSize: Int, comparator: Comparator[SearchAction]) {
  
  val inversedQueue = new PriorityQueue[SearchAction](topkSize, comparator);
  
  def addOnly(action: SearchAction) {
    inversedQueue.add(action); // add a new element
  }
  
  def addAndKeepTopK(action: SearchAction) {
    //println("Add action with score " + action.score);
    inversedQueue.add(action);
    if (inversedQueue.size() > topkSize) {
      dropTail();
    }
  }
  
  def dropTail() {
    // drop the in
    while (inversedQueue.size() > topkSize) {
      inversedQueue.poll(); // drop one~
    }
  }
  
  def getAllTopK() = {
    if (inversedQueue.size() > topkSize) {
      dropTail();
    }
    getAll();
  }
  
  def getAll() = {
    val arr = inversedQueue.asScala.toList;
    //for (i <- 0 until arr.size) {
    //  println("Act(" + i + ").score = " + arr(i).score);
    //}
    arr;
  }
  
  def getPeek() = {
    inversedQueue.peek(); // here actually returns the minimum score
  }
  
}


  // compare action according to predict score
class ActionPredComparator extends Comparator[SearchAction] {
  def compare(a1: SearchAction, a2: SearchAction): Int = {
    if (a1.score > a2.score) {
      -1;
    } else if (a1.score < a2.score) {
      1;
    } else {
      0;
    }
  }
}

// compare action according to true accuracy
class ActionOracleComparator extends Comparator[SearchAction] {
  def compare(a1: SearchAction, a2: SearchAction): Int = {
    if (a1.trueAcc > a2.trueAcc) {
      -1;
    } else if (a1.trueAcc < a2.trueAcc) {
      1;
    } else {
      0;
    }
  }
}

/////////////////////////////
//// Inversed Comparator ////
/////////////////////////////

  // compare action according to predict score
class ActionInversedPredComparator extends Comparator[SearchAction] {
  def compare(a1: SearchAction, a2: SearchAction): Int = {
    if (a1.score > a2.score) {
      1;
    } else if (a1.score < a2.score) {
      -1;
    } else {
      0;
    }
  }
}

// compare action according to true accuracy
class ActionInversedOracleComparator extends Comparator[SearchAction] {
  def compare(a1: SearchAction, a2: SearchAction): Int = {
    if (a1.trueAcc > a2.trueAcc) {
      1;
    } else if (a1.trueAcc < a2.trueAcc) {
      -1;
    } else {
      0;
    }
  }
}