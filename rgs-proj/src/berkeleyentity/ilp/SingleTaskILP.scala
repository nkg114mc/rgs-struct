package berkeleyentity.ilp

import scala.util.control.Breaks._

/**
 * Structural learning for single task only
 * To study on each task
 */
class SingleTaskILP {
  
  // ACE /////////////////
  // structural learning for 
  
  
  // Only ACE coref
  def corefTrainingACE() {
    
  }
  
  def corefTestACE() {
    
  }
  
  
  // Only ACE NER
  
  
  
  // OntoNotes
  
  

}


object SingleTaskILP {
  
  
  
}


class SingleDecision(val score: Double, val isCorrect: Boolean, val domainIdx: Int) {
  def compare (that: SingleDecision) = {
        (that.score > this.score);
    }
}

class HistgramRecord() {
  
    var corefCrrctRank = Array.fill[Int](10000)(0);//new Array[Int](10000);
    var nerCrrctRank = Array.fill[Int](10000)(0);//new Array[Int](10000);
    var wikiCrrctRank = Array.fill[Int](10000)(0);//new Array[Int](10000);
    
    def increaseForMention(cdecisions: Array[SingleDecision],
                           ndecisions: Array[SingleDecision],
                           wdecisions: Array[SingleDecision]) {
      val bestCrrtCorefIdx = getFirstCorrectRank(cdecisions, 9999);
      corefCrrctRank(bestCrrtCorefIdx) += 1;
      val bestCrrtNerIdx = getFirstCorrectRank(ndecisions, 9999);
      nerCrrctRank(bestCrrtNerIdx) += 1;
      val bestCrrtWikiIdx = getFirstCorrectRank(wdecisions, 9999);
      wikiCrrctRank(bestCrrtWikiIdx) += 1;
    }
    
    def getFirstCorrectRank(decisions: Array[SingleDecision], defaultNoCrrctRank: Int) = {
    	var result = defaultNoCrrctRank;
    	breakable {
    		for (rank <- 0 until decisions.length) {
    			if (decisions(rank).isCorrect) {
    				result = rank;
            break;
    			}
    		}
    	}
    	result;
    }
    
    def printHistgram() {
      for (i <- 0 until 10) {
        println("Coref["+i+"]: " + corefCrrctRank(i));
      }
      for (i <- 0 until 10) {
        println("Ner["+i+"]: " + nerCrrctRank(i));
      }
      for (i <- 0 until 10) {
        println("Wiki["+i+"]: " + wikiCrrctRank(i));
      }
    }
}