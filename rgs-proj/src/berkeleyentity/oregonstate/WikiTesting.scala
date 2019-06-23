package berkeleyentity.oregonstate;

import berkeleyentity.wiki.ACETester
import berkeleyentity.wiki.WikipediaAuxDB
import berkeleyentity.wiki.WikipediaCategoryDB
import berkeleyentity.wiki.WikipediaInterface
import berkeleyentity.wiki.WikipediaLinkDB
import berkeleyentity.wiki.WikipediaRedirectsDB
import berkeleyentity.wiki.WikipediaTitleGivenSurfaceDB
import berkeleyentity.wiki.JointQueryDenotationChoiceComputer
import berkeleyentity.wiki.JointQueryDenotationChooser
import berkeleyentity.wiki.JointQueryDenotationExample
import berkeleyentity.wiki.Query
import berkeleyentity.wiki.WikiAnnotReaderWriter
import berkeleyentity.wiki.WikificationEvaluator
import berkeleyentity.wiki.CorpusWikiAnnots
import berkeleyentity.wiki.DocWikiAnnots
import berkeleyentity.Driver
import berkeleyentity.lang.Language
import edu.berkeley.nlp.futile.LightRunner
import berkeleyentity.coref.CorefDocAssembler
import berkeleyentity.ConllDocReader
import berkeleyentity.coref.MentionPropertyComputer
import berkeleyentity.GUtil
import edu.berkeley.nlp.futile.fig.basic.Indexer
import berkeleyentity.joint.LikelihoodAndGradientComputer
import scala.collection.mutable.ArrayBuffer
import berkeleyentity.coref.CorefDoc
import edu.berkeley.nlp.futile.math.SloppyMath
import edu.berkeley.nlp.futile.util.Logger
import berkeleyentity.Chunk
import berkeleyentity.joint.GeneralTrainer
import berkeleyentity.coref.NumberGenderComputer
import berkeleyentity.coref.Mention
import java.io.PrintWriter
import java.io.File

/*
class queryWikiValue(val query: Query,
      val wiki: String,
      val qfeats: Array[Int],
      val qwfeats: Array[Int],
      val isCorrect: Boolean) {
    
      var qidx: Int = -1;
      var widx: Int = -1;
      val concatenateFeats = qfeats ++ qwfeats;

    def computeScore(wght: Array[Double]): Double = {
      var result : Double = 0;
      for (idx1 <- qfeats) {
        result += (wght(idx1));
      }
      for (idx2 <- qwfeats) {
        result += (wght(idx2));
      }
      result;
    }
    
    def getConcatenateFeatures() = {
      concatenateFeats
    }
    
    def getFeatStr(qid: Int): String = {
      var result = "";
      val feat = concatenateFeats;
      result += (if (isCorrect) "1" else "0");
      result += (" qid:" + qid);
      for (i <- feat) {
        result += (" " + String.valueOf(i + 1) + ":" + String.valueOf(1.0));
      }
      result;
    }

}
*/
class QueryWikiValue(val query: Query,
		                 val wiki: String,
		                 val qfeats: Array[Int],
		                 val qwfeats: Array[Int],
		                 val isCorrect: Boolean) {

	var qidx: Int = -1;
  var widx: Int = -1;
	val concatenateFeats = qfeats ++ qwfeats;

	def computeScore(wght: Array[Double]): Double = {
			var result : Double = 0;
	for (idx1 <- qfeats) {
		result += (wght(idx1));
	}
	for (idx2 <- qwfeats) {
		result += (wght(idx2));
	}
	result;
	}

	def getConcatenateFeatures() = {
		concatenateFeats
	}

	def getFeatStr(qid: Int): String = {
		var result = "";
		val feat = concatenateFeats;
		result += (if (isCorrect) "1" else "0");
		result += (" qid:" + qid);
		for (i <- feat) {
			result += (" " + String.valueOf(i + 1) + ":" + String.valueOf(1.0));
		}
		result;
	}
  
  override def toString() = {
    wiki;
  }

}

object WikiTesting {
	
	def main(args: Array[String]) {
    trainWikificationACE(args);
	}
  
  

  
  class myWikiPredictor(val featIndexer: Indexer[String],
                        val weights: Array[Double],
                        wikiDB: WikipediaInterface) {
    
    // compute the wiki value with highest score
    def predictBestDenotation(ex: JointQueryDenotationExample) : String = {
      /*
      val domains = constructWikiExampleDomains(ex, wikiDB);
      var bestLbl = -1;
      var bestScore = -Double.MaxValue;
      for (l <- 0 until domains.size) {
          var score = domains(l).computeScore(weights);
          if (score > bestScore) {
            bestScore = score;
            bestLbl = l;
          }
      }
      domains(bestLbl).wiki;
      */
      val bestDmVal = predictBestDomainValue(ex);
      bestDmVal.wiki;
    }
    
    def predictBestDomainValue(ex: JointQueryDenotationExample): QueryWikiValue = {
      val domains = constructWikiExampleDomains(ex, wikiDB);
      var bestLbl = -1;
      var bestScore = -Double.MaxValue;
      for (l <- 0 until domains.size) {
          var score = domains(l).computeScore(weights);
          if (score > bestScore) {
            bestScore = score;
            bestLbl = l;
          }
      }
      domains(bestLbl);
    }
    
    def predictBestQuery(ex: JointQueryDenotationExample) = {
      val queries = ex.queries;
      var bestq: Query = null;
      var bestsc = -Double.MaxValue;
      for (qidx <- 0 until queries.size) {
        val q = queries(qidx);
        val fq = ex.cachedFeatsEachQuery(qidx);
        val sc = computeScoreGivenFeat(fq, weights);
        if (sc > bestsc) {
          bestsc = sc;
          bestq = q;
        }
      }
      bestq;
    }
    
    def predictWikiWithBestQuery(ex: JointQueryDenotationExample) = {
      val bestq: Query = predictBestQuery(ex);
      val denotations = ex.allDenotations;
      val queryOutcomes = wikiDB.disambiguateBestGetAllOptions(bestq);
      
      var crrDeno: String = null;
      for (didx <- 0 until denotations.size) {
          val d = denotations(didx);
          if ((d == NilToken) || (queryOutcomes.containsKey(d))) {
            val isCorrect = ex.correctDenotationIndices.contains(didx);
            if (isCorrect) {
              crrDeno = d;
            }
          }
      }
      
      if (crrDeno == null) {
        crrDeno = denotations(0);
      }
      crrDeno
    }
    
    def predictTwoSteps(queryPicker: myWikiPredictor,
                        ex: JointQueryDenotationExample) = {
      val domains = constructDenotationDomains(queryPicker, ex, wikiDB);
      var bestLbl = -1;
      var bestScore = -Double.MaxValue;
      var bestCrrLbl = -1;
      var bestCrrScore = -Double.MaxValue;
      
      for (l <- 0 until domains.size) {
          var score = domains(l).computeScore(weights);
          if (score > bestScore) {
            bestScore = score;
            bestLbl = l;
          }
          if (domains(l).isCorrect) {
            if (score > bestCrrScore) {
              bestCrrScore = score;
              bestCrrLbl = l;
            }
          }
      }
      
      bestCrrLbl = -1;
      var re = if (bestCrrLbl != -1) {
        domains(bestCrrLbl).wiki;
      } else {
        domains(bestLbl).wiki;
      }
      re;
    }
    
    
    
    def computeScoreGivenFeat(feat: Array[Int], wght: Array[Double]): Double = {
      var result : Double = 0;
      for (idx1 <- feat) {
        result += (wght(idx1));
      }
      result;
    }
  }

  def getGoldWikification(goldWiki: DocWikiAnnots, ment: Mention): Seq[String] = {
    if (!goldWiki.contains(ment.sentIdx)) {
      Seq[String]();
    } else {
      val matchingChunks = goldWiki(ment.sentIdx).filter(chunk => chunk.start == ment.startIdx && chunk.end == ment.endIdx);
      //if (matchingChunks.isEmpty) Seq[String]() else matchingChunks(0).label;
      if (matchingChunks.size != 1) Seq[String]() else matchingChunks(0).label;
    }
  }
  
  def isCorrect(gold: Seq[String], guess: String): Boolean = {
    (gold.map(_.toLowerCase).contains(guess.toLowerCase.replace(" ", "_"))); // handles the -NIL- case too
  }
  
  
  def extractExamples(corefDocs: Seq[CorefDoc], goldWikification: CorpusWikiAnnots, wikiDB: WikipediaInterface, filterImpossible: Boolean = false) = {
    val writer = new PrintWriter("wiki_men.log");
    val exs = new ArrayBuffer[JointQueryDenotationExample];
    var numImpossible = 0;
    // Go through all mentions in all documents
    for (corefDoc <- corefDocs) {
      val docName = corefDoc.rawDoc.docID
      for (i <- 0 until corefDoc.predMentions.size) {
        // Discard "closed class" mentions (pronouns) since these don't have interesting entity links
        if (!corefDoc.predMentions(i).mentionType.isClosedClass()) {
          val ment = corefDoc.predMentions(i);
          // There are multiple possible gold Wikipedia titles for some mentions. Note that
          // NIL (no entry in Wikipedia) is included as an explicit choice, so this includes NILs (as
          // it should according to how the task is defined)
          val goldLabel = getGoldWikification(goldWikification(docName), ment)
          
          
          
          //if (goldLabel.size >= 1) {
            val queries = Query.extractQueriesBest(ment, true);
            val queryDisambigs = queries.map(wikiDB.disambiguateBestGetAllOptions(_));
            
            
            //val denotations = Query.extractDenotationSetWithNil(queries, queryDisambigs, maxNumWikificationOptions);
            //val correctDenotations = denotations.filter(denotation => isCorrect(goldLabel, denotation))
            // N.B. The use of "isCorrect" here is needed to canonicalize 
            //val correctIndices = denotations.zipWithIndex.filter(denotationIdx => isCorrect(goldLabel, denotationIdx._1)).map(_._2);
            val denotationsRaw = Query.extractDenotationSetWithNil(queries, queryDisambigs, maxNumWikificationOptions);
            var correctDenotations = denotationsRaw.filter(den => isCorrect(goldLabel, den));
            if (correctDenotations.isEmpty) correctDenotations = denotationsRaw;
            // N.B. The use of "isCorrect" here is needed to canonicalize 
            //val correctIndices = denotations.zipWithIndex.filter(denotationIdx => isCorrect(goldLabel, denotationIdx._1)).map(_._2);
            
            if (false) {
              numImpossible += 1;
/*
              writer.println("Impossible Ment: "+ ment.spanToString);
              writer.println(" === Gold Labels === ");
              for (gl <- goldLabel) {
                writer.println("GoldLbl: " + gl);
              }
              writer.println(" === Gen Labels === ");
              for (ii <- 0 until denotations.size) {
                writer.println("GenLbl: " + denotations(ii));
              }
              writer.println(" === ======== === ");
*/
            } else {
              exs += new JointQueryDenotationExample(queries, denotationsRaw, correctDenotations, goldLabel)
            }
          //}
          
          
        }
      }
    }
    Logger.logss(exs.size + " possible, " + numImpossible + " impossible");
    writer.close();
    exs;
  }

  
  val trainDataPath = "data/ace05/train";
  val devDataPath = "data/ace05/dev";
  val testDataPath = "data/ace05/test";
  val wikiPath = "data/ace05/ace05-all-conll-wiki"
  val wikiDBPath = "models/wiki-db-ace.ser.gz"
  
  val NilToken = "-NIL-";
  
  val lambda = 1e-8F
  val batchSize = 1
  val numItrs = 20
  
  val maxNumWikificationOptions = 7
  
  def trainWikificationACE(args: Array[String]) {
    
    // Read in CoNLL documents 
    val numberGenderComputer = NumberGenderComputer.readBergsmaLinData(Driver.numberGenderDataPath);
    val mentionPropertyComputer = new MentionPropertyComputer(Some(numberGenderComputer));
    
    val assembler = CorefDocAssembler(Language.ENGLISH, true);
    val trainDocs = ConllDocReader.loadRawConllDocsWithSuffix(trainDataPath, -1, "", Language.ENGLISH);
    val trainCorefDocs = trainDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    
    // Read in gold Wikification labels
    val goldWikification = WikiAnnotReaderWriter.readStandoffAnnotsAsCorpusAnnots(wikiPath)
    // Read in the title given surface database
    val wikiDB = GUtil.load(wikiDBPath).asInstanceOf[WikipediaInterface];
    // Make training examples, filtering out those with solutions that are unreachable because
    // they're not good for training
    val trainExs = extractExamples(trainCorefDocs, goldWikification, wikiDB, filterImpossible = true)
    
    // Extract features
    val featIndexer = new Indexer[String]
    val computer = new JointQueryDenotationChoiceComputer(wikiDB, featIndexer);
    for (trainEx <- trainExs) {
      computer.featurizeUseCache(trainEx, true);
    }
    Logger.logss(featIndexer.size + " features");

    
    // extract test examples
    // Build the test examples and decode the test set
    // No filtering now because we're doing test
    val testDocs = ConllDocReader.loadRawConllDocsWithSuffix(testDataPath, -1, "", Language.ENGLISH);
    val testCorefDocs = testDocs.map(doc => assembler.createCorefDoc(doc, mentionPropertyComputer));
    val testExs = extractExamples(testCorefDocs, goldWikification, wikiDB, filterImpossible = false);
    for (tstEx <- testExs) {
      computer.featurizeUseCache(tstEx, false);
    }
    
/*     
    //val weights = queryLearning(trainExs, featIndexer, wikiDB, testExs);
    
    
    
    val queryWghtFile = new File("query_weight_ace05.weight_full");
    val weights = if (!queryWghtFile.exists()) {
      val pw = queryLearning(trainExs, featIndexer, wikiDB, testExs);//runUnaryPrunerTraining(trainIndepExs, testIndepExs, featIndexer,  goldWikification);
      SearchDomainPruner.savePrunerWeight(pw, queryWghtFile.getAbsolutePath);
      pw;  
    } else {
      val pw = new Array[Double](featIndexer.size);
      SearchDomainPruner.loadPrunerWeight(pw, queryWghtFile.getAbsolutePath);
      pw;
    }
    
    
    
   
    val wghtDeno = denotationLearning(weights, trainExs, featIndexer, wikiDB, testExs);
    // Train
    //val gt = new GeneralTrainer[JointQueryDenotationExample]();
    //val weights = structurePerceptrion(trainExs, featIndexer, wikiDB, testExs);//trainWikification(trainExs, featIndexer, testExs);

    
    
    val queryPicker = new myWikiPredictor(featIndexer, weights, wikiDB);
    //val chooser = new JointQueryDenotationChooser(featIndexer, weights)
    val myModel = new myWikiPredictor(featIndexer, wghtDeno, wikiDB);
    
    //testWikiResults(trainExs, myModel); // test on train docs
    //testWikiResults(testExs, myModel);
    testWikiResultsTwoSteps(trainExs, queryPicker, myModel);
    testWikiResultsTwoSteps(testExs, queryPicker, myModel);
    
    testWikiResultsWithPredictQueryOnly(trainExs, queryPicker);
    testWikiResultsWithPredictQueryOnly(testExs, queryPicker);
    computeAccuracySeparatly(testExs, queryPicker, wikiDB); 

 */
  }
  
  def testWikiResults(exs: ArrayBuffer[JointQueryDenotationExample], model: myWikiPredictor) {
    val goldTestDenotationsAsTrivialChunks = (0 until exs.size).map(i => new Chunk[Seq[String]](i, i+1, exs(i).rawCorrectDenotations))
    //val predTestDenotationsAsTrivialChunks = (0 until testExs.size).map(i => new Chunk[String](i, i+1, chooser.pickDenotation(testExs(i).queries, wikiDB)))
    val predTestDenotationsAsTrivialChunks = (0 until exs.size).map(i => new Chunk[String](i, i+1, model.predictBestDenotation(exs(i))))
    // Hacky but lets us reuse some code that normally evaluates things with variable endpoints
    println("==== Wikification Results ====");
    WikificationEvaluator.evaluateFahrniMetrics(Seq(goldTestDenotationsAsTrivialChunks), Seq(predTestDenotationsAsTrivialChunks), Set())
  }
  
  // only predict query, pick the truth denotation once query was given
  def testWikiResultsWithPredictQueryOnly(exs: ArrayBuffer[JointQueryDenotationExample], model: myWikiPredictor) {
    val goldTestDenotationsAsTrivialChunks = (0 until exs.size).map(i => new Chunk[Seq[String]](i, i+1, exs(i).rawCorrectDenotations))
    //val predTestDenotationsAsTrivialChunks = (0 until testExs.size).map(i => new Chunk[String](i, i+1, chooser.pickDenotation(testExs(i).queries, wikiDB)))
    val predTestDenotationsAsTrivialChunks = (0 until exs.size).map(i => new Chunk[String](i, i+1, model.predictWikiWithBestQuery(exs(i))))
    // Hacky but lets us reuse some code that normally evaluates things with variable endpoints
    println("==== Wikification Results Predict Query Only ====");
    WikificationEvaluator.evaluateFahrniMetrics(Seq(goldTestDenotationsAsTrivialChunks), Seq(predTestDenotationsAsTrivialChunks), Set())
  }
  
  def testWikiResultsTwoSteps(exs: ArrayBuffer[JointQueryDenotationExample], qpicker: myWikiPredictor, model: myWikiPredictor) {
    val goldTestDenotationsAsTrivialChunks = (0 until exs.size).map(i => new Chunk[Seq[String]](i, i+1, exs(i).rawCorrectDenotations))
    //val predTestDenotationsAsTrivialChunks = (0 until testExs.size).map(i => new Chunk[String](i, i+1, chooser.pickDenotation(testExs(i).queries, wikiDB)))
    val predTestDenotationsAsTrivialChunks = (0 until exs.size).map(i => new Chunk[String](i, i+1, model.predictTwoSteps(qpicker, exs(i))))
    // Hacky but lets us reuse some code that normally evaluates things with variable endpoints
    println("==== Wikification Results Two Steps ====");
    WikificationEvaluator.evaluateFahrniMetrics(Seq(goldTestDenotationsAsTrivialChunks), Seq(predTestDenotationsAsTrivialChunks), Set())
  }
  
  def computeAccuracySeparatly(exs: ArrayBuffer[JointQueryDenotationExample], model: myWikiPredictor, wikiDB: WikipediaInterface) {
	  // query chooser accuracy
	  var total = 0;
	  var correct = 0;
	  for (ex <- exs) {
		  //val pred = model.predictBestDomainValue(ex);
		  val bestq = model.predictBestQuery(ex);
		  val queryOutcomes = wikiDB.disambiguateBestGetAllOptions(bestq);
		  val denotations = ex.allDenotations;
		  var isCrr = false;
		  for (didx <- 0 until denotations.size) {
			  val d = denotations(didx);
			  if ((d == NilToken) || (queryOutcomes.containsKey(d))) {
				  if (ex.correctDenotationIndices.contains(didx)) {
					  isCrr = true;
				  }
			  }
		  }
      
      total += 1;
      if (isCrr) {
        correct += 1;
      }
	  }
    
    println("Query acc: " + correct + "/" + total);

  }
  
  // structural learning
  def trainWikification(trainExs: ArrayBuffer[JointQueryDenotationExample],
                        featIndexer: Indexer[String],
                        testExs: ArrayBuffer[JointQueryDenotationExample]): Array[Double] = {
    ???
  }

  def constructWikiExampleDomains(ex: JointQueryDenotationExample,
		  wikiDB: WikipediaInterface): ArrayBuffer[QueryWikiValue] = {
		  val qwvals = new ArrayBuffer[QueryWikiValue]();
		  val queries = ex.queries;
		  val denotations = ex.allDenotations;
		  val queryOutcomes = queries.map(query => wikiDB.disambiguateBestGetAllOptions(query));
      
		  for (qidx <- 0 until queries.size) {
			  for (didx <- 0 until denotations.size) {
				  val q = queries(qidx);
				  val d = denotations(didx);
				  if ((d == NilToken) || (queryOutcomes(qidx).containsKey(d))) {
					  val fq = ex.cachedFeatsEachQuery(qidx);
					  val fqd = ex.cachedFeatsEachQueryDenotation(qidx)(didx);
					  //println(fq.length + ", " + fqd.length)
					  val isCorrect = ex.correctDenotationIndices.contains(didx);
            
            val newValue = new QueryWikiValue(q, d, fq, fqd, isCorrect);
            newValue.qidx = qidx;
            newValue.widx = didx;
					  qwvals += (newValue);
				  }
			  }
		  }
		  qwvals
  }
  
  def constructQueryDomains(ex: JointQueryDenotationExample,
                            wikiDB: WikipediaInterface): ArrayBuffer[QueryWikiValue] = {

      val qwvals = new ArrayBuffer[QueryWikiValue]();
      val queries = ex.queries;
      val denotations = ex.allDenotations;
      val queryOutcomes = queries.map(query => wikiDB.disambiguateBestGetAllOptions(query));
      
      for (qidx <- 0 until queries.size) {
        val q = queries(qidx);
        val fq = ex.cachedFeatsEachQuery(qidx);
        var containCrr = false;
        for (didx <- 0 until denotations.size) {
          val d = denotations(didx);
          if ((d == NilToken) || (queryOutcomes(qidx).containsKey(d))) {
            val fqd = ex.cachedFeatsEachQueryDenotation(qidx)(didx);
            //println(fq.length + ", " + fqd.length)
            val isCorrect = ex.correctDenotationIndices.contains(didx);
            if (isCorrect) containCrr = true;
          }
        }
        
        // only query
        val newValue = new QueryWikiValue(q, "none", fq, Array(), containCrr);
        newValue.qidx = qidx;
        newValue.widx = 0;
        qwvals += (newValue);
      }
      qwvals
  }
  
  def constructDenotationDomains(queryPicker: myWikiPredictor,
                                 ex: JointQueryDenotationExample,
                                 wikiDB: WikipediaInterface): ArrayBuffer[QueryWikiValue] = {

      
    
      val queries = ex.queries;
      var bestq: Query = null;
      var bestsc = -Double.MaxValue;
      var bestqid = -1;
      for (qidx <- 0 until queries.size) {
        val q = queries(qidx);
        val fq = ex.cachedFeatsEachQuery(qidx);
        val sc = queryPicker.computeScoreGivenFeat(fq, queryPicker.weights);
        if (sc > bestsc) {
          bestsc = sc;
          bestq = q;
          bestqid = qidx;
        }
      }
      
      //val query = bestq

      /////////////////////////////////
      
      val qwvals = new ArrayBuffer[QueryWikiValue]();
      val denotations = ex.allDenotations;
      val queryOutcomes = wikiDB.disambiguateBestGetAllOptions(bestq);

      //var containCrr = false;
      val fq = ex.cachedFeatsEachQuery(bestqid);
      for (didx <- 0 until denotations.size) {
    	  val d = denotations(didx);
    	  if ((d == NilToken) || (queryOutcomes.containsKey(d))) {
    		  val fqd = ex.cachedFeatsEachQueryDenotation(bestqid)(didx);
    		  val isCorrect = ex.correctDenotationIndices.contains(didx);
    		  //if (isCorrect) containCrr = true;

    		  // only query
    		  val newValue = new QueryWikiValue(bestq, d, fq, fqd, isCorrect);
    		  newValue.qidx = bestqid;
    		  newValue.widx = didx;
    		  qwvals += (newValue);
    	  }
      }
        
      qwvals
  }

/*
  def latentPerceptrion(allTrains: ArrayBuffer[JointQueryDenotationExample],
                        featIndexer: Indexer[String],
                        wikiDB: WikipediaInterface,
                        testExs: ArrayBuffer[JointQueryDenotationExample]) : Array[Double] = {
   
    //val logger = new PrintWriter("wiki_ace05_train.txt");
    //val logger2 = new PrintWriter("wiki_ace05_test.txt");
  
    var weight = Array.fill[Double](featIndexer.size)(0);//new Array[Double](featIndexer.size());
    var weightSum = Array.fill[Double](featIndexer.size)(0);
    var lastWeight = Array.fill[Double](featIndexer.size)(0);
    
    val Iteration = 100;
    val learnRate = 0.1;
    val lambda = 1e-8;
    
    var updateCnt = 0;
    var lastUpdtCnt = 0;
    
    var exId2 = 0;
    for (extst <- testExs) {
      exId2 += 1;
      val domains = constructWikiExampleDomains(extst, wikiDB);
      var corrCnt = 0;
      for (l <- 0 until domains.size) {
      //  logger2.println(domains(l).getFeatStr(exId2));
        if (domains(l).isCorrect) corrCnt += 1;
      }
      if (corrCnt <= 0) {
        throw new RuntimeException("No correct value in the domain!");
      }
    }
    
    
    
    for (iter <- 0 until Iteration) {
      lastUpdtCnt = updateCnt;
      Array.copy(weight, 0, lastWeight, 0, weight.length);
        
      println("Iter " + iter);
      var exId = 0;
      for (example <- allTrains) {
        
        exId += 1;
        val domains = constructWikiExampleDomains(example, wikiDB);
        val domainCorrects = domains.filter { d => d.isCorrect };
        
        // inference to get h*
        var besthIdx = -1;
        var besthScore = -Double.MaxValue;
        for (l <- 0 until domains.size) {
        	if (domains(l).isCorrect) {
        		var score = domains(l).computeScore(weight);
        		if (score > besthScore) {
        			besthScore = score;
        			besthIdx = domains(l).qidx;
        		}
        	}
        }
        
        //println("qstar = " + besthIdx)

        //val qstarValues = domains.filter { d => { d.qidx == besthIdx } };

        var bestLbl = -1;
        var bestScore = -Double.MaxValue;
        var bestCorrectLbl = -1; // latent best
        var bestCorrectScore = -Double.MaxValue;
        for (l <- 0 until domains.size) {
        	if (domains(l).qidx == besthIdx) {
            var lossAug = if (domains(l).isCorrect) 0 else 1.0;
        		var score = domains(l).computeScore(weight) + lossAug;
        		if (score > bestScore) {
        			bestScore = score;
        			bestLbl = l;
        		}
        		if (domains(l).isCorrect) {
        			if (score > bestCorrectScore) {
        				bestCorrectScore = score;
        				bestCorrectLbl = l;
        			}
        		}
        	}
        }
        

        //println("bestLbl = " + bestLbl)
        //println("bestCorrectLbl = " + bestCorrectLbl)
        
        
        // update?
        //if (!domains(bestLbl).isCorrect) {
          updateCnt += 1;
          if (updateCnt % 1000 == 0) println("Update " + updateCnt);
          updateWeight(weight, 
                       domains(bestCorrectLbl).concatenateFeats,
                       domains(bestLbl).concatenateFeats,
                       learnRate,
                       lambda);
          sumWeight(weightSum, weight);
        //}
      }
    
      ///////////////////////////////////////////////////
      // have a test after each iteration (for learning curve)
      val tmpAvg = new Array[Double](weightSum.size)
      Array.copy(weightSum, 0, tmpAvg, 0, weightSum.size);
      divdeNumber(tmpAvg, updateCnt.toDouble);
      //testAceNerSystem(testExs, tmpAvg, Some(logger));
      //quickTest(testExs, tmpAvg);
      
      quickTest(allTrains, tmpAvg, wikiDB);
      quickTest(testExs, tmpAvg, wikiDB);
      println("Iter Update Cnt = " + (updateCnt - lastUpdtCnt));
      
      val wdiff = checkWeight(weight, lastWeight);
      println("Weight diff = " + wdiff);
    }
    
    divdeNumber(weightSum, updateCnt.toDouble);
    weightSum;
  }
  
  def inferenceLossAugmented() {
    
  }
*/
  
  def queryLearning(allTrains: ArrayBuffer[JointQueryDenotationExample],
                    featIndexer: Indexer[String],
                    wikiDB: WikipediaInterface,
                    testExs: ArrayBuffer[JointQueryDenotationExample]) : Array[Double] = {
  
    var weight = Array.fill[Double](featIndexer.size)(0);//new Array[Double](featIndexer.size());
    var weightSum = Array.fill[Double](featIndexer.size)(0);
    var lastWeight = Array.fill[Double](featIndexer.size)(0);
    
    val Iteration = 100;
    val learnRate = 0.1;
    val lambda = 1e-8;
    
    var updateCnt = 0;
    var lastUpdtCnt = 0;
    
    var exId2 = 0;
    for (extst <- testExs) {
      exId2 += 1;
      val domains = constructQueryDomains(extst, wikiDB);
      var corrCnt = 0;
      for (l <- 0 until domains.size) {
        if (domains(l).isCorrect) corrCnt += 1;
      }
      if (corrCnt <= 0) {
        throw new RuntimeException("No correct value in the domain!");
      }
    }
    
    for (iter <- 0 until Iteration) {
      lastUpdtCnt = updateCnt;
      Array.copy(weight, 0, lastWeight, 0, weight.length);
        
      println("Iter " + iter);
      var exId = 0;
      for (example <- allTrains) {
        
        exId += 1;
        val domains = constructQueryDomains(example, wikiDB);

        var bestLbl = -1;
        var bestScore = -Double.MaxValue;
        var bestCorrectLbl = -1; // latent best
        var bestCorrectScore = -Double.MaxValue;
        for (l <- 0 until domains.size) {
          var score = domains(l).computeScore(weight);
          if (score > bestScore) {
            bestScore = score;
            bestLbl = l;
          }
          if (domains(l).isCorrect) {
            if (score > bestCorrectScore) {
              bestCorrectScore = score;
              bestCorrectLbl = l;
            }
          }
        }
        
        //println("size = " + domains.size + " pred = " + bestLbl + " correct = " + bestCorrectLbl)
        
        // update?
        if (!domains(bestLbl).isCorrect) {
          updateCnt += 1;
          if (updateCnt % 1000 == 0) println("Update " + updateCnt);
          updateWeight(weight, 
                       domains(bestCorrectLbl).concatenateFeats,
                       domains(bestLbl).concatenateFeats,
                       learnRate,
                       lambda);
          sumWeight(weightSum, weight);
        }
      }
    
      ///////////////////////////////////////////////////
      // have a test after each iteration (for learning curve)
      val tmpAvg = new Array[Double](weightSum.size)
      Array.copy(weightSum, 0, tmpAvg, 0, weightSum.size);
      divdeNumber(tmpAvg, updateCnt.toDouble);
      //testAceNerSystem(testExs, tmpAvg, Some(logger));
      //quickTest(testExs, tmpAvg);
      
      quickTestQuery(allTrains, tmpAvg, wikiDB);
      quickTestQuery(testExs, tmpAvg, wikiDB);
      println("Iter Update Cnt = " + (updateCnt - lastUpdtCnt));
      
      val wdiff = checkWeight(weight, lastWeight);
      println("Weight diff = " + wdiff);
    }
    
    divdeNumber(weightSum, updateCnt.toDouble);

    weightSum;
  }
  
  def denotationLearning(queryWeight: Array[Double],
                         allTrains: ArrayBuffer[JointQueryDenotationExample],
                         featIndexer: Indexer[String],
                         wikiDB: WikipediaInterface,
                         testExs: ArrayBuffer[JointQueryDenotationExample]) : Array[Double] = {
    
    val queryPredictor = new myWikiPredictor(featIndexer, queryWeight, wikiDB);
  
    var weight = Array.fill[Double](featIndexer.size)(0);//new Array[Double](featIndexer.size());
    var weightSum = Array.fill[Double](featIndexer.size)(0);
    var lastWeight = Array.fill[Double](featIndexer.size)(0);
    
    val Iteration = 100;
    val learnRate = 0.1;
    val lambda = 1e-8;
    
    var updateCnt = 0;
    var lastUpdtCnt = 0;
    
    for (iter <- 0 until Iteration) {
      lastUpdtCnt = updateCnt;
      Array.copy(weight, 0, lastWeight, 0, weight.length);
        
      println("Deontation Iter " + iter);
      for (example <- allTrains) {
        val domains = constructDenotationDomains(queryPredictor, example, wikiDB);
        //println("candidates = " + domains.size);

        var bestLbl = -1;
        var bestScore = -Double.MaxValue;
        var bestCorrectLbl = -1; // latent best
        var bestCorrectScore = -Double.MaxValue;
        for (l <- 0 until domains.size) {
          var score = domains(l).computeScore(weight);
          if (score > bestScore) {
            bestScore = score;
            bestLbl = l;
          }
          if (domains(l).isCorrect) {
            if (score > bestCorrectScore) {
              bestCorrectScore = score;
              bestCorrectLbl = l;
            }
          }
        }
        
                
        // update?
        if (bestCorrectLbl != -1) {
        	if (!domains(bestLbl).isCorrect) {
        		updateCnt += 1;
        		if (updateCnt % 1000 == 0) println("Update " + updateCnt);
            //println(bestScore + " " + bestCorrectScore + " " + bestLbl + " " + bestCorrectLbl);
            //println(domains(bestCorrectLbl).concatenateFeats.toString());
            //println(domains(bestLbl).concatenateFeats.toString());
        		updateWeight(weight, 
        				         domains(bestCorrectLbl).concatenateFeats,
        				         domains(bestLbl).concatenateFeats,
        				         learnRate,
        				         lambda);
        		sumWeight(weightSum, weight);
        	}
        }

      }
    
      ///////////////////////////////////////////////////
      // have a test after each iteration (for learning curve)
      val tmpAvg = new Array[Double](weightSum.size)
      Array.copy(weightSum, 0, tmpAvg, 0, weightSum.size);
      divdeNumber(tmpAvg, updateCnt.toDouble);
      //testAceNerSystem(testExs, tmpAvg, Some(logger));
      //quickTest(testExs, tmpAvg);
      
      var nzCnt = 0;
      for (i <- 0 until weight.length) {
        if (weight(i) > 0) {
          nzCnt += 1;
          //println("weight(" + i + ") = " + weight(i));
        }
      }
      println("non-zero count = " + nzCnt);
      
      
      quickTestTwoStep(allTrains, queryPredictor, tmpAvg, wikiDB);
      quickTestTwoStep(testExs, queryPredictor, tmpAvg, wikiDB);
      //quickTestQuery(allTrains, tmpAvg, wikiDB);
      //quickTestQuery(testExs, tmpAvg, wikiDB);
      println("Iter Update Cnt = " + (updateCnt - lastUpdtCnt));
      
      val wdiff = checkWeight(weight, lastWeight);
      println("Weight diff = " + wdiff);
    }
    
    divdeNumber(weightSum, updateCnt.toDouble);
    weightSum;
  }
  
  
  
  
  
  
  
  
  
  
  def structurePerceptrion(allTrains: ArrayBuffer[JointQueryDenotationExample],
                           featIndexer: Indexer[String],
                           wikiDB: WikipediaInterface,
                           testExs: ArrayBuffer[JointQueryDenotationExample]) : Array[Double] = {
   
    //val logger = new PrintWriter("wiki_ace05_train.txt");
    //val logger2 = new PrintWriter("wiki_ace05_test.txt");
  
    var weight = Array.fill[Double](featIndexer.size)(0);//new Array[Double](featIndexer.size());
    var weightSum = Array.fill[Double](featIndexer.size)(0);
    var lastWeight = Array.fill[Double](featIndexer.size)(0);
    
    val Iteration = 200;
    val learnRate = 0.1;
    val lambda = 1e-8;
    
    var updateCnt = 0;
    var lastUpdtCnt = 0;
    
    var exId2 = 0;
    for (extst <- testExs) {
    	exId2 += 1;
    	val domains = constructWikiExampleDomains(extst, wikiDB);
      var corrCnt = 0;
    	for (l <- 0 until domains.size) {
    	//	logger2.println(domains(l).getFeatStr(exId2));
        if (domains(l).isCorrect) corrCnt += 1;
    	}
      if (corrCnt <= 0) {
        throw new RuntimeException("No correct value in the domain!");
      }
    }
    
    for (iter <- 0 until Iteration) {
      lastUpdtCnt = updateCnt;
      Array.copy(weight, 0, lastWeight, 0, weight.length);
        
      println("Iter " + iter);
      var exId = 0;
      for (example <- allTrains) {
        
        exId += 1;
        val domains = constructWikiExampleDomains(example, wikiDB);

        var bestLbl = -1;
        var bestScore = -Double.MaxValue;
        var bestCorrectLbl = -1; // latent best
        var bestCorrectScore = -Double.MaxValue;
        for (l <- 0 until domains.size) {
          var score = domains(l).computeScore(weight);
          if (score > bestScore) {
            bestScore = score;
            bestLbl = l;
          }
          if (domains(l).isCorrect) {
            if (score > bestCorrectScore) {
              bestCorrectScore = score;
              bestCorrectLbl = l;
            }
          }
        }
        
        //println("size = " + domains.size + " pred = " + bestLbl + " correct = " + bestCorrectLbl)
        
        // update?
        if (!domains(bestLbl).isCorrect) {
          updateCnt += 1;
          if (updateCnt % 1000 == 0) println("Update " + updateCnt);
          updateWeight(weight, 
                       domains(bestCorrectLbl).concatenateFeats,
                       domains(bestLbl).concatenateFeats,
                       learnRate,
                       lambda);
          sumWeight(weightSum, weight);
        }
      }
    
      ///////////////////////////////////////////////////
      // have a test after each iteration (for learning curve)
      val tmpAvg = new Array[Double](weightSum.size)
      Array.copy(weightSum, 0, tmpAvg, 0, weightSum.size);
      divdeNumber(tmpAvg, updateCnt.toDouble);
      //testAceNerSystem(testExs, tmpAvg, Some(logger));
      //quickTest(testExs, tmpAvg);
      
      quickTest(allTrains, tmpAvg, wikiDB);
      quickTest(testExs, tmpAvg, wikiDB);
      println("Iter Update Cnt = " + (updateCnt - lastUpdtCnt));
      
      val wdiff = checkWeight(weight, lastWeight);
      println("Weight diff = " + wdiff);
    }
    
    divdeNumber(weightSum, updateCnt.toDouble);

    weightSum;
  }
  
  def checkWeight(curWeight: Array[Double],
                  oldWeight: Array[Double]): Int = {
    var different: Int = 0;
    for (i <- 0 until curWeight.length) {
      if (curWeight(i) != oldWeight(i)) different += 1;
    }
    different;
  }
  
  
  // first predict query, then wiki title
  def quickTestTwoStep(testExs: ArrayBuffer[JointQueryDenotationExample],
                       queryPicker: myWikiPredictor,
                       weight: Array[Double],
                       wikiDB: WikipediaInterface) {
    
    var correct: Double = 0.0;
    var nocrr: Double = 0.0;
    
    for (ex  <- testExs) {
      val domains = constructDenotationDomains(queryPicker, ex, wikiDB);
      var bestLbl = -1;
      var bestScore = -Double.MaxValue;
      var bestCorrectLbl = -1; // latent best
      var bestCorrectScore = -Double.MaxValue;
      for (l <- 0 until domains.size) {
          var score = domains(l).computeScore(weight);
          if (score > bestScore) {
            bestScore = score;
            bestLbl = l;
          }
          if (domains(l).isCorrect) {
            if (score > bestCorrectScore) {
              bestCorrectScore = score;
              bestCorrectLbl = l;
            }
          }
      }
      if (bestCorrectLbl != -1) {
    	  if (domains(bestLbl).isCorrect) {
    		  correct += 1.0;
    	  }
      } else {
        nocrr += 1.0;
      }
    }
    
    var total = testExs.size.toDouble;
    var acc = correct / total;
    println("Acc: " + correct + "/" + total + " = " + acc + ", no-corr: " + nocrr);
  }
  
  
  def quickTestQuery(testExs: ArrayBuffer[JointQueryDenotationExample],
                     weight: Array[Double],
                     wikiDB: WikipediaInterface) {
    
    var correct: Double = 0.0;
    
    for (ex  <- testExs) {
      val domains = constructQueryDomains(ex, wikiDB);
      var bestLbl = -1;
      var bestScore = -Double.MaxValue;
      for (l <- 0 until domains.size) {
          var score = domains(l).computeScore(weight);
          //println(score);
          if (score > bestScore) {
            bestScore = score;
            bestLbl = l;
          }
      }
      if (domains(bestLbl).isCorrect) {
        correct += 1.0;
      }
    }
    
    var total = testExs.size.toDouble;
    var acc = correct / total;
    println("Query acc: " + correct + "/" + total + " = " + acc);
  }
  
  def quickTest(testExs: ArrayBuffer[JointQueryDenotationExample],
                weight: Array[Double],
                wikiDB: WikipediaInterface) {
    
    var correct: Double = 0.0;
    
    for (ex  <- testExs) {
      val domains = constructWikiExampleDomains(ex, wikiDB);
      var bestLbl = -1;
      var bestScore = -Double.MaxValue;
      for (l <- 0 until domains.size) {
          var score = domains(l).computeScore(weight);
          //println(score);
          if (score > bestScore) {
            bestScore = score;
            bestLbl = l;
          }
      }
      if (domains(bestLbl).isCorrect) {
        correct += 1.0;
      }
    }
    
    var total = testExs.size.toDouble;
    var acc = correct / total;
    println("Acc: " + correct + "/" + total + " = " + acc);
  }
  
  def updateWeight(currentWeight: Array[Double], 
                   featGold: Array[Int],
                   featPred: Array[Int],
                   eta: Double,
                   lambda: Double) {
    var gradient = Array.fill[Double](currentWeight.length)(0);//new Array[Double](currentWeight.length);
    
    
    val possibleNonZeroIdx = new ArrayBuffer[Int]();
    
    for (i <- featGold) {
      if (i >= 0) {
        gradient(i) += (1.0);
        possibleNonZeroIdx += (i);
      }
    }
    for (j <- featPred) {
      if (j >= 0) {
        gradient(j) -= (1.0);
        possibleNonZeroIdx += (j);
      }
    }
    /*
    var cnt = 0;
    for (nzidx <- possibleNonZeroIdx) {
      if (gradient(nzidx) != 0) cnt += 1;
    }*/
    
    //println("gradient non-zero element cnt: " + cnt);
    
    // do L2 Regularization
    //var l1norm = getL1Norm(currentWeight);
    for (i2 <- 0 until currentWeight.length) {
      //var regularizerNum: Double = Math.max(0, b);
      //var regularizerDen: Double = Math.max(0, b);
      var reg: Double = 1.0 - (eta * lambda)
      var curWeightVal = currentWeight(i2);// * reg;
      currentWeight(i2) = curWeightVal + (gradient(i2) * eta);
      //currentWeight(i2) += (gradient(i2) * eta);
    }
  }
  
  def computeScore(wght: Array[Double], feat: Array[Int]) : Double = {
    var result : Double = 0;
    for (idx <- feat) {
      if (idx >= 0) {
        result += (wght(idx));
      }
    }
    result;
  }
  
  def sumWeight(sum: Array[Double], w: Array[Double]) {
    for (i <- 0 until w.length) {
      sum(i) += w(i);
    }
  }
  
  def divdeNumber(w: Array[Double], deno: Double) {
    for (i <- 0 until w.length) {
      w(i) = (w(i) / deno);
    }
  }
  
  def getL1Norm(w: Array[Double]): Double = {
    var norm: Double = 0;
    for (v <- w) {
      norm += Math.abs(v);
    }
    return norm;
  }
  
}
