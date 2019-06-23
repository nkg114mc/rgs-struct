/*
package berkeleyentity.oregonstate

import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._
import scala.util.Random
import scala.util.Sorting
import scala.math.Ordering
import scala.util.control.Breaks._

import java.util.ArrayList

import edu.berkeley.nlp.futile.fig.basic.Indexer
import berkeleyentity.ner.NerFeaturizer
import berkeleyentity.coref.CorefDoc
import berkeleyentity.coref.PairwiseIndexingFeaturizer
import berkeleyentity.coref.OrderedClustering
import berkeleyentity.coref.DocumentGraph
import berkeleyentity.GUtil
import berkeleyentity._
import berkeleyentity.ner.NerSystemLabeled
import edu.berkeley.nlp.futile.util.Logger
import berkeleyentity.coref.PairwiseLossFunctions
import berkeleyentity.Driver
import berkeleyentity.bp.UnaryFactorGeneral
import berkeleyentity.ner.MCNerFeaturizer
import berkeleyentity.wiki._
import berkeleyentity.joint.LikelihoodAndGradientComputer
import berkeleyentity.joint.JointDocFactorGraph
import berkeleyentity.joint.JointDocFactorGraphACE
import berkeleyentity.joint.FactorGraphFactory
import berkeleyentity.joint.JointLossFcns
import berkeleyentity.bp.Domain
import berkeleyentity.oregonstate.pruner.StaticDomainPruner

import gurobi._


class ILPModelInfo(// Add coref variables
      val corefVars: HashMap[Int, HashMap[Int, GRBVar]], 
      val corefValues: HashMap[Int, Array[Int]],
      // Add ner variables
      val nerVars: HashMap[Int, HashMap[Int, GRBVar]],
      val nerValues: HashMap[Int, Array[String]],
      // Add wiki variables
      val wikiVars: HashMap[Int, HashMap[Int, GRBVar]],
      val wikiValues: HashMap[Int, Array[String]],
      // Add coref+ner variables
      val cnijVars: ArrayList[GRBVar],
      val cnijMap: HashMap[String, ArrayList[GRBVar]],
      // Add coref+wiki variables
      val cwijVars: ArrayList[GRBVar],
      val cwijMap: HashMap[String, ArrayList[GRBVar]],
      // Add ner+wiki variables
      val nwiVars: ArrayList[GRBVar],
      val nwiMap: HashMap[Int, ArrayList[GRBVar]]) {

}

class ILPJointInferencer {
  
  def extractQueryWikiDomain(factorGraph: JointDocFactorGraphACE, idx: Int): Domain[QueryWikiValue] = {
    
    val qwvalues = new ArrayBuffer[QueryWikiValue]();

    val qn = factorGraph.queryNodes(idx);
    val quf = factorGraph.queryUnaryFactors(idx);
    val wn = factorGraph.wikiNodes(idx);
    val qwbf = factorGraph.queryWikiBinaryFactors(idx);
    
    val wikiDB = factorGraph.wikiDB.get;
    val queries = qn.domain.entries;
    val denotations = wn.domain.entries;
    val queryOutcomes = queries.map(query => wikiDB.disambiguateBestGetAllOptions(query));
    
    for (qidx <- 0 until qn.domain.size) {
      for (didx <- 0 until wn.domain.size) {
          val q = qn.domain.value(qidx);
          val d = wn.domain.value(didx);
          if ((d.equals(ExcludeToken)) || (d.equals(NilToken)) || (queryOutcomes(qidx).containsKey(d))) {
            val fq = quf.indexedFeatures(qidx);
            val fqd = qwbf.indexedFeatureMatrix(qidx)(didx);
            val isCorct = false;//isCorrect(ex.rawCorrectDenotations, d);
            qwvalues += (new QueryWikiValue(q, d, fq, fqd, isCorct));
          }
      }
    }
    
    val domain = new Domain[QueryWikiValue](qwvalues.toArray);
    domain;
  }
  
  def computeScore(wght: Array[Double], feat: Array[Int]) : Double = {
    var result : Double = 0;
    for (idx <- feat) {
      if (idx >= 0) {
        result += (wght(idx).toDouble);
      }
    }
    result;
  }
  
  def getIlpVarValue(varvalues: HashMap[Int, HashMap[Int, GRBVar]], idx: Int, vidx: Int) = {
    var result: GRBVar = null;
    if (varvalues.get(idx) != None) {
      val values = varvalues(idx);
      result = values(vidx);
    }
    result;
  }
  
  def getIlpValue[T](varvalues: HashMap[Int, Array[T]], idx: Int, vidx: Int): T = {
    var result: T = if (varvalues.get(idx) != None) {
      val values = varvalues.get(idx).get;
      values(vidx);
    } else {
      ???;
    }
    result;
  }
  
  
  // return an array of value indices in the value array
  def getVarDomain[T](variable: IndepVariable[T], useGold: Boolean, referenceIdx: Int, useRef: Boolean): Array[Int] = {
    var result : Array[Int] = variable.getAllNonPruningValueIndices(); // origin domain
    // are we going to use correct values?
    if (useGold) {
      result = variable.getCorrectNonPruningValueIndices();
      if (result.length == 0) { // no correct value ...
    	  result = new Array[Int](1);
    	  result(0) = variable.getSortedNonPruningValueIndices()(0); // pick a best value from unary score
      }
    }
    // any fixed value as discrpendy?
    if (useRef) {
    	if (referenceIdx >= 0) {
    		result = new Array[Int](1);
    		result(0) = referenceIdx;
    	}
    }

    result;
  }
  
  //def getVarDomainFromExample(example: AceJointTaskExample, singleTaskIdx: Int, useGold: Boolean, referenceOutput: Array[Int]): Array[Int] = {
  //}
  
  def runILPinference(example: AceJointTaskExample, wght: Array[Double], useGold: Boolean, referOutput: Array[Int], useReference: Boolean) = {
    
    // 
    if (useReference == false) {
      val allMinusOne = Array.fill[Int](referOutput.length)(-1);
      Array.copy(allMinusOne, 0, referOutput, 0, referOutput.length)
    }
    ///////////////////////////////////////////

      val env = new GRBEnv();
      val model = new GRBModel(env);
      /////////////////////////////////////////
      
      
      val docGraph = example.docGraph;
      val varCoeffMap = new HashMap[GRBVar, Double]();
      //val neLabelIndexer = MCNerFeaturizer.StdLabelIndexer;
      
      //////////////////////////////////
      ///// single task variables //////
      //////////////////////////////////
 
      // Add coref variables
      val corefVars = new HashMap[Int, HashMap[Int, GRBVar]]();
      val corefValues = new HashMap[Int, Array[Int]]();
      for (i <- 0 until docGraph.size) {
        val cNode = example.getCorefVar(i);//factorGraph.corefNodes(i); //val cuf = factorGraph.corefUnaryFactors(i);
        val possibleVals = getVarDomain[Int](cNode, useGold, example.getCorefGlobalOutputValue(i, referOutput), useReference);//cNode.getAllNonPruningValueIndices();
        val corefVarAndValues = new HashMap[Int, GRBVar]();//ArrayList[GRBVar]();
        val currentCorefValues = new Array[Int](i + 1);//(possibleVals.size);
        for (j <- possibleVals) { //}0 until possibleVals.size) {
          val jantIdx = cNode.values(j).value;
          val currProb = computeScore(wght, cNode.values(j).feature);//0.00;// cNode.cachedBeliefsOrMarginals(j);// allProbs(i);
          val vName = "VCoref@" + String.valueOf(i) + "@" + String.valueOf(j) + "@" + jantIdx;
          val newVar = model.addVar(0.0, 1.0, currProb, GRB.BINARY, vName);
          //corefVarAndValues.add(newVar);
          corefVarAndValues += (j -> newVar);
          currentCorefValues(j) = (jantIdx);
        }
        corefVars += (i -> corefVarAndValues);
        corefValues += (i -> currentCorefValues);
      }
      
      // Add ner variables
      //val nerVars = new HashMap[Int, ArrayList[GRBVar]]();//new ArrayList[ArrayList[GRBVar]]();//new GRBVar(curId + 1);
      val nerVars = new HashMap[Int, HashMap[Int, GRBVar]]();
      val nerValues = new HashMap[Int, Array[String]]();
      for (i <- 0 until docGraph.size) {
        val nNode = example.getNerVar(i);//factorGraph.nerNodes(i); //val nuf = factorGraph.nerUnaryFactors(i);
        val possibleVals = getVarDomain[String](nNode, useGold, example.getNerGlobalOutputValue(i, referOutput), useReference);//nNode.getAllNonPruningValueIndices();
        val nerVarAndValues = new HashMap[Int, GRBVar]();//new ArrayList[GRBVar]();
        val currentNerValues = new ArrayBuffer[String]();//new Array[String](possibleVals.size);
        for (j <- possibleVals) { //}0 until possibleVals.size) {
          //val jvalIdx = possibleVals(j);
          val currProb = computeScore(wght, nNode.values(j).feature);//0.00;//nNode.cachedBeliefsOrMarginals(j);// allProbs(i);
          val vName = "VNer@" + String.valueOf(i) + "@" + String.valueOf(j) + "@" + nNode.values(j).value;
          val newVar = model.addVar(0.0, 1.0, currProb, GRB.BINARY, vName);
          //nerVarAndValues.add(newVar);
          nerVarAndValues += (j -> newVar);
          currentNerValues += (nNode.values(j).value);
        }
        nerVars += (i -> nerVarAndValues);
        nerValues += (i -> currentNerValues.toArray);
      }
      
      
      //val wikiDenotationIndexer = new HashMap[Int, HashMap[String, Int]]();
      
      // Add wiki variables
      //val wikiVars = new HashMap[Int, ArrayList[GRBVar]]();
      val wikiVars = new HashMap[Int, HashMap[Int, GRBVar]]();
      val wikiValues = new HashMap[Int, Array[String]]();
      for (i <- 0 until docGraph.size) {
        val wNode = example.getWikiVar(i);
        val possibleVals = getVarDomain[QueryWikiValue](wNode, useGold, example.getWikiGlobalOutputValue(i, referOutput), useReference);//wNode.getAllNonPruningValueIndices();  //extractQueryWikiDomain(factorGraph, i);// wNode.domain;
        val wikiVarAndValues = new HashMap[Int, GRBVar]();//new ArrayList[GRBVar]();
        val currentqWikiValues = new ArrayBuffer[String]();//new Array[String](possibleVals.size);
        for (j <- possibleVals) { //0 until possibleVals.size) {
          val currProb = computeScore(wght, wNode.values(j).feature);//0.00;//wNode.cachedBeliefsOrMarginals(j);// allProbs(i);
          val vName = "VWiki@" + String.valueOf(i) + "@" + String.valueOf(j) + "@" + wNode.values(j).value.wiki;
          val newVar = model.addVar(0.0, 1.0, currProb, GRB.BINARY, vName);
          
          //wikiVarAndValues.add(newVar);
          wikiVarAndValues += (j -> newVar);
          currentqWikiValues += (wNode.values(j).value.wiki);
        }
        wikiVars += (i -> wikiVarAndValues);
        wikiValues += (i -> currentqWikiValues.toArray);
        
        /// build denotation indexer
        //val iDenoIdxer = new HashMap[String, Int]();
        //for (k <- 0 until factorGraph.wikiNodes(i).domain.size) {
        //  iDenoIdxer += (factorGraph.wikiNodes(i).domain.value(k) -> k);
        //}
        //wikiDenotationIndexer += (i -> iDenoIdxer);
      }
      
      model.update();

      

      //////////////////////////////////
      ///// cross task variables ///////
      //////////////////////////////////

      // Add coref+ner variables
      val cnijVars = new ArrayList[GRBVar]();
      val cnijMap = new HashMap[String, ArrayList[GRBVar]]();

      for (iIdx <- 0 until docGraph.size) {
        val corefVarAndValues = corefVars(iIdx);
        val curCorefValues = corefValues(iIdx);
        val i = iIdx;
        for ((jIdx, cvar) <- corefVarAndValues) {
          //val cvar = corefVarAndValues.get(jIdx);
          //println(cvar.toString());
          val cvarName = cvar.get(GRB.StringAttr.VarName);
          //val vIdx = parseNameInt(cvarName, 2);
          val j = curCorefValues(jIdx);
          // i and j are different
          if (j != i) {
            val cnuf = example.corefNerFactors(i)(j);
            val ijstr = String.valueOf(i) + "@" + String.valueOf(jIdx);
            val ijners = new ArrayList[GRBVar]();
            
            val njVal = getVarDomain[String](example.getNerVar(j), useGold, example.getNerGlobalOutputValue(j, referOutput), useReference);//example.getNerVar(j).getAllNonPruningValueIndices();//nerValues.get(j).get;
            val niVal = getVarDomain[String](example.getNerVar(i), useGold, example.getNerGlobalOutputValue(i, referOutput), useReference);//example.getNerVar(i).getAllNonPruningValueIndices();//nerValues.get(i).get;
            for (niIdx <- niVal) { // 0 until niVal.size) {
              for (njIdx <- njVal) { //0 until njVal.size) {
                val currProb = computeScore(wght, cnuf.feature(niIdx)(njIdx));//0.000;//cNode.cachedBeliefsOrMarginals(jIdx) + nNodei.cachedBeliefsOrMarginals(niIdx);
                val vName = "VCorefNer@" + String.valueOf(i) + "@" + String.valueOf(jIdx) + "@" + String.valueOf(niIdx) + "@" + String.valueOf(njIdx) + "@" + example.getNerVar(i).values(niIdx).value + "@" + example.getNerVar(j).values(njIdx).value;
                val newVar = model.addVar(0.0, 1.0, currProb, GRB.BINARY, vName);
                cnijVars.add(newVar);
                ijners.add(newVar);
              }
            }// end of i != j
            cnijMap += (ijstr -> ijners);
          }
        }
      }


      ///////////////////////////////////////////////
      // Add ner+wiki variables
      val nwiVars = new ArrayList[GRBVar]();
      val nwiMap = new HashMap[Int, ArrayList[GRBVar]](); // N_i and E_i map => vars
      // variables
      for (i <- 0 until docGraph.size) {
        val niVals = getVarDomain[String](example.getNerVar(i), useGold, example.getNerGlobalOutputValue(i, referOutput), useReference);//example.getNerVar(i).getAllNonPruningValueIndices();
        val wiVals = getVarDomain[QueryWikiValue](example.getWikiVar(i), useGold, example.getWikiGlobalOutputValue(i, referOutput), useReference);//example.getWikiVar(i).getAllNonPruningValueIndices();
        val iVarAndValues = new ArrayList[GRBVar](); // at position i only        
        val nwfactor = example.nerWikiFactors(i);//factorGraph.wikiNerFactors(i);
        for (nvIdx <- niVals) { //0 until niVals.size) {
          for (wvIdx <- wiVals) { //0 until wiVals.size) {
            val currProb = computeScore(wght, nwfactor.feature(nvIdx)(wvIdx));
            val vName = "VNerWiki@" + String.valueOf(i) + "@" +  String.valueOf(nvIdx) + "@" + String.valueOf(wvIdx);
            val newVar = model.addVar(0.0, 1.0, currProb, GRB.BINARY, vName);
            nwiVars.add(newVar);
            iVarAndValues.add(newVar);
          }
        }
        nwiMap += (i -> iVarAndValues);
      }

      model.update();

      
      
      
      ///////////////////////////////////////
      ///////// Adding constraints //////////
      ///////////////////////////////////////

      
      // single variable constraints
      ///////////////////////////////
      
      // Add coref constraints
      for (i <- 0 until docGraph.size) {
        val exprCoref = new GRBLinExpr();
        val menValues = corefVars(i).values;
        val varsArr = menValues.toArray;//.toArray(new Array[GRBVar](0));
        exprCoref.addTerms(null, varsArr);
        val constraintName = "Constr_Coref_" + String.valueOf(i) + "_Val";
        model.addConstr(exprCoref, GRB.EQUAL, 1.0, constraintName);
      }
      
      // Add ner constraints
      for (i <- 0 until docGraph.size) {
        val exprNer = new GRBLinExpr();
        val menValues = nerVars(i).values;
        val varsArr = menValues.toArray;//.toArray(new Array[GRBVar](0));
        exprNer.addTerms(null, varsArr);
        val constraintName = "Constr_Ner_" + String.valueOf(i) + "_Val";
        model.addConstr(exprNer, GRB.EQUAL, 1.0, constraintName);
      }
      
      // add wiki constraints
      for (i <- 0 until docGraph.size) {
        val exprWiki = new GRBLinExpr();
        val menValues = wikiVars(i).values;
        val varsArr = menValues.toArray;//.toArray(new Array[GRBVar](0));
        exprWiki.addTerms(null, varsArr);
        val constraintName = "Constr_Wiki_" + String.valueOf(i) + "_Val";
        model.addConstr(exprWiki, GRB.EQUAL, 1.0, constraintName);
      }
      
      model.update();


      // Add consistency constraints
      ////////////////////////////////
      // cn consistency
      for ((ijstr, cnlist) <- cnijMap) {
        for (jj <- 0 until cnlist.size) {
          val combinedVar = cnlist.get(jj);
          val vname = combinedVar.get(GRB.StringAttr.VarName);
          //println(vname);
          val iIdx = parseNameInt(ijstr, 0);
          val jvalueIdx = parseNameInt(ijstr, 1);
          val ivIdx = parseNameInt(vname, 3);
          val jvIdx = parseNameInt(vname, 4);
          val cvar = getIlpVarValue(corefVars, iIdx, jvalueIdx); // x1
          val jIdx = getIlpValue[Int](corefValues, iIdx, jvalueIdx);
          val nivar = getIlpVarValue(nerVars, iIdx, ivIdx); // x2
          val njvar = getIlpVarValue(nerVars, jIdx, jvIdx); // x3
          constructAddConstraint3Var(cvar, nivar, njvar, combinedVar, model, "Constr_CorefNer_" + ijstr + "@" + ivIdx + "@" + jvIdx);
          //println(parseNameStr(nivar.get(GRB.StringAttr.VarName), 3) + " & " + parseNameStr(njvar.get(GRB.StringAttr.VarName), 3) + " = " + parseNameStr(vname, 5) + "_" + parseNameStr(vname, 6));
        }
      }

      // nw consistency
      for ((ii, nwlist) <- nwiMap) {
        for (jj <- 0 until nwlist.size) {
          val combinedVar = nwlist.get(jj);
          val vname = combinedVar.get(GRB.StringAttr.VarName);
          val iIdx = parseNameInt(vname, 1);//ii;
          val ivIdx = parseNameInt(vname, 2);
          val jvIdx = parseNameInt(vname, 3);
          val nivar = getIlpVarValue(nerVars, iIdx, ivIdx); // x1
          val wivar = getIlpVarValue(wikiVars, iIdx, jvIdx); // x2
          constructAddConstraint2Var(nivar, wivar, combinedVar, model, "Constr_NerWiki_" + iIdx + "@" + ivIdx + "@" + jvIdx);
        }
        //val constraintName = "Constr_NerWiki_" + ii + "_Val";
        //model.addConstr(exprNerWiki, GRB.EQUAL, 1.0, constraintName);
      }

      
      model.update();
 
      
      ///// About objective /////
      val exprObjective = new GRBLinExpr();
      
      // single task terms /////////////////////////////
      // coref coefficiency
      for (i <- 0 until docGraph.size) {
        //val vars = corefVars.get(i).get;
        val vars = corefVars(i);
        for ((k, v) <- vars) {
          val coeff = v.get(GRB.DoubleAttr.Obj);
          exprObjective.addTerm(coeff, v);
        }
      }
      // ner coefficiency
      for (i <- 0 until docGraph.size) {
        //val vars = nerVars.get(i).get;
        val vars = nerVars(i);
        for ((k, v) <- vars) {
          val coeff = v.get(GRB.DoubleAttr.Obj);
          exprObjective.addTerm(coeff, v);
        }
      }
      // wiki coefficiency
      for (i <- 0 until docGraph.size) {
        //val vars = wikiVars.get(i).get;
        val vars = wikiVars(i);
        for ((k, v) <- vars) {
          val coeff = v.get(GRB.DoubleAttr.Obj);
          exprObjective.addTerm(coeff, v);
        }
      }

      // cross task terms ////////////////////////////////
      // coref+ner coefficiency
      for ((ijstr, cnlist) <- cnijMap) {
        for (jj <- 0 until cnlist.size) {
          val coeff = cnlist.get(jj).get(GRB.DoubleAttr.Obj);
          exprObjective.addTerm(coeff, cnlist.get(jj));
        }
      }

      // Add ner+wiki coefficiency
      for ((ii, nwlist) <- nwiMap) {
        for (jj <- 0 until nwlist.size) {
          val coeff = nwlist.get(jj).get(GRB.DoubleAttr.Obj);
          exprObjective.addTerm(coeff, nwlist.get(jj));
        }
      }

      model.update();
      
      /*
      for (i2 <- 0 until exprObjective.size()) {
        val varName = exprObjective.getVar(i2).get(GRB.StringAttr.VarName);
        val vCoef = exprObjective.getCoeff(i2);
        println(vCoef+" * "+varName);
      }
      */
      
      model.setObjective(exprObjective);
      model.update();
      //println(model.get);
      
      
      // set inital values
      
      
      ///// About the model /////
      model.set(GRB.IntAttr.ModelSense, -1);
      model.update();

      
      // Optimize model! 
      model.optimize();

    	println("optimization status: " + model.get(GRB.IntAttr.Status).toString());
      
      
      // about Irreducable Infeasible Set 

      model.update();
      
      val varCnt = model.getVars().length;
      val varCnt1 = model.getVars().filter{ p => p.get(GRB.StringAttr.VarName).contains("VCoref@") }.length;
      val varCnt2 = model.getVars().filter{ p => p.get(GRB.StringAttr.VarName).contains("VNer@") }.length;
      val varCnt3 = model.getVars().filter{ p => p.get(GRB.StringAttr.VarName).contains("VWiki@") }.length;
      val varCnt4 = model.getVars().filter{ p => p.get(GRB.StringAttr.VarName).contains("VCorefNer@") }.length;
      val varCnt5 = model.getVars().filter{ p => p.get(GRB.StringAttr.VarName).contains("VCorefWiki@") }.length;
      val varCnt6 = model.getVars().filter{ p => p.get(GRB.StringAttr.VarName).contains("VNerWiki@") }.length;
      val constrCnt = model.getConstrs.length;
      

      val constrCnt1 = model.getConstrs.filter{ p => p.get(GRB.StringAttr.ConstrName).contains("Constr_Coref_") }.length; // val;
      val constrCnt2 = model.getConstrs.filter{ p => p.get(GRB.StringAttr.ConstrName).contains("Constr_Ner_") }.length; // val
      val constrCnt3 = model.getConstrs.filter{ p => p.get(GRB.StringAttr.ConstrName).contains("Constr_Wiki_") }.length; // val
      
      val constrCnt4 = model.getConstrs.filter{ p => p.get(GRB.StringAttr.ConstrName).contains("Constr_CorefNer_") }.length; // pair
      val constrCnt5 = model.getConstrs.filter{ p => p.get(GRB.StringAttr.ConstrName).contains("Constr_CorefWiki_") }.length; // pair
      val constrCnt6 = model.getConstrs.filter{ p => p.get(GRB.StringAttr.ConstrName).contains("Constr_NerWiki_") }.length; // val
      //val constrCnt6 = model.getConstrs.filter{ p => p.get(GRB.StringAttr.ConstrName).contains("Constr_iCoref") }.length; // cross
      //val constrCnt7 = model.getConstrs.filter{ p => p.get(GRB.StringAttr.ConstrName).contains("Constr_iNer") }.length; // cross
      //val constrCnt8 = model.getConstrs.filter{ p => p.get(GRB.StringAttr.ConstrName).contains("Constr_iWiki") }.length; // cross

      println("Mention count = " + example.numMentions);
      println("Total vars cnt = " + varCnt + " = " + varCnt1 + "+" + varCnt2+ "+" + varCnt3);
      println("Total combined_vars cnt = " +  varCnt4 + "+" + varCnt5+ "+" + varCnt6);
      println("Constrs: " + constrCnt + " = " + constrCnt1 + "+" + constrCnt2 + "+" + constrCnt3 + "+" + constrCnt4 + "+" + constrCnt5 + "+" + constrCnt6);// + "+" + constrCnt7 + "+" + constrCnt8);

      /////////////////////////////////
      // output result
      //val predBackptrs = outputCorefBackpointers(docGraph, corefVars, corefValues);
      //val nerChunks = outputPairNERChunks(docGraph, nerVars, nerValues);
      //val wikiChunks = outputPairWikificationChunks(docGraph, wikiVars, wikiValues);
      //(predBackptrs, OrderedClustering.createFromBackpointers(predBackptrs), nerChunks, wikiChunks);
      
      //val ilpInfo = new ILPModelInfo(corefVars, corefValues, nerVars, nerValues, wikiVars, wikiValues, cnijVars, cnijMap, cwijVars, cwijMap, nwiVars,nwiMap);
      //checkOutputConsistency(ilpInfo);
      
      val returnOutput = constructResultOutput(example, corefVars, nerVars, wikiVars);
      returnOutput;
    ///////////////////////////////////////////
  }
  
  def parseNameStr(varName: String, idx: Int) : String = {
    val strArr = varName.split("@");
    val id = strArr(idx);
    id;
  }
  def parseNameInt(varName: String, idx: Int) : Int = {
    val strArr = varName.split("@");
    val id = Integer.parseInt(strArr(idx));
    id;
  }
  
  def constructAddConstraint3Var(x1: GRBVar, x2: GRBVar, x3: GRBVar, y: GRBVar, model: GRBModel, nameHead: String) {
    // x1 + x2 + x3 - y <= 2
    val expr1Left = new GRBLinExpr();
    expr1Left.addTerm( 1.0, x1);
    expr1Left.addTerm( 1.0, x2);
    expr1Left.addTerm( 1.0, x3);
    expr1Left.addTerm(-1.0, y);
    model.addConstr(expr1Left, GRB.LESS_EQUAL, 2.0, nameHead + "_Consist1");
    // y - x1 <= 0
    val expr2Left = new GRBLinExpr();
    expr2Left.addTerm( 1.0, y);
    expr2Left.addTerm(-1.0, x1);
    model.addConstr(expr2Left, GRB.LESS_EQUAL, 0, nameHead + "_Consist2");
    // y - x2 <= 0
    val expr3Left = new GRBLinExpr();
    expr3Left.addTerm( 1.0, y);
    expr3Left.addTerm(-1.0, x2);
    model.addConstr(expr3Left, GRB.LESS_EQUAL, 0, nameHead + "_Consist3");
    // y - x3 <= 0
    val expr4Left = new GRBLinExpr();
    expr4Left.addTerm( 1.0, y);
    expr4Left.addTerm(-1.0, x3);
    model.addConstr(expr4Left, GRB.LESS_EQUAL, 0, nameHead + "_Consist4");
  }
  
  def constructAddConstraint2Var(x1: GRBVar, x2: GRBVar, y: GRBVar, model: GRBModel, nameHead: String) {
    // x1 + x2 - y <= 1
    val expr1Left = new GRBLinExpr();
    expr1Left.addTerm( 1.0, x1);
    expr1Left.addTerm( 1.0, x2);
    expr1Left.addTerm(-1.0, y);
    model.addConstr(expr1Left, GRB.LESS_EQUAL, 1.0, nameHead + "_Consist1");
    // y - x1 <= 0
    val expr2Left = new GRBLinExpr();
    expr2Left.addTerm( 1.0, y);
    expr2Left.addTerm(-1.0, x1);
    model.addConstr(expr2Left, GRB.LESS_EQUAL, 0, nameHead + "_Consist2");
    // y - x2 <= 0
    val expr3Left = new GRBLinExpr();
    expr3Left.addTerm( 1.0, y);
    expr3Left.addTerm(-1.0, x2);
    model.addConstr(expr3Left, GRB.LESS_EQUAL, 0, nameHead + "_Consist3");
  }
  
  def constructResultOutput(example: AceJointTaskExample, 
                            corefVars: HashMap[Int, HashMap[Int, GRBVar]], 
                            nerVars: HashMap[Int, HashMap[Int, GRBVar]], 
                            wikiVars: HashMap[Int, HashMap[Int, GRBVar]]) = {
    
    val output = new Array[Int](example.totalSize);
    
    for (j2 <- 0 until corefVars.size) {
      val vari = corefVars.get(j2).get;
      val result1 = pickBestvIdx(vari);
      //println("bestCoref("+j2+") = " + result1);
      val cidx = example.getCorefGlobalIndex(j2);
      output(cidx) = result1;
    }

    for (j2 <- 0 until nerVars.size) {
      val vari = nerVars.get(j2).get;
      val result2 = pickBestvIdx(vari);
      //println("bestCoref("+j2+") = " + result1);
      val nidx = example.getNerGlobalIndex(j2);
      output(nidx) = result2;
    }
           
    for (j2 <- 0 until wikiVars.size) {
      val vari = wikiVars.get(j2).get;
      val result3 = pickBestvIdx(vari);
      //println("bestCoref("+j2+") = " + result1);
      val widx = example.getWikiGlobalIndex(j2);
      output(widx) = result3;
    }
    
    output;
  }
  
  // return value index
  def pickBestvIdx(vars: HashMap[Int, GRBVar]): Int = {
	  var bestVal: Int = -1;
	  //for (j <- 0 until vars.size) {
    val allVar = vars.values;
    for (variValj <- allVar) { 
		  //val variValj = vars.get(j);
		  val jvalIdx = parseNameInt(variValj.get(GRB.StringAttr.VarName), 2);
		  //println("coref varname = " + variValj.get(GRB.StringAttr.VarName));
		  val ilpVal = variValj.get(GRB.DoubleAttr.X);
		  if (ilpVal > 0) {
			  bestVal = jvalIdx;
		  }
	  }
	  bestVal;
  }
  
  //////////////
  // SOLUTION //
  //////////////
  // for solutions
  def outputCorefBackpointers(docGraph: DocumentGraph, 
                             corefVars: HashMap[Int, ArrayList[GRBVar]], 
                             corefValues: HashMap[Int, Array[Int]]) = {
    val result = new Array[Int](corefVars.size);
    for (j2 <- 0 until corefVars.size) {
      val vari = corefVars.get(j2).get;
      val vali = corefValues.get(j2).get;
      val result1 = pickBestIndex(vari, vali);
      println("bestCoref("+j2+") = " + result1);
      result(j2) = result1;
    }
    result;
  }
  def pickBestIndex(vari: ArrayList[GRBVar], vali: Array[Int]): Int = {
    var bestVal = -1;
    for (j <- 0 until vari.size) {
      val variValj = vari.get(j);
      val jvalIdx = parseNameInt(variValj.get(GRB.StringAttr.VarName), 2);
      println("coref varname = " + variValj.get(GRB.StringAttr.VarName));
      val ilpVal = variValj.get(GRB.DoubleAttr.X);
      if (ilpVal > 0) {
        bestVal = vali(jvalIdx);
      }
    }
    bestVal;
  }
  
  
  def outputPairNERChunks(docGraph: DocumentGraph, 
                          nerVars: HashMap[Int, ArrayList[GRBVar]], 
                          nerValues: HashMap[Int, Array[String]]) = {
    val result = new ArrayBuffer[String]();//(vars.size);
    for (j2 <- 0 until nerVars.size) {
      var bestIdx: Int = -1;
      var bestVal = "????";
      val vari = nerVars.get(j2).get;
      val vali = nerValues.get(j2).get;
      for (i2 <- 0 until vari.size) {
        val variValj = vari.get(i2);
        val nIdx = parseNameInt(variValj.get(GRB.StringAttr.VarName), 2);
        val ilpVal = variValj.get(GRB.DoubleAttr.X);
        if (ilpVal > 0) {
          bestIdx = i2;
          bestVal = vali(nIdx);
        }
      }
      println("bestNer("+j2+") = " + bestVal);
      result += bestVal;
    }
    chunkifyMentionAnnots(docGraph, result.toSeq);
  } 
  
  def outputPairWikificationChunks(docGraph: DocumentGraph, 
                                   wikiVars: HashMap[Int, ArrayList[GRBVar]], 
                                   wikiValues: HashMap[Int, Array[String]]) = {
    val result = new ArrayBuffer[String]();//(vars.size);
    for (j2 <- 0 until wikiVars.size) {
      var bestIdx: Int = -1;
      var bestVal = "????";
      val vari = wikiVars.get(j2).get;
      val vali = wikiValues.get(j2).get;
      for (i2 <- 0 until vari.size) {
        val variValj = vari.get(i2);
        val nIdx = parseNameInt(variValj.get(GRB.StringAttr.VarName), 2);
        val ilpVal = variValj.get(GRB.DoubleAttr.X);
        if (ilpVal > 0) {
          bestIdx = i2;
          bestVal = vali(nIdx);
        }
      }
      println("bestWiki("+j2+") = " + bestVal);
      result += bestVal;
    }
    chunkifyMentionAnnots(docGraph, result.toSeq);
  }

  private def chunkifyMentionAnnots(docGraph: DocumentGraph, mentAnnots: Seq[String]) = {
    val chunksPerSentence = (0 until docGraph.corefDoc.rawDoc.numSents).map(i => new ArrayBuffer[Chunk[String]]);
    for (i <- 0 until docGraph.getMentions.size) {
      val ment = docGraph.getMention(i);
      chunksPerSentence(ment.sentIdx) += new Chunk[String](ment.startIdx, ment.endIdx, mentAnnots(i));
    }
    chunksPerSentence;
  }

  def pickMaxVal(arr: Array[Double]) : Double = {
    val maxVal : Double = arr.reduceLeft(math.max);
    maxVal;
  }

    
    
  /////////////////////////////////////////////////////////
  // Check output consistency /////////////////////////////
  /////////////////////////////////////////////////////////
  def checkOutputConsistency(ilpInfo: ILPModelInfo) {
    
    
      // Add consistency constraints
      ////////////////////////////////
      // cn consistency
      for ((ijstr, cnlist) <- ilpInfo.cnijMap) {
        for (jj <- 0 until cnlist.size) {
          val combinedVar = cnlist.get(jj);
          val vname = combinedVar.get(GRB.StringAttr.VarName);
          //println(vname);
          val iIdx = parseNameInt(ijstr, 0);
          val jvalueIdx = parseNameInt(ijstr, 1);
          val ivIdx = parseNameInt(vname, 3);
          val jvIdx = parseNameInt(vname, 4);
          val cvar = getIlpVarValue(ilpInfo.corefVars, iIdx, jvalueIdx); // x1
          val jIdx = getIlpValue[Int](ilpInfo.corefValues, iIdx, jvalueIdx);
          val nivar = getIlpVarValue(ilpInfo.nerVars, iIdx, ivIdx); // x2
          val njvar = getIlpVarValue(ilpInfo.nerVars, jIdx, jvIdx); // x3
          //constructAddConstraint3Var(cvar, nivar, njvar, combinedVar, model, "Constr_CorefNer_" + ijstr + "@" + ivIdx + "@" + jvIdx);
          checkTeneryConsistency(combinedVar, cvar, nivar, njvar);
        }
      }

      // cw consistency
      for ((ijstr, cwlist) <- ilpInfo.cwijMap) {
        for (jj <- 0 until cwlist.size) {
          val combinedVar = cwlist.get(jj);
          val vname = combinedVar.get(GRB.StringAttr.VarName);
          val iIdx = parseNameInt(ijstr, 0);
          val jvalueIdx = parseNameInt(ijstr, 1);
          val ivIdx = parseNameInt(vname, 3);
          val jvIdx = parseNameInt(vname, 4);
          val cvar = getIlpVarValue(ilpInfo.corefVars, iIdx, jvalueIdx); // x1
          val jIdx = getIlpValue[Int](ilpInfo.corefValues, iIdx, jvalueIdx);
          val wivar = getIlpVarValue(ilpInfo.wikiVars, iIdx, ivIdx); // x2
          val wjvar = getIlpVarValue(ilpInfo.wikiVars, jIdx, jvIdx); // x3
          //constructAddConstraint3Var(cvar, wivar, wjvar, combinedVar, model, "Constr_CorefWiki_" + ijstr + "@" + ivIdx + "@" + jvIdx);
          checkTeneryConsistency(combinedVar, cvar, wivar, wjvar);
        }
      }

      // nw consistency
      for ((ii, nwlist) <- ilpInfo.nwiMap) {
        for (jj <- 0 until nwlist.size) {
          val combinedVar = nwlist.get(jj);
          val vname = combinedVar.get(GRB.StringAttr.VarName);
          val iIdx = parseNameInt(vname, 1);//ii;
          val ivIdx = parseNameInt(vname, 2);
          val jvIdx = parseNameInt(vname, 3);
          val nivar = getIlpVarValue(ilpInfo.nerVars, iIdx, ivIdx); // x1
          val wivar = getIlpVarValue(ilpInfo.wikiVars, iIdx, jvIdx); // x2
          //constructAddConstraint2Var(nivar, wivar, combinedVar, model, "Constr_NerWiki_" + iIdx + "@" + ivIdx + "@" + jvIdx);
          checkBinaryConsistency(combinedVar, nivar, wivar);
        }
      }
    
      println("All check consistency passed!");
  }
  
  def checkTeneryConsistency(combined: GRBVar, x1: GRBVar, x2: GRBVar, x3: GRBVar) {
	  val combVal = combined.get(GRB.DoubleAttr.X);
	  val x1Val = x1.get(GRB.DoubleAttr.X);
	  val x2Val = x2.get(GRB.DoubleAttr.X);
	  val x3Val = x3.get(GRB.DoubleAttr.X);
    
    val left = (combVal > 0);
    val right = ((x1Val > 0) && (x2Val > 0) && (x3Val > 0));
    if (left == right) {
      // pass check!
    } else {
      throw new RuntimeException("Inconsist: " + combined.get(GRB.StringAttr.VarName) + " " + combVal + " = " + x1Val + " & " + x2Val + " & " + x3Val);
    }
  }
  
  def checkBinaryConsistency(combined: GRBVar, x1: GRBVar, x2: GRBVar) {
    val combVal = combined.get(GRB.DoubleAttr.X);
    val x1Val = x1.get(GRB.DoubleAttr.X);
    val x2Val = x2.get(GRB.DoubleAttr.X);
    
    val left = (combVal > 0);
    val right = ((x1Val > 0) && (x2Val > 0));
    if (left == right) {
      // pass check!
    } else {
      throw new RuntimeException("Inconsist: " + combined.get(GRB.StringAttr.VarName) + " " + combVal + " = " + x1Val + " & " + x2Val);
    }
  }
  
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  //// About Learning 
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  
  
  // Due to pruning, and latent variables, there might be different values for some position i, that
  // gold and predict are different, but both of them are correct. We fixed gold with those predict 
  // values to avoid update on correct values...
  def constructGoldReference(ex: AceJointTaskExample, predBestOutput: Array[Int]) = {

	  //val goldOutput = initState.getSelfCopy();

	  val goldOutput = Array.fill[Int](predBestOutput.length)(-1); // no reference

	  // correct-variable not need to change
	  for (i <- 0 until predBestOutput.length) {
		  val vari = ex.getVariableGivenIndex(i);
		  if (vari.values(predBestOutput(i)).isCorrect) { // correct value
			  goldOutput(i) = predBestOutput(i);
		  }
	  }
	  // no-correct-value variable no need to change
	  for (i <- 0 until predBestOutput.length) {
		  val vari = ex.getVariableGivenIndex(i);
		  if (vari.getCorrectNonPruningValueIndices().length == 0) { // no correct value
			  goldOutput(i) = predBestOutput(i);
		  }
	  }
	  // set musk
    goldOutput;
  }
  

    // run learner
  def runLearningPerceptron(allTrains: ArrayBuffer[AceJointTaskExample], 
                             featIndexer: Indexer[String],
                             testExs: ArrayBuffer[AceJointTaskExample],
                             unaryPruner: StaticDomainPruner,
                             numIter: Int): Array[Double] = {

      var weight = Array.fill[Double](featIndexer.size)(0);
      var weightSum = Array.fill[Double](featIndexer.size)(0);
      var lastWeight = Array.fill[Double](featIndexer.size)(0);

      val Iteration = numIter;//10;
      val learnRate = 0.1;
      val lambda = 1e-8;

      var updateCnt = 0;
      var lastUpdtCnt = 0;

      for (iter <- 0 until Iteration) {
        lastUpdtCnt = updateCnt;
        Array.copy(weight, 0, lastWeight, 0, weight.length);

        println("Iteration " + iter);
        var exId = 0;
        for (example <- allTrains) {

          exId += 1;

          println("\ndocCnt " + exId  + ": " + example.docGraph.corefDoc.rawDoc.docID);
          //val initState = SearchBasedLearner.getInitStateWithUnaryScore(example, unaryPruner.weight);//.getRandomInitState(example);
          val noRef = Array.fill[Int](example.totalSize)(-1);
          val predBestOutput = runILPinference(example, weight, false, noRef, false);
          val goldRef = constructGoldReference(example, predBestOutput);
          val goldBestOutput = runILPinference(example, weight, true, goldRef, true);//hillClimbing(example, goldInit, weight, true).output;//example.infereceIndepGoldBest(weight);  // gold best
          
          // update?
          if (!example.isCorrectOutput(predBestOutput)) {
            updateCnt += 1;
            if (updateCnt % 1000 == 0) println("Update " + updateCnt);
            
            val featGold = example.featurize(goldBestOutput);
            val featPred = example.featurize(predBestOutput);
            
            updateWeight(weight, 
                         featGold,
                         featPred,
                         learnRate,
                         lambda);
            SingleTaskStructTesting.sumWeight(weightSum, weight);
          }
        }

        ///////////////////////////////////////////////////
        // have a test after each iteration (for learning curve)
        val tmpAvg = new Array[Double](weightSum.size)
        Array.copy(weightSum, 0, tmpAvg, 0, weightSum.size);
        SingleTaskStructTesting.divdeNumber(tmpAvg, updateCnt.toDouble);

        //greedySearchQuickTest(allTrains, tmpAvg, unaryPruner);
        ILPQuickTest(testExs, tmpAvg);
        println("Iter Update Cnt = " + (updateCnt - lastUpdtCnt));
      }

      SingleTaskStructTesting.divdeNumber(weightSum, updateCnt.toDouble);

      weightSum;
  }

  def updateWeight(currentWeight: Array[Double], 
                  featGold: HashMap[Int,Double],
                  featPred: HashMap[Int,Double],
                  eta: Double,
                  lambda: Double) {
    var gradient = Array.fill[Double](currentWeight.length)(0);//new Array[Double](currentWeight.length);
    for ((i, vgold) <- featGold) {
      gradient(i) += (vgold);
    }
    for ((j, vpred) <- featPred) {
       gradient(j) -= (vpred);
    }

    // do L2 Regularization
    //var l1norm = getL1Norm(currentWeight);
    for (i2 <- 0 until currentWeight.length) {
      //var regularizerNum: Double = Math.max(0, b);
      //var regularizerDen: Double = Math.max(0, b);
      var reg: Double = 1.0 - (eta * lambda)
          var curWeightVal = currentWeight(i2) * reg;
    currentWeight(i2) = curWeightVal + (gradient(i2) * eta);
    //currentWeight(i2) += (gradient(i2) * eta);
    }
  }
  
  def ILPQuickTest(testExs: ArrayBuffer[AceJointTaskExample], w: Array[Double]) {
    var sumTotal : Double = 0;
    var sumErr: Double = 0
    var sumErr1: Double = 0
    var sumErr2: Double = 0
    var sumErr3: Double = 0
    
    var cnt = 0;
    for (ex <- testExs) {
      cnt += 1;
      println("TestCnt:" + cnt + ", " + ex.docGraph.corefDoc.rawDoc.docID);
      val predBestOutput = runILPinference(ex, w, false, (Array.fill[Int](ex.totalSize)(-1)), false);
      val err = ex.getZeroOneError(predBestOutput);
      val (err1, err2, err3) = ex.getZeroOneErrorEachTask(predBestOutput, 0);
      val total = ex.totalSize;
      sumErr += err;
      sumTotal += total;
      
      sumErr1 += err1;
      sumErr2 += err2;
      sumErr3 += err3;
    }

    val eachSum = sumTotal / 3;
    val crct = sumTotal - sumErr;
    val acc = crct / sumTotal;
    println("Error each task = [" + sumErr1 + "," + sumErr2 + "," + sumErr3 +  "] / " + eachSum );
    println("quick test: 01-Acc = " + crct + "/" + sumTotal + " = " + acc);
  }
}
*/