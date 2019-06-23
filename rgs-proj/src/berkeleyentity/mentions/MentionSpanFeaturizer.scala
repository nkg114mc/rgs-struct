package berkeleyentity.mentions
import scala.collection.mutable.ArrayBuffer
import edu.berkeley.nlp.futile.fig.basic.Indexer
import edu.berkeley.nlp.futile.util.Logger
import scala.collection.mutable.HashMap
import scala.collection.JavaConverters._
import edu.berkeley.nlp.futile.util.Counter
import berkeleyentity.sem.QueryCountsBundle
import berkeleyentity.ner.NerExample
import edu.berkeley.nlp.futile.syntax.Trees.PennTreeRenderer
import berkeleyentity.wiki.WikipediaInterface
import edu.mit.jwi.item.Pointer
import berkeleyentity.sem.FancyHeadMatcher
import berkeleyentity.sem.AbbreviationHandler
import berkeleyentity.sem.SemClass
import berkeleyentity.sem.SemClasser
import berkeleyentity.ner.NerFeaturizer
import berkeleyentity.WordNetInterfacer
import berkeleyentity.coref.FeatureSetSpecification
import berkeleyentity.coref.ConjFeatures
import berkeleyentity.coref.DocumentGraph
import berkeleyentity.coref.PronounDictionary
import berkeleyentity.coref.LexicalCountsBundle
import berkeleyentity.coref.Mention
import berkeleyentity.coref.PairwiseIndexingFeaturizer
import berkeleyentity.coref.ConjScheme
import berkeleyentity.coref.MentionType

/**
 * DO NOT try to add WordNetInterfacer here! It is not serializable and so
 * everything will explode when we try to serialize the model. So we choose
 * to cache it on the documents even though this is pretty hacky.
 */
@SerialVersionUID(1L)
class MentionSpanFeaturizer(val featureIndexer: Indexer[String],
                            val featureSet: FeatureSetSpecification,
                            val lexicalCounts: LexicalCountsBundle,
                            val queryCounts: Option[QueryCountsBundle],
                            val semClasser: Option[SemClasser]) {
  
  val cachedWni: WordNetInterfacer = null;
  
  import featureSet.featsToUse
  
  def getIndexer = featureIndexer;
  
  private def maybeAddFeat(indexedFeats: ArrayBuffer[Int], feat: String, addToIndexer: Boolean) {
    if (addToIndexer) {
      indexedFeats += featureIndexer.getIndex(feat)
    } else {
      val idx = featureIndexer.indexOf(feat)
      if (idx != -1) indexedFeats += idx;
    }
  }
  
  def getQueryCountsBundle = queryCounts;
  
  def featurizeIndex(currMent: Mention, addToFeaturizer: Boolean): Array[Int] = {
    featurizeIndexStandard(currMent, addToFeaturizer);
  }
  
  
  private def addFeatureAndConjunctions(feats: ArrayBuffer[Int],
                                        featName: String,
                                        currMent: Mention,
                                        addToFeaturizer: Boolean) {
    if (featureSet.conjScheme == ConjScheme.BOTH) {
      val currConjunction = "&C=" + currMent.computeConjStr(featureSet.conjFeatures, Some(cachedWni), semClasser);
      val prevConjunction =  "";
      maybeAddFeat(feats, featName + currConjunction + prevConjunction, addToFeaturizer);
    } else if (featureSet.conjScheme == ConjScheme.COARSE_CURRENT_BOTH) {
      // Note that this one behaves differently than the one above because BOTH is not always fired.
      // If it were always fired, it would often double with CURRENT.
      maybeAddFeat(feats, featName, addToFeaturizer);
      val currConjunction = "&C=" + currMent.computeConjStr(featureSet.conjFeatures, Some(cachedWni), semClasser);
      val featAndCurrConjunction = featName + currConjunction;
      maybeAddFeat(feats, featAndCurrConjunction, addToFeaturizer)
    } else { // All varieties of COARSE_BOTH
      // N.B. WHITELIST has the behavior that if no templates (type pairs) are specified,
      // then all templates (type pairs) are assumed to be whitelisted
      maybeAddFeat(feats, featName, addToFeaturizer);
      val validPair = false;
      val validTemplate = (featureSet.conjScheme == ConjScheme.COARSE_BOTH ||
                          (featureSet.conjScheme == ConjScheme.COARSE_BOTH_WHITELIST && (featureSet.conjListedTemplates.isEmpty || featureSet.conjListedTemplates.contains(PairwiseIndexingFeaturizer.getTemplate(featName)))) ||
                          (featureSet.conjScheme == ConjScheme.COARSE_BOTH_BLACKLIST && !featureSet.conjListedTemplates.contains(PairwiseIndexingFeaturizer.getTemplate(featName))));
      if (validPair && validTemplate) {
        val currConjunction = "&C=" + currMent.computeConjStr(featureSet.conjFeatures, Some(cachedWni), semClasser);
        val prevConjunction = "";
        maybeAddFeat(feats, featName + currConjunction + prevConjunction, addToFeaturizer);
      } else { // Back off to the EMNLP conjunctions
        val currConjunction = "&C=" + currMent.computeConjStr(ConjFeatures.TYPE_OR_CANONICAL_PRON, Some(cachedWni), semClasser);
        val prevConjunction =  "";
        maybeAddFeat(feats, featName + currConjunction + prevConjunction, addToFeaturizer);
      }
    }
  }
  
  //def featurizeIndexStandard(docGraph2: DocumentGraph, currMentIdx: Int, antecedentIdx: Int, addToFeaturizer: Boolean): Array[Int] = {
  def featurizeIndexStandard(currMent: Mention, addToFeaturizer: Boolean): Array[Int] = {
    val feats = new ArrayBuffer[Int]();
    def addFeatureShortcut = (featName: String) => {
      // Only used in CANONICAL_ONLY_PAIR, so only compute the truth value in this case
      addFeatureAndConjunctions(feats, featName, currMent, addToFeaturizer);
    }
    // FEATURES ON THE CURRENT MENTION (mostly targeting anaphoricity)
    val mentType = currMent.mentionType;
    // When using very minimal feature sets, you might need to include this so every decision
    // has at least one feature over it.
    val snStr = "";
    // N.B. INCLUDED IN SURFACE
    if (!featsToUse.contains("nomentlen")) {
//      addFeatureShortcut("SNMentLen=" + currMent.spanToString.split("\\s+").size + snStr);
      addFeatureShortcut("SNMentLen=" + currMent.words.size + snStr);
    }
    // N.B. INCLUDED IN SURFACE
    if (!featsToUse.contains("nolexanaph") && !currMent.mentionType.isClosedClass) {
      addFeatureShortcut("SNMentHead=" + fetchHeadWordOrPos(currMent) + snStr);
    }
    // N.B. INCLUDED IN SURFACE
    if (!featsToUse.contains("nolexfirstword") && !currMent.mentionType.isClosedClass) {
      addFeatureShortcut("SNMentFirst=" + fetchFirstWordOrPos(currMent) + snStr);
    }
    // N.B. INCLUDED IN SURFACE
    if (!featsToUse.contains("nolexlastword") && !currMent.mentionType.isClosedClass) {
      addFeatureShortcut("SNMentLast=" + fetchLastWordOrPos(currMent) + snStr);
    }
    // N.B. INCLUDED IN SURFACE
    if (!featsToUse.contains("nolexprecedingword")) {
      addFeatureShortcut("SNMentPreceding=" + fetchPrecedingWordOrPos(currMent) + snStr);
    }
    // N.B. INCLUDED IN SURFACE
    if (!featsToUse.contains("nolexfollowingword")) {
      addFeatureShortcut("SNMentFollowing=" + fetchFollowingWordOrPos(currMent) + snStr);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // ADD YOUR OWN FEATURES HERE!                                                                //
    //   See above for examples of how to do this. Typically use addFeatureShortcut since this    //
    // gives you your feature as well as conjunctions, but you can also directly call             //
    // feats += getIndex(feat, addToFeaturizer);                                                  //
    //                                                                                            //
    // To control feature sets, featsToUse is passed down from pairwiseFeats (the command line    //
    // argument). We currently use magic words all starting with +, but you do have to make       //
    // sure that you don't make a magic word that's a prefix of another, or else both will be     //
    // added when the longer one is.                                                              //
    //                                                                                            //
    // Happy feature engineering!                                                                 //
    ////////////////////////////////////////////////////////////////////////////////////////////////
    feats.toArray;
  }
  
  def fetchHeadWordOrPos(ment: Mention) = fetchWordOrPosDefault(ment.headStringLc, ment.pos(ment.headIdx - ment.startIdx), lexicalCounts.commonHeadWordCounts);
  def fetchFirstWordOrPos(ment: Mention) = fetchWordOrPosDefault(ment.words(0).toLowerCase, ment.pos(0), lexicalCounts.commonFirstWordCounts);
  
  def fetchLastWordOrPos(ment: Mention) = {
    if (ment.words.size == 1 || ment.endIdx - 1 == ment.headIdx) {
      ""
    } else {
      fetchWordOrPosDefault(ment.words(ment.words.size - 1).toLowerCase, ment.pos(ment.pos.size - 1), lexicalCounts.commonLastWordCounts);
    }
  }
  private def fetchPenultimateWordOrPos(ment: Mention) = {
    if (ment.words.size <= 2) {
      ""
    } else {
      fetchWordOrPosDefault(ment.words(ment.words.size - 2).toLowerCase, ment.pos(ment.pos.size - 2), lexicalCounts.commonPenultimateWordCounts);
    }
  }
  private def fetchSecondWordOrPos(ment: Mention) = {
    if (ment.words.size <= 3) {
      ""
    } else {
      fetchWordOrPosDefault(ment.words(1).toLowerCase, ment.pos(1), lexicalCounts.commonSecondWordCounts);
    }
  }
  
  def fetchPrecedingWordOrPos(ment: Mention) = fetchWordOrPosDefault(ment.contextWordOrPlaceholder(-1).toLowerCase, ment.contextPosOrPlaceholder(-1), lexicalCounts.commonPrecedingWordCounts);
  def fetchFollowingWordOrPos(ment: Mention) = fetchWordOrPosDefault(ment.contextWordOrPlaceholder(ment.words.size).toLowerCase, ment.contextPosOrPlaceholder(ment.words.size), lexicalCounts.commonFollowingWordCounts);
  private def fetchPrecedingBy2WordOrPos(ment: Mention) = fetchWordOrPosDefault(ment.contextWordOrPlaceholder(-2).toLowerCase, ment.contextPosOrPlaceholder(-2), lexicalCounts.commonPrecedingBy2WordCounts);
  private def fetchFollowingBy2WordOrPos(ment: Mention) = fetchWordOrPosDefault(ment.contextWordOrPlaceholder(ment.words.size + 1).toLowerCase, ment.contextPosOrPlaceholder(ment.words.size + 1), lexicalCounts.commonFollowingBy2WordCounts);
  private def fetchGovernorWordOrPos(ment: Mention) = fetchWordOrPosDefault(ment.governor.toLowerCase, ment.governorPos, lexicalCounts.commonGovernorWordCounts);
  
  
  private def fetchWordOrPosDefault(word: String, pos: String, counter: Counter[String]) = {
    if (counter.containsKey(word)) {
      word;
    } else if (featsToUse.contains("NOPOSBACKOFF")) {
      ""
    } else {
      pos;
    }
  }
  
  private def fetchPrefix(word: String) = {
    if (word.size >= 3 && lexicalCounts.commonPrefixCounts.containsKey(word.substring(0, 3))) {
      word.substring(0, 3);
    } else if (word.size >= 2 && lexicalCounts.commonPrefixCounts.containsKey(word.substring(0, 2))) {
      word.substring(0, 2);
    } else if (lexicalCounts.commonPrefixCounts.containsKey(word.substring(0, 1))) {
      word.substring(0, 1);
    } else {
      "";
    }
  }
  
  private def fetchSuffix(word: String) = {
    if (word.size >= 3 && lexicalCounts.commonSuffixCounts.containsKey(word.substring(word.size - 3))) {
      word.substring(word.size - 3);
    } else if (word.size >= 2 && lexicalCounts.commonSuffixCounts.containsKey(word.substring(word.size - 2))) {
      word.substring(word.size - 2);
    } else if (lexicalCounts.commonSuffixCounts.containsKey(word.substring(word.size - 1))) {
      word.substring(word.size - 1);
    } else {
      "";
    }
  }
  
  def fetchShape(word: String) = {
    if (lexicalCounts.commonShapeCounts.containsKey(NerFeaturizer.shapeFor(word))) {
      NerFeaturizer.shapeFor(word);
    } else {
      "";
    }
  }
  
  def fetchClass(word: String) = {
    if (lexicalCounts.commonClassCounts.containsKey(NerFeaturizer.classFor(word))) {
      NerFeaturizer.classFor(word);
    } else {
      "";
    }
  }
  
  private def fetchHeadWord(ment: Mention) = ment.words(ment.headIdx - ment.startIdx);
  private def fetchFirstWord(ment: Mention) = ment.words(0);
  private def fetchLastWord(ment: Mention) = ment.words(ment.pos.size - 1);
  private def fetchPrecedingWord(ment: Mention) = ment.contextWordOrPlaceholder(-1);
  private def fetchFollowingWord(ment: Mention) = ment.contextWordOrPlaceholder(ment.pos.size);
  private def fetchHeadPos(ment: Mention) = ment.pos(ment.headIdx - ment.startIdx);
  
  private def computeDefiniteness(ment: Mention) = {
    val firstWord = ment.words(0).toLowerCase;
    if (firstWord.equals("the")) {
      "DEF"
    } else if (firstWord.equals("a") || firstWord.equals("an")) {
      "INDEF"
    } else {
      "NONE"
    }
  }
  
  private def computePronNumber(ment: Mention) = {
    val firstWord = ment.words(0).toLowerCase;
    if (PronounDictionary.singularPronouns.contains(ment.headStringLc)) {
      "SING"
    } else if (PronounDictionary.pluralPronouns.contains(ment.headStringLc)) {
      "PLU"
    } else {
      "UNKNOWN"
    }
  }
  
  private def computePronGender(ment: Mention) = {
    val firstWord = ment.words(0).toLowerCase;
    if (PronounDictionary.malePronouns.contains(ment.headStringLc)) {
      "MALE"
    } else if (PronounDictionary.femalePronouns.contains(ment.headStringLc)) {
      "FEMALE"
    } else if (PronounDictionary.neutralPronouns.contains(ment.headStringLc)) {
      "NEUTRAL"
    } else {
      "UNKNOWN"
    }
  }
  
  private def computePronPerson(ment: Mention) = {
    val firstWord = ment.words(0).toLowerCase;
    if (PronounDictionary.firstPersonPronouns.contains(ment.headStringLc)) {
      "1st"
    } else if (PronounDictionary.secondPersonPronouns.contains(ment.headStringLc)) {
      "2nd"
    } else if (PronounDictionary.firstPersonPronouns.contains(ment.headStringLc)) {
      "3rd"
    } else {
      "OTHER"
    }
  }
  
  private def computeSentMentIdx(docGraph: DocumentGraph, ment: Mention) = {
    var currIdx = ment.mentIdx - 1;
    while (currIdx >= 0 && docGraph.getMention(currIdx).sentIdx == ment.sentIdx) {
      currIdx -= 1;
    }
    ment.mentIdx - currIdx + 1;
  }

  def computeTopicLabel(docGraph: DocumentGraph, clustererIdx: Int, mentIdx: Int): String = {
    val ment = docGraph.getMention(mentIdx);
    if (ment.mentionType == MentionType.PRONOMINAL && featsToUse.contains("noprons")) {
      "PRON"
    } else if ((ment.mentionType == MentionType.NOMINAL || ment.mentionType == MentionType.PROPER) && featsToUse.contains("nonomsprops")) {
      "NOMPROP"
    } else {
      docGraph.getBestCluster(clustererIdx, mentIdx) + ""
    }
  }
  
  def computeDistribLabel(docGraph: DocumentGraph, clustererIdx: Int, mentIdx: Int, valIdx: Int): Int = {
    docGraph.storedDistributedLabels(clustererIdx)(mentIdx)(valIdx);
  }
  
  def numDistribLabels(docGraph: DocumentGraph, clustererIdx: Int): Int = {
    docGraph.numClusters(clustererIdx);
  }
}
  
object MentionSpanFeaturizer {
  val UnkFeatName = "UNK_FEAT";
}
