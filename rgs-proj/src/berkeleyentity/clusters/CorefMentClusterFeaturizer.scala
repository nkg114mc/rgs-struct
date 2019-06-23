package berkeleyentity.clusters

import edu.berkeley.nlp.futile.fig.basic.Indexer

class CorefMentionClusterFeaturizer(val featureIndexer: Indexer[String]) extends Serializable {
  
	//, "ChainSize1", "ChainSize2", "CombinedSize"
	//      ,"ChainSize1GT5", "ChainSize1LTE1", "ChainSize1LTE2", "ChainSize1LTE3", "ChainSize1LTE5"
	//, "ChainSize1LTE1", "ChainSize2LTE1"
	//      ,"ChainSize2GT5", "ChainSize2LTE1", "ChainSize2LTE2", "ChainSize2LTE3", "ChainSize2LTE5"
	//,"CombinedSizeGT20","CombinedSizeLTE2","CombinedSizeLTE3","CombinedSizeLTE4","CombinedSizeLTE5","CombinedSizeLTE10","CombinedSizeLTE20"
	//,"NormChainSize1","NormChainSize2"
	//      ,"ParSizeLTE1","ParSizeLTE2","ParSizeLTE3","ParSizeLTE5","ParSizeGT5"
	//      ,"ParSize2LTE1","ParSize2LTE2","ParSize2LTE3","ParSize2LTE5","ParSize2GT5"
  
  val clusterFeatures = Array(
      "SameGender", "CompatibleGender", "IncompatibleGender", 
      "SameNumber", "CompatibleNumber", "IncompatibleNumber",
      
			"IncompatiblePersonName",
			"SameAnimacy", "CompatibleAnimacy", "IncompatibleAnimacy", 
			"SameSemType", "CompatibleSemType",// "IncompatibleSemType",
			
			"HeadMatch", "ProperName", "PNStr", "PNSubstr", "PNIncomp", "ContainsPN","Constraints"

			,"PairTypeEE","PairTypeEL","PairTypeEP","PairTypeLL","PairTypeOE"
			,"PairTypeNE","PairTypeNL","PairTypeNP","PairTypeNO","PairTypeNN"
			,"PairTypeOL","PairTypeOO","PairTypeOP","PairTypePP","PairTypeLP"
			
			,"NormCombinedSize", "PossibleAnte"
			
			//,"ProResolveCl"
			,"ProResolveRuleR1","ProResolveRuleR2","ProResolveRuleR3","ProResolveRuleR5","ProResolveRuleR6","ProResolveRuleR7","ProResolveRuleR8"
			,"Demonyms", "CountryCapital", "WordSubstr", "HeadWordSubstr", "Modifier", "PostModifier"//, "Confidence","Confidence1","Confidence2"
	);
  
  
  def featurize() = {
    
  }
  
  
}