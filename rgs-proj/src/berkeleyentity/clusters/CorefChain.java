package berkeleyentity.clusters;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

/*
import berkeleyentity.clusters.clustfeats.Animacy;
import berkeleyentity.clusters.clustfeats.Conjunction;
import berkeleyentity.clusters.clustfeats.EmptyProperty;
import berkeleyentity.clusters.clustfeats.FeatureUtils;
import berkeleyentity.clusters.clustfeats.FeatureUtils.AnimacyEnum;
import berkeleyentity.clusters.clustfeats.FeatureUtils.GenderEnum;
import berkeleyentity.clusters.clustfeats.FeatureUtils.NPSemTypeEnum;
import berkeleyentity.clusters.clustfeats.FeatureUtils.NumberEnum;
import berkeleyentity.clusters.clustfeats.HeadNoun;
import berkeleyentity.clusters.clustfeats.InfWords;
import berkeleyentity.clusters.clustfeats.NPSemanticType;
import berkeleyentity.clusters.clustfeats.PostModifier;
import berkeleyentity.clusters.clustfeats.ProperNameType;
import berkeleyentity.clusters.clustfeats.Property;
*/
import berkeleyentity.coref.DocumentGraph;
import berkeleyentity.coref.Gender;
import berkeleyentity.coref.Mention;



public class CorefChain {
/*
	//Specify default confidence values for different CE types
	//the indicies for the types are 0 = unknown; 1 = proper name; 2 = pronoun; 3=definite; 4=indefinite
	private static final int[] GEN_CONF = {0,3,4,1,2};
	private static final int[] ANI_CONF = {0,3,4,1,2};
	private static final int[] NUM_CONF = {0,3,4,1,2};
	private static final int[] NP_SEM_TYPE_CONF = {0,3,4,1,2};

	Integer id;
	private ArrayList<Mention> ces;
	private HashMap<Property, Object> properties;
	private HashMap<Property, Integer> confidences;

	private HashMap<Mention,HashMap<String,HashSet<Integer>>> proAntecedents = new HashMap<Mention, HashMap<String,HashSet<Integer>>>();

	private Mention firstCE;
	private Mention representitiveCE; // sometimes it is similar to firstCE
	private boolean pronoun=false;
	private boolean conjunction = false;

	private Integer redirect = -1;
	public Integer getRedirect() {
		return redirect;
	}
	public void setRedirect(Integer redirect) {
		this.redirect = redirect;
	}
	public boolean isRedirect(){
		return redirect>=0;
	}
	private CorefChain() {
		ces = new ArrayList<Mention>();
		properties = new HashMap<Property, Object>();
	}
	public CorefChain(int redir){
		//A mere redirecting placeholder coref chain
		redirect = redir;
	}
	public CorefChain(Integer id, Mention ce, DocumentGraph doc) {
		initialize(id,ce,doc);
	}
	public CorefChain(CorefChain c, DocumentGraph doc) {
		if(c.isRedirect())
			redirect=c.getRedirect();
		else{
			initialize(c.id,c.getFirstCe(),doc);
			proAntecedents = new HashMap<Mention, HashMap<String,HashSet<Integer>>>();
			for(Mention a:c.proAntecedents.keySet()){
				HashMap<String,HashSet<Integer>> ant = new HashMap<String, HashSet<Integer>>();
				HashMap<String,HashSet<Integer>> ant1 = c.proAntecedents.get(a);
				for(String s:ant1.keySet()){
					HashSet<Integer> copy = new HashSet<Integer>(ant1.get(s));
					ant.put(s, copy);
				}
				proAntecedents.put(a, ant);
			}
			ces=new ArrayList<Mention>();
			ces.addAll(c.ces);
			//possibleAntes = new ArrayList<Mention>(c.possibleAntes);
			//possibleAntesChain = new ArrayList<CorefChain>(c.possibleAntesChain);
		}
	}
	public void initialize(Integer id, Mention ce, DocumentGraph doc){
		redirect = -1;

		this.id = id;
		firstCE = ce;
		ces = new ArrayList<Mention>();
		ces.add(ce);
		properties = new HashMap<Property, Object>();
		confidences = new HashMap<Property, Integer>();
		int npType = PairType.getTypeNumber(ce, doc); // 1 = proper name; 2 = pronoun; 3=definite; 4=indefinite
		if(npType==2){
			proAntecedents=new HashMap<Mention, HashMap<String,HashSet<Integer>>>();
			pronoun = true;
		}
		if(Conjunction.getValue(ce, doc))
			conjunction=true;
		properties.put(Gender.getInstance(), Gender.getValue(ce, doc));
		confidences.put(Gender.getInstance(), GEN_CONF[npType]);
		properties.put(Animacy.getInstance(), Animacy.getValue(ce, doc));	
		confidences.put(Animacy.getInstance(), ANI_CONF[npType]);
		properties.put(NPSemanticType.getInstance(), NPSemanticType.getValue(ce, doc));
		confidences.put(NPSemanticType.getInstance(), NP_SEM_TYPE_CONF[npType]);
		properties.put(Number.getInstance(), Number.getValue(ce, doc));
		confidences.put(Number.getInstance(), NUM_CONF[npType]);
		NPSemTypeEnum pnType = ProperNameType.getValue(ce, doc);
		if(pnType!=null){
			if(pnType.equals(NPSemTypeEnum.PERSON)||pnType.equals(NPSemTypeEnum.ORGANIZATION)||pnType.equals(NPSemTypeEnum.LOCATION)){
				Mention ne = (Mention)Property.LINKED_PROPER_NAME.getValueProp(ce, doc);
				String[] names = Document.getWordsPN(doc.getAnnotText(ne));
				String name = doc.getAnnotText(ne);
				HashSet<String> nameSet = new HashSet<String>();
				nameSet.add(name);
				properties.put(NAMES, nameSet); 
				properties.put(ALL_NAMES,nameSet);
				String lastName = names[names.length-1];
				String firstName = names.length<2?null:(FeatureUtils.memberArray(names[0], FeatureUtils.PERSON_PREFIXES)?(names.length>2?names[1]:null):names[0]);
				HashSet<String> lastNames = new HashSet<String>();
				lastNames.add(lastName);
				properties.put(LAST_NAMES, lastNames);
				if(firstName!=null){
					HashSet<String> firstNames = new HashSet<String>();
					firstNames.add(firstName);
					properties.put(FIRST_NAMES, firstNames);
				}
			}
		}
		if(npType!=2){// && npType!=1){
			Mention headAn = HeadNoun.getValue(ce,doc);
			String head = doc.getAnnotText(headAn);
			HashSet<String> heads = new HashSet<String>();
			heads.add(head);
			properties.put(HEADS, heads); 
		}
		if(npType!=2){
			String[] infWords = (String[])InfWords.getInstance().getValueProp(ce, doc);
			HashSet<String> words = new HashSet<String>();
			for(String w:infWords)
				words.add(w);
			properties.put(WORDS, words);
		}
		String[] mod = (String[])Modifier.getInstance().getValueProp(ce, doc);
		HashSet<String> mods = new HashSet<String>();
		for(String w:mod)
			mods.add(w);
		properties.put(MODIFIERS, mods);
		HashSet<String> postmods = new HashSet<String>();
		mod = (String[])PostModifier.getInstance().getValueProp(ce, doc);
		for(String w:mod)
			postmods.add(w);
		properties.put(POST_MODIFIERS, postmods);
	}

	public ArrayList<Mention> getCes() {
		return ces;
	}
	public Mention getFirstCe(){
		return firstCE;
	}
	public void setRepresentCe(Mention representCE) {
		representitiveCE = representCE;
	}
	public Mention getRepresentCe(){
		return representitiveCE;
	}
	

	public Integer getId() {
		return id;
	}
	public void setId(Integer id) {
		this.id = id;
	}
	
	
	
	public Object getProperty(Property p){
		return properties.get(p);
	}

	public void setProperty(Property p, Object o){
		properties.put(p,o);
	}

	public void setCes(ArrayList<Mention> ces) {
		this.ces = ces;
	}

	public HashMap<Property, Object> getProperties() {
		return properties;
	}

	public void setProperties(HashMap<Property, Object> properties) {
		this.properties = properties;
	}
	public CorefChain join(CorefChain c){
		ces.addAll(c.ces);
		String clusterId = ces.get(0).getAttribute(Constants.CLUSTER_ID);
		for(Mention ce2: c.ces){
			ce2.setAttribute(Constants.CLUSTER_ID, clusterId);
		}     

		consolidateProperties(c);
		pronoun = firstCE.compareSpan(c.firstCE)<0?pronoun:c.pronoun;
		firstCE = firstCE.compareSpan(c.firstCE)<0?firstCE:c.firstCE;
		conjunction = conjunction||c.conjunction;
		return this;
	}

	public boolean before(CorefChain c){
		return firstCE.compareSpan(c.firstCE)<0;
	}

	public int compareTo(CorefChain c){
		return firstCE.compareSpan(c.firstCE);
	}


	private void consolidateProperties(CorefChain c){
		HashMap<Property,Object> props = c.getProperties();
		GenderEnum gender = GenderEnum.consolidate((GenderEnum)properties.get(Gender.getInstance()),(GenderEnum)props.get(Gender.getInstance()));
		if(gender.equals(GenderEnum.AMBIGUOUS)){
			if(confidences.get(Gender.getInstance())>c.confidences.get(Gender.getInstance())){
				gender = (GenderEnum)properties.get(Gender.getInstance());
			}else if(confidences.get(Gender.getInstance())<c.confidences.get(Gender.getInstance())){
				gender = (GenderEnum)props.get(Gender.getInstance());
			}else if(ces.size()>=c.ces.size())
				gender = (GenderEnum)properties.get(Gender.getInstance());
			else
				gender = (GenderEnum)props.get(Gender.getInstance());
		}
		properties.put(Gender.getInstance(), gender);
		confidences.put(Gender.getInstance(), max(confidences.get(Gender.getInstance()),c.confidences.get(Gender.getInstance())));
		NumberEnum number = NumberEnum.consolidate((NumberEnum)properties.get(Number.getInstance()),(NumberEnum)props.get(Number.getInstance()));
		if(number.equals(NumberEnum.AMBIGUOUS)){
			NPSemTypeEnum type1 = (NPSemTypeEnum) properties.get(NPSemanticType.getInstance());
			NPSemTypeEnum type2 = (NPSemTypeEnum) props.get(NPSemanticType.getInstance());
			if((type1.equals(NPSemTypeEnum.ORGANIZATION)&&type2.equals(NPSemTypeEnum.PERSON))||
					(type2.equals(NPSemTypeEnum.ORGANIZATION)&&type1.equals(NPSemTypeEnum.PERSON)) ){
				number = NumberEnum.PLURAL;
			}else if(confidences.get(Number.getInstance())>c.confidences.get(Number.getInstance())){
				number = (NumberEnum)properties.get(Number.getInstance());
			}else if(confidences.get(Number.getInstance())<c.confidences.get(Number.getInstance())){
				number = (NumberEnum)props.get(Number.getInstance());
			}else if(ces.size()>=c.ces.size())
				number = (NumberEnum)properties.get(Number.getInstance());
			else
				number = (NumberEnum)props.get(Number.getInstance());
		}
		properties.put(Number.getInstance(), number);
		confidences.put(Number.getInstance(), max(confidences.get(Number.getInstance()),c.confidences.get(Number.getInstance())));
		AnimacyEnum animacy = AnimacyEnum.consolidate((AnimacyEnum)properties.get(Animacy.getInstance()),(AnimacyEnum)props.get(Animacy.getInstance()));
		if(animacy.equals(AnimacyEnum.AMBIGUOUS)){
			if(confidences.get(Animacy.getInstance())>c.confidences.get(Animacy.getInstance())){
				animacy = (AnimacyEnum)properties.get(Animacy.getInstance());
			}else if(confidences.get(Animacy.getInstance())<c.confidences.get(Animacy.getInstance())){
				animacy = (AnimacyEnum)props.get(Animacy.getInstance());
			}else if(ces.size()>=c.ces.size())
				animacy = (AnimacyEnum)properties.get(Animacy.getInstance());
			else
				animacy = (AnimacyEnum)props.get(Animacy.getInstance());
		}
		properties.put(Animacy.getInstance(), animacy);
		confidences.put(Animacy.getInstance(), max(confidences.get(Animacy.getInstance()),c.confidences.get(Animacy.getInstance())));
		NPSemTypeEnum type = NPSemTypeEnum.consolidate((NPSemTypeEnum)properties.get(NPSemanticType.getInstance()),(NPSemTypeEnum)props.get(NPSemanticType.getInstance()));
		if(type.equals(NPSemTypeEnum.AMBIGUOUS)){
			if(confidences.get(NPSemanticType.getInstance())>c.confidences.get(NPSemanticType.getInstance())){
				type = (NPSemTypeEnum)properties.get(NPSemanticType.getInstance());
			}else if(confidences.get(NPSemanticType.getInstance())<c.confidences.get(NPSemanticType.getInstance())){
				type = (NPSemTypeEnum)props.get(NPSemanticType.getInstance());
			}else if(ces.size()>=c.ces.size())
				type = (NPSemTypeEnum)properties.get(NPSemanticType.getInstance());
			else
				type = (NPSemTypeEnum)props.get(NPSemanticType.getInstance());
		}
		properties.put(NPSemanticType.getInstance(), type);
		confidences.put(NPSemanticType.getInstance(), max(confidences.get(NPSemanticType.getInstance()),c.confidences.get(NPSemanticType.getInstance())));
		properties.put(NAMES, joinSets((HashSet<String>)properties.get(NAMES),(HashSet<String>)props.get(NAMES)));
		properties.put(ALL_NAMES, joinSets((HashSet<String>)properties.get(ALL_NAMES),(HashSet<String>)props.get(ALL_NAMES)));
		properties.put(LAST_NAMES, joinSets((HashSet<String>)properties.get(LAST_NAMES),(HashSet<String>)props.get(LAST_NAMES)));
		properties.put(FIRST_NAMES, joinSets((HashSet<String>)properties.get(FIRST_NAMES),(HashSet<String>)props.get(FIRST_NAMES)));
		properties.put(HEADS, joinSets((HashSet<String>)properties.get(HEADS),(HashSet<String>)props.get(HEADS)));
		properties.put(WORDS, joinSets((HashSet<String>)properties.get(WORDS),(HashSet<String>)props.get(WORDS)));
		properties.put(MODIFIERS, joinSets((HashSet<String>)properties.get(MODIFIERS),(HashSet<String>)props.get(MODIFIERS)));
		properties.put(POST_MODIFIERS, joinSets((HashSet<String>)properties.get(POST_MODIFIERS),(HashSet<String>)props.get(POST_MODIFIERS)));
	}

	public Integer min(Integer i1, Integer i2){
		return i1<i2?i1:i2;
	}
	public Integer max(Integer i1, Integer i2){
		return i1>i2?i1:i2;
	}



	public static final Property FIRST_NAMES = new EmptyProperty("FirstNames");
	public static final Property LAST_NAMES = new EmptyProperty("LastNames");
	public static final Property NAMES = new EmptyProperty("Names");
	public static final Property ALL_NAMES = new EmptyProperty("AllNames");
	public static final Property HEADS = new EmptyProperty("Heads");
	public static final Property WORDS = new EmptyProperty("Words");
	public static final Property MODIFIERS = new EmptyProperty("Modifiers");
	public static final Property POST_MODIFIERS = new EmptyProperty("PostModifiers");

	public static <T>  Set<T> joinSets(Set<T> l1,HashSet<T> l2){
		Set<T> result = l1;
		if(result==null){
			result = l2;
		}else if(l2!=null){
			result.addAll(l2);
		}
		return result;
	}
	public void setProAntecedents(HashMap<Mention,HashMap<String,HashSet<Integer>>> proAntecedents) {
		this.proAntecedents = proAntecedents;
	}
	public HashMap<String,HashSet<Integer>> getProAntecedents(Mention a) {
		HashMap<String,HashSet<Integer>> result = proAntecedents.get(a);
		if(result==null){
			result = new HashMap<String, HashSet<Integer>>();
			proAntecedents.put(a, result);
		}
		return result;
	}
	public HashMap<Mention,HashMap<String,HashSet<Integer>>> getAllProAntecedents() {
		return proAntecedents;
	}

	public boolean isGenderIncompatible(CorefChain c2){
		FeatureUtils.GenderEnum gen1 = (GenderEnum) getProperty(Gender.getInstance());
		FeatureUtils.GenderEnum gen2 = (GenderEnum) c2.getProperty(Gender.getInstance());
		if (gen1.equals(FeatureUtils.GenderEnum.MASC) && gen2.equals(FeatureUtils.GenderEnum.FEMININE)) 
			return true;
		if (gen2.equals(FeatureUtils.GenderEnum.MASC) && gen1.equals(FeatureUtils.GenderEnum.FEMININE)) 
			return true;

		return false;
	}
	public boolean isNumberIncompatible(CorefChain c2){
		NumberEnum num1 = (NumberEnum) getProperty(Number.getInstance());
		NumberEnum num2 = (NumberEnum) c2.getProperty(Number.getInstance());
		if (num1.equals(NumberEnum.PLURAL) && num2.equals(NumberEnum.SINGLE)) return true;
		if (num2.equals(NumberEnum.PLURAL) && num1.equals(NumberEnum.SINGLE)) return true;
		return false;
	}
	public boolean isAnimacyIncompatible(CorefChain c2){
		AnimacyEnum an1 = (AnimacyEnum) getProperty(Animacy.getInstance());
		AnimacyEnum an2 = (AnimacyEnum) c2.getProperty(Animacy.getInstance());
		if (an1.equals(AnimacyEnum.ANIMATE) && an2.equals(AnimacyEnum.UNANIMATE)) return true;
		if (an2.equals(AnimacyEnum.ANIMATE) && an1.equals(AnimacyEnum.UNANIMATE)) return true;
		return false;
	}
	public boolean isSemTypeIncompatible(CorefChain c2){
		NPSemTypeEnum type1 = (NPSemTypeEnum) getProperty(NPSemanticType.getInstance());
		NPSemTypeEnum type2 = (NPSemTypeEnum) c2.getProperty(NPSemanticType.getInstance());
		if (type1.equals(NPSemTypeEnum.UNKNOWN) || type1.equals(NPSemTypeEnum.AMBIGUOUS)||type2.equals(NPSemTypeEnum.UNKNOWN) || type2.equals(NPSemTypeEnum.AMBIGUOUS)) return false;
		if ((type1.equals(NPSemTypeEnum.NOTPERSON) && type2.equals(NPSemTypeEnum.PERSON))||(type2.equals(NPSemTypeEnum.NOTPERSON) && type1.equals(NPSemTypeEnum.PERSON))) return true;
		if (type1.equals(NPSemTypeEnum.NOTPERSON)||type2.equals(NPSemTypeEnum.NOTPERSON)) return false;
		if (type1.equals(type2)) return false;
		return true;
	}
	public NPSemTypeEnum getSemType(){
		return (NPSemTypeEnum)getProperty(NPSemanticType.getInstance());
	}

	public boolean isPronoun(){
		return pronoun;
	}
	public boolean isConjunction(){
		return conjunction;
	}

	public HashMap<Property, Integer>getPropConfidence()
	{
		return confidences;
	}
*/	

}


//public TwoDMap<Integer, Action> getSimilarities() {
//	return similarities;
//}
//public void setSimilarities(TwoDMap<Integer, Action> similarities) {
//	this.similarities = similarities;
//}

/*
public void addPossibleAntes(Mention ant, CorefChain ch){
	possibleAntes.add(ant);
	possibleAntesChain.add(ch);
}
public ArrayList<Mention> getPossibleAntes(){
	return possibleAntes;
}
public ArrayList<CorefChain> getPossibleAntesChain(){
	return possibleAntesChain;
}*/