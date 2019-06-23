package berkeleyentity.coref;


public enum MentionType {

  PROPER(false), NOMINAL(false), PRONOMINAL(true), DEMONSTRATIVE(true);//, UNKNOWN(false);
  
  private boolean isClosedClass;
  
  private MentionType(boolean isClosedClass) {
    this.isClosedClass = isClosedClass;
  }
  
  public boolean isClosedClass() {
    return isClosedClass;
  }
}
