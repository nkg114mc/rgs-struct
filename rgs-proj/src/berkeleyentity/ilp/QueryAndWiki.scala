package berkeleyentity.ilp

import berkeleyentity.wiki.Query;

/**
 * a combined value with query and wiki title
 */
class QueryAndWiki(val query: Query, val wiki: String) {
  def getQuery() = {
    query;
  }
  def getWiki() = {
    wiki;
  }
  
  override def toString(): String = {
    wiki;
  }
}