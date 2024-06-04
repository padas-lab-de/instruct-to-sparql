## SPARQL query execution, expansion and optimization agents

The idea here is to implement a set of agents that can be used to execute, expand and optimize SPARQL queries. 
The agents will later on be used in a Chain of Thought (CoT), Thought/Action Chain (TAC) or Thought/Action Chain with Feedback (TACF) to execute a query, expand it and optimize it.

### Useful links:
- https://owlready2.readthedocs.io/en/latest/sparql.html#open-a-sparql-endpoint
- https://sparqlwrapper.readthedocs.io/en/latest/main.html#introduction
- https://github.com/eea/sparql-client
- https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service/query_optimization
- https://www.mediawiki.org/wiki/Wikibase/Indexing/RDF_Dump_Format#Prefixes_used
- https://www.mediawiki.org/wiki/Wikidata_Query_Service/User_Manual
- https://query.wikidata.org
- https://www.wikidata.org/wiki/Wikidata:Database_reports/List_of_properties/all
- https://github.com/jcklie/wikimapper#precomputed-indices


### Potential syntax endpoint: http://sparql.org/query-validator.html

### Prefixes for WIKIdata:
```sparql
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX cc: <http://creativecommons.org/ns#>
PREFIX dct: <http://purl.org/dc/terms/>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX hint: <http://www.bigdata.com/queryHints#> 
PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

PREFIX p: <http://www.wikidata.org/prop/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX pqn: <http://www.wikidata.org/prop/qualifier/value-normalized/>
PREFIX pqv: <http://www.wikidata.org/prop/qualifier/value/>
PREFIX pr: <http://www.wikidata.org/prop/reference/>
PREFIX prn: <http://www.wikidata.org/prop/reference/value-normalized/>
PREFIX prv: <http://www.wikidata.org/prop/reference/value/>
PREFIX psv: <http://www.wikidata.org/prop/statement/value/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX psn: <http://www.wikidata.org/prop/statement/value-normalized/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdata: <http://www.wikidata.org/wiki/Special:EntityData/>
PREFIX wdno: <http://www.wikidata.org/prop/novalue/>
PREFIX wdref: <http://www.wikidata.org/reference/>
PREFIX wds: <http://www.wikidata.org/entity/statement/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wdtn: <http://www.wikidata.org/prop/direct-normalized/>
PREFIX wdv: <http://www.wikidata.org/value/>
PREFIX wikibase: <http://wikiba.se/ontology#>
```