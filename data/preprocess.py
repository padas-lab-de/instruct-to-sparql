import json
import re

import pywikibot
import requests
from cleantext import clean
from qwikidata.linked_data_interface import get_entity_dict_from_api
from rdflib.plugins.sparql.parser import parseQuery

from constants import wiki_prefixes, schema_dict, PREFIX_TO_URL

from wikidata import WikidataAPI
import pandas as pd

wikidata = site = pywikibot.Site("wikidata", "wikidata")


def get_entity_dict_from_api_v2(id_):
    """This function takes a wikidata id and return the corresponding entity dict"""
    url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={id_}&format=json"
    r = requests.get(url)
    return r.json()["entities"][id_]


def get_entity_id_from_api_v2(label):
    """This function takes a wikidata label and return the corresponding entity id"""
    url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={label}&language=en&format=json"
    r = requests.get(url)
    return r.json()["search"][0]["id"]


def get_property_id_from_api_v2(label):
    """This function takes a wikidata label and return the corresponding property id"""
    url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={label}&language=en&format=json&type=property"
    r = requests.get(url)
    return r.json()["search"][0]["id"]


def get_patterns(wiki_prefixes):
    """This function takes a dictionary of prefixes and return a list of patterns to be used to replace them in the queries
    e.g: "wdt:P31" -> regex experession using to detect all occurences of "wdt:*" in a query. The regex should only match IDs in the form of a Letter follwed by numbers.
    """
    patterns = dict()
    for prefix in wiki_prefixes:
        patterns[prefix] = re.compile(rf"{prefix}:[A-Z]\d+", re.IGNORECASE)
    return patterns


def get_schema_patterns(schema_dict):
    """This function takes a dictionary of prefixes and return a list of patterns to be used to replace them in the queries
    e.g: "wikibase:directClaim" -> regex experession using to detect all occurences of "wikibase:*" in a query. The regex should only match prefix:entity_name patterns.
    """
    patterns = dict()
    for prefix in schema_dict:
        patterns[prefix] = re.compile(rf"{prefix}:\w+", re.IGNORECASE)
    return patterns


def clean_label(label):
    return re.sub(r"[^a-zA-Z0-9 ]", "", label)


def replace_ids(query, pattern, wiki_prefix):
    """This function takes a query and a pattern and replace all occurences of the pattern in the query with the corresponding wikidata id"""
    matches = re.findall(pattern, query)
    base_uri = wiki_prefixes[wiki_prefix]
    for match in matches:
        id_ = match.split(":")[1]
        id_ = re.sub(r"[^a-zA-Z0-9]", "", id_)
        entity = get_entity_dict_from_api(id_)
        label = clean_label(entity["labels"]["en"]["value"])
        # replace all special characters in the label with an underscore
        label = re.sub(r"[^a-zA-Z0-9]", "_", label)
        label = clean_label(entity["labels"]["en"]["value"]).replace(' ', '_')
        query = query.replace(match, f"{wiki_prefix}:{label}")
        query = f"# {wiki_prefix}:{label} is the text label corresponding to ID:{id_} (<{base_uri}{id_}>)\n" + query
    if matches:
        # prepend the query with the prefix
        query = "PREFIX " + wiki_prefix + ": <" + base_uri + "> \n" + query
    return query


def add_schema_prefix(query, pattern, schema_prefix):
    """This function simply prepends the query with the schema prefix if it's used in the query"""
    matches = re.findall(pattern, query)
    base_uri = schema_dict[schema_prefix]
    if matches:
        # prepend the query with the prefix
        query = "PREFIX " + schema_prefix + ": <" + base_uri + "> \n" + query
    return query

def clean_queries():
    """This function takes a json object with SPARQL queries with metadata and clean the queries to be usable by the SPARQL endpoint"""
    with open("raw/queries.json", "r") as f:
        queries = json.load(f)
    cleaned_queries = []
    embed_view_re = re.compile(r"^.+?(?=SELECT)", flags=re.IGNORECASE | re.DOTALL)
    for query in queries:
        q = query["query"]
        q = clean(q, fix_unicode=True, to_ascii=True, lower=False, no_line_breaks=False, normalize_whitespace=True,
                  no_urls=False, no_emails=False,
                  no_phone_numbers=False, no_numbers=False, no_digits=False, no_currency_symbols=False, no_punct=False,
                  replace_with_punct="", replace_with_url="", replace_with_email="", replace_with_phone_number="",
                  replace_with_number="", replace_with_digit="", replace_with_currency_symbol="", lang="en")
        # remove the embed view
        if not q.lower().startswith("select"):
            q = embed_view_re.sub("", q)
        query["query"] = q
        # clean the metadata
        metadata = query["metadata"][0]
        metadata["context"] = clean(metadata["context"], fix_unicode=True, to_ascii=True, lower=False,
                                    no_line_breaks=False, normalize_whitespace=True,
                                    no_urls=False, no_emails=False,
                                    no_phone_numbers=False, no_numbers=False, no_digits=False,
                                    no_currency_symbols=False, no_punct=False,
                                    replace_with_punct="", replace_with_url="", replace_with_email="",
                                    replace_with_phone_number="",
                                    replace_with_number="", replace_with_digit="", replace_with_currency_symbol="",
                                    lang="en")
        metadata["description"] = clean(metadata["description"], fix_unicode=True, to_ascii=True, lower=False,
                                        no_line_breaks=False, normalize_whitespace=True,
                                        no_urls=False, no_emails=False,
                                        no_phone_numbers=False, no_numbers=False, no_digits=False,
                                        no_currency_symbols=False, no_punct=False,
                                        replace_with_punct="", replace_with_url="", replace_with_email="",
                                        replace_with_phone_number="",
                                        replace_with_number="", replace_with_digit="", replace_with_currency_symbol="",
                                        lang="en")
        query["metadata"] = metadata
        cleaned_queries.append(query)

    with open("raw/cleaned_queries.json", "w") as f:
        json.dump(cleaned_queries, f, indent=2)


def get_query_template():
    """This function is to dissect the SPARQL queries into templates and variables or IDs for the COT instruction with Actions."""
    # TODO using the SPARQL parser to parse the queries into templates and variables
    # load the clean queries
    with open("raw/cleaned_queries.json", "r") as f:
        queries = json.load(f)
    prefix_patterns = get_patterns(wiki_prefixes)
    schedma_patters = get_schema_patterns(schema_dict)
    query_templates = []
    for query in queries:
        q = query["query"]
        # Replace IDS with variables
        try:
            for prefix, pattern in prefix_patterns.items():
                q = replace_ids(q, pattern, prefix)
            for prefix, pattern in schedma_patters.items():
                q = add_schema_prefix(q, pattern, prefix)
            query["query"] = q
        except Exception as e:
            print(e)
            continue
        # parse the query
        try:
            parsed_query = parseQuery(q)
            query["parsed_query"] = parsed_query.dump()
        except Exception as e:
            query["parsed_query"] = f"Query syntax parsing failed with the following error:{e}"
            print(e)
        query_templates.append(query)
    print("Number of queries:", len(query_templates))
    with open("raw/query_templates_v0.json", "w") as f:
        json.dump(query_templates, f, indent=2)


def add_ids_to_queries():
    """This function is to add IDs to the queries to be used in the COT instruction with Actions."""
    # load the query templates
    with open("raw/query_templates.json", "r") as f:
        query_templates = json.load(f)
    # add the ids to the queries in both files
    invalid_count = 0
    for idx, query in enumerate(query_templates):
        query["id"] = idx
        # add a flag to indicate if the query is a valid SPARQL query
        if query["parsed_query"].startswith("Query syntax parsing failed"):
            query["valid"] = False
            invalid_count += 1
        else:
            query["valid"] = True
    print("Number of invalid queries:", invalid_count)
    with open("raw/query_templates_v1.json", "w") as f:
        json.dump(query_templates, f, indent=2)


def clean_comments():
    # load the dataset
    with open("processed/sparql_instruct.json", "r") as f:
        query_instruct = json.load(f)
    # clean the comments
    queries = []
    for query in query_instruct:
        q = query["query"]
        # remove the comments that are at the start of every line with their line breaks
        # split the query on SELECT and remove the comments from the first part and then join the query back
        q = re.split(r"SELECT", q, maxsplit=1, flags=re.IGNORECASE)
        q[0] = re.sub(r"^\s*#.*\n", "", q[0], flags=re.MULTILINE)
        q = "SELECT".join(q)
        query["query"] = q
        queries.append(query)

    with open("processed/sparql_instruct_v1.json", "w") as f:
        json.dump(queries, f, indent=2)

def remove_styling_markup_and_urls():
    with open("raw/cleaned_queries.json", "r") as f:
        queries = json.load(f)
    
    # x is str
    remove_markup = lambda x: re.sub("<\/?[a-zA-z?;=\"' ]*>|<.+[\W]>", ' ', x, flags=0)
    remove_css = lambda x: re.sub("(?:class|colspan|style|rowspan)=\".+\"", " ", x, flags=0)
    remove_url = lambda x: re.sub("https:[a-zA-Z%:./#_0-9]+", " ", x, flags=0)
    post_remove = lambda x: x.strip().replace("\n\n", "\n")
    
    apply_preprocess = lambda x: {
        "query": x["query"],
        "metadata": {
            "description": post_remove(remove_url(x["metadata"]["description"])),
            "context": post_remove(remove_markup(remove_css(x["metadata"]["context"]))),
        }
    }
    
    new_queries = [apply_preprocess(x) for x in queries]
    
    with open("processed/cleaner_queries.json", "w") as f:
        json.dump(new_queries, f, indent=2)

def annotate_queries():
    def extract_entities_properties_ids(query:str):
        pattern = re.compile(r":(Q\d+|P\d+|L\d+)")
        results = pattern.findall(query)

        if results:
            return results
        else:
            return []

    def replace_entities_and_properties_id_with_labels(query: str):
        extracted_properties_and_entities = set(extract_entities_properties_ids(query))
        
        api = WikidataAPI()
        
        entities_id_w_labels = [api._smart_get_labels_from_entity_id(entity_id)[0] for entity_id in filter(lambda x: x.startswith("Q"), extracted_properties_and_entities)]
        properties_id_w_labels = [api._smart_get_labels_from_entity_id(property_id)[0] for property_id in filter(lambda x: x.startswith("P"), extracted_properties_and_entities)]
        lexemes_id_w_labels = [api._smart_get_labels_from_entity_id(property_id)[0] for property_id in filter(lambda x: x.startswith("L"), extracted_properties_and_entities)]
        
        new_query = query
        for e, label in entities_id_w_labels:
            new_query = re.sub(f":{e}", f":[entity:{label}]", new_query)
        for p, label in properties_id_w_labels:
            new_query = re.sub(f":{p}", f":[property:{label}]", new_query)
        for l, label in lexemes_id_w_labels:
            new_query = re.sub(f":{l}", f":[lexeme:{label}]", new_query)
        
        return new_query
    
    def add_relevant_prefixes_to_query(query: str) -> str:
        prefixes = ""
        copy_query = query
        for k in PREFIX_TO_URL.keys():
            current_prefix = f"PREFIX {k}: <{PREFIX_TO_URL[k]}>"
            
            # Some queries already have some prefixes, duplicating them will cause an error
            # So first we check that the prefix we want to add is not already included.
            if not re.search(current_prefix, copy_query): 
                
                # Then we look for the prefix in the query
                if re.search(rf"\W({k}):", copy_query):
                    prefixes += current_prefix + "\n"
            
            # For safety, we remove all the constants that starts with the prefix
            while re.search(rf"\W({k}):", copy_query):
                copy_query = re.sub(rf"\W({k}):", " ", copy_query)
        
        if prefixes != "":
            prefixes += "\n"
        
        return prefixes + query
    
    # TODO: change to fq18
    queries = pd.read_json("processed/final_fq17.json")
    
    # This can take several hours.
    queries["query_templated"] = queries.apply(lambda x: replace_entities_and_properties_id_with_labels(add_relevant_prefixes_to_query(x["query"])), axis=1)
    
    queries.to_json("processed/final_fq17-annotated_new.json")
    

if __name__ == '__main__':
    # clean_queries()
    # get_query_template()
    # add_ids_to_queries()
    clean_comments()
