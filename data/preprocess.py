import json
import re

import pywikibot
import requests
from cleantext import clean
from qwikidata.linked_data_interface import get_entity_dict_from_api
from rdflib.plugins.sparql.parser import parseQuery

from constants import wiki_prefixes, schema_dict

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


if __name__ == '__main__':
    # clean_queries()
    # get_query_template()
    # add_ids_to_queries()
    clean_comments()
