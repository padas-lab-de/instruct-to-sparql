

from data.utils import parse_sparql_templates, parse_sql_links, extract_queries_from_parse
import pywikibot
from pywikibot import pagegenerators
import json




def download_and_prepare(page_titles, page_titles_links):
    #
    # 1. Create a generator to iterate over the pages
    site = pywikibot.Site("wikidata", "wikidata")
    pages = (pywikibot.Page(site, title) for title in page_titles)
    generator = pagegenerators.PreloadingGenerator(pages)

    results = {}
    queries = []
    # 2. Iterate over the pages and extract SPARQL queries
    for page in generator:
        text = page.text
        results[page.title()] = parse_sparql_templates(text, verbose=False)
        queries.extend(extract_queries_from_parse(results[page.title()]))
        print(f"Found {results[page.title()]['count']} queries in {page.title()}")

    pages = (pywikibot.Page(site, title) for title in page_titles_links)
    pagelist = pagegenerators.PreloadingGenerator(pages)
    additional_queries = parse_sql_links(pagelist, verbose=False)
    print(f"Found {len(additional_queries)} queries in link pages")
    # 3. Iterate over the pages and extract SPARQL queries
    queries.extend(additional_queries)
    return queries



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    page_titles=["User:MartinPoulter/queries", "User:MartinPoulter/queries/botany",
                 "User:MartinPoulter/queries/collections","User:MartinPoulter/queries/ee",
                 "User:MartinPoulter/queries/hillforts", "User:MartinPoulter/sappho",
                 "User:D1gggg/Wikidata_model_and_SPARQL", "User:Pigsonthewing/Queries",
                "Wikidata:SPARQL_query_service/queries", "Wikidata:SPARQL_query_service/queries/examples"
                 ,"Wikidata:SPARQL_tutorial/en"]
    page_titles_links = ["Wikidata:Weekly_query_examples", "Wikidata:Weekly_query_examples/2016",
                         "Wikidata:Weekly_query_examples/2018", "Wikidata:Weekly_query_examples/2017",
                         "Wikidata:Weekly_query_examples/2019", "Wikidata:Weekly_query_examples/2020",
                         "Wikidata:Weekly_query_examples/2021", "Wikidata:Weekly_query_examples/2022",
                            "Wikidata:Weekly_query_examples/2023"]

    queries=download_and_prepare(page_titles, page_titles_links)
    with open("raw/queries.json", "w") as f:
        json.dump(queries,f ,indent=2)

