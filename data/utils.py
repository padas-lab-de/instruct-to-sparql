
import re
import requests
from urllib.parse import unquote
import json

def parse_sparql_wdtmp(text, block):

    regex = r"\{\{SPARQL\d*\|query=(.*?)\}\}"
    block["queries"] = []
    if text.find("SPARQL")>-1 and len(re.findall(regex, text, re.DOTALL))==0:
        print("WARNING: SPARQL in text, but no matching regex found", text)
    excludes = []
    for match in re.finditer(regex, text, re.DOTALL):
        query_and_pairs = match.groups()[0].strip().split("\n")
        query = ""
        pairs = []

        for line in query_and_pairs:
            if line.startswith("#"):
                # Parse key-value pair
                splits = line[1:].split(":", 1)
                if len(splits)<2:
                    key, value = "comment", splits[0]
                else:
                    key, value = splits
                pairs.append((key.strip(), value.strip()))
            else:
                # Append line to query
                query += line+"\n"
        block["queries"].append({"query": query, "metadata": pairs})
        excludes.append(match.span())
    stripped_txt=""
    last_pos=0
    for e in excludes:
        stripped_txt += text[last_pos:e[0]]
        last_pos = e[1]
    return stripped_txt

def parse_block(lines, parent, parent_level, pos=0):
    parent["childs"] =[]
    ix = pos
    block = parent
    rline = r"^ *==+(.*)==+ *\n*"
    while ix < len(lines):
        line= lines[ix]
        if re.match(rline, line):
            line = line.strip()
            level = line.split(" ")[0].count("=")
            heading = line.strip("=").strip()
            block = {
                "heading": heading,
                "level": level,
                "text": []
            }
            #print("\t"*level+heading)
            if level > parent_level:
                #print("\t"*level+"going down")
                ix = parse_block(lines, block, level, ix+1)
                parent["childs"].append(block)
            else:
                #print("\t"*level+"going up")
                return ix-1
        else:
           if "text" in block:
               #if len(block["text"])==0: print("\t"*block["level"]+"adding text to" + block["heading"])
               block["text"].append(line)
        ix += 1
    return ix

def postprocess(node,level=0, verbose=False):
    count = 0
    if "text" in node:
        stripped_text = parse_sparql_wdtmp("\n".join(node["text"]), node)
        count+=len(node["queries"]) if "queries" in node else 0
        node["text"] = stripped_text
        if verbose:
            print("\t"*level+node["heading"]+";".join([q["query"] for q in node["queries"]]))

    for child in node["childs"]:
        count+=postprocess(child, level+1, verbose)
    return count

def parse_sparql_templates(wikitext, verbose=False):
    root = {}
    #check if wikitext is a string
    if isinstance(wikitext, str):
       wikitext = wikitext.splitlines()
    parse_block(wikitext, root, 0)
    count= postprocess(root, 0,  verbose)
    root["count"] = count
    return root

def extract_queries_from_parse(node, results = [], context=""):
    new_ctx=context
    if "heading" in node:
        new_ctx += "\n" + node["heading"]
    if "text" in node:
        new_ctx += "\n" + node["text"]
    if "queries" in node:
        for q in node["queries"]:
            desc = "\n".join([v[1] for v in q["metadata"]]) if "metadata" in q else new_ctx
            results.append(_get_query_entry(q["query"], desc , new_ctx))
    for child in node["childs"]:
        extract_queries_from_parse(child, results, new_ctx)
    return results


def _get_query_entry(query, desc, context=""):
    if query.startswith("embed.html"):
        query = query[len("embed.html"):]
    return { "query": query,
            "metadata": [{"description":desc, "context": context}]}


def parse_sql_links(pagelist, verbose=False):
    results = []
    wdq_inner = r"http[s]://query.wikidata.org/(\S+)"
    wdquery = r"\[http[s]://query.wikidata.org/(\S+) +(.*)\]"
    wquery= r"\[(https://w.wiki/(\S+)) +(.*)\]"
    for page in pagelist:
        #print(page.text)
        #text = page.text
        #results[page.title()] = parse_sparql_templates(text, verbose=False)
        #rint(f"Found {results[page.title()]['count']} queries in {page.title()}")

        query_links = re.finditer(wdquery, page.text)
        for query in query_links:
            results.append(_get_query_entry(unquote(query.groups()[0]), query.groups()[1])
            )

        query_links = re.finditer(wquery, page.text)
        for query in query_links:
            try:
                print("Resolving ", query.groups()[0], " Text: ", query.groups()[2])
                response = requests.get(query.groups()[0], allow_redirects=False)
                if response.status_code == 301:
                    # The URL has been redirected
                    full_url = response.headers["location"]
                    match = next(re.finditer(wdq_inner, full_url))
                    if match:
                        desc = query.groups()[2] if query.groups()[2].find("]")==-1 else query.groups()[2][:query.groups()[2].find("]")]
                        results.append(_get_query_entry(unquote(match.groups()[0]), desc))
                else:
                    print("no redirect for ", query.groups()[0], "\n", query.groups()[2])
            except Exception as e:
                print("failed to resolve ", query.groups()[0], "\n", query.groups()[2], e)

    return results

if __name__ == '__main__':

    with open('raw/crawled_queries.txt', 'r') as file:
        lines = file.readlines()

    text = "\n".join(lines)
    print(parse_sparql_templates(text))