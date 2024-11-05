import re
import time
import urllib.error
from functools import lru_cache
from typing import List, Tuple

import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from SPARQLWrapper.SPARQLExceptions import EndPointInternalError
from requests.exceptions import HTTPError, ConnectionError
from tenacity import wait_random_exponential, stop_after_attempt, retry, retry_if_exception_type


class RetryAfterException(Exception):
    pass


class SPARQLResponse:
    def __init__(self, data) -> None:
        self.data = data
        if isinstance(data, dict):
            if "results" in data and "bindings" in data["results"]:
                self.bindings = data['results']['bindings']
                self.success = True
        else:
            self.bindings = False
            self.success = False


class WikidataAPI:
    extraction_regex = re.compile(r"\[(entity|property):([\w\s,:;'`\".!?]+)\]")

    def __init__(self, base_url: str = "https://www.wikidata.org/w/api.php",
                 sparql_endpoint: str = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql',
                 start_tag: str = "[QUERY]",
                 end_tag: str = "[/QUERY]",
                 timeout: int = 60,
                 agent_id=0) -> None:
        self.base_url = base_url
        self.sparql_agent = f"WikidataLLM Query evaluator v{agent_id}"
        self.api_agent = f"WikidataLLM Entity Translator v{agent_id}"
        self.sparql = SPARQLWrapper(sparql_endpoint, returnFormat=JSON,
                                    agent=self.sparql_agent)
        self.sparql.setTimeout(timeout)
        self.start_tag = start_tag
        self.end_tag = end_tag

    def _recover_redirected_id(self, name: str, is_property: bool = False):
        if is_property:
            print(f"{name=}")
            raise NotImplementedError("Not implemented for property yet.")

        endpoint = "http://www.wikidata.org/"
        endpoint += "property/" if is_property else "entity/"

        response = requests.get(f"{endpoint}{name}", allow_redirects=True)
        data = response.json()
        return list(data['entities'].keys())[0]

    def _get_label_from_wbsearchentities(self, item):
        if 'label' in item['display'].keys():
            return (item['id'], item['display']['label']['value'])
        elif 'description' in item['display'].keys():
            return (item['id'], item['display']['description']['value'])
        else:
            raise NotImplementedError("Not implemented for case where there is no label or description.")

    def _get_response_from_wbsearchentities(self, name: str, search_property: bool = False):
        payload = {
            "action": "wbsearchentities",
            "search": name,
            "language": "en",
            "format": "json"
        }

        if search_property:
            payload.update({"type": "property"})

        response = requests.get(self.base_url, params=payload, headers={'User-agent': f"{self.api_agent}"})
        response.raise_for_status()
        return response

    def _check_and_get_labels_from_wbsearchentities_response(self, response: requests.Response):
        data = response.json()

        if not 'search' in data.keys():
            raise KeyError("The search key is not in the response data.")

        items = data["search"]

        if len(items) == 0:
            raise Exception("There is no results.")

        results = []
        for item in items:
            if not 'display' in item.keys():
                raise KeyError("There is no 'display' in item results.")

            if len(item['display']) == 0:
                raise NameError("It has been redirected.")

            results.append(self._get_label_from_wbsearchentities(item))

        return results

    def _retry_after_middle_man(self, func, num_retries: int = 3, **func_kwargs):
        is_error = True
        num_connection_retries = 3
        while num_retries > 0 and num_connection_retries > 0 and is_error:
            try:
                response = func(**func_kwargs)
                is_error = False
            except HTTPError as inst:
                if inst.response.status_code == 429:
                    retry_after = int(inst.response.headers['retry-after'])
                    time.sleep(retry_after + 1)
                    num_retries -= 1
                else:
                    raise inst
            except ConnectionError:
                num_connection_retries -= 1
                time.sleep((3 - num_connection_retries) * 10)
        return response

    def _smart_get_label_from_wbsearchentities(self, name: str, is_property: bool = False, num_recurrence=1):
        if num_recurrence < 0:
            raise RecursionError("The recursion limit set has been exceeded. No name has been found.")

        response = self._retry_after_middle_man(self._get_response_from_wbsearchentities, num_retries=3, name=name,
                                                search_property=is_property)

        try:
            return self._check_and_get_labels_from_wbsearchentities_response(response=response)
        except KeyError as inst:
            raise inst
        except NameError:
            name = self._recover_redirected_id(name, is_property=is_property)
            try:
                return self._smart_get_label_from_wbsearchentities(name, is_property=is_property,
                                                                   num_recurrence=num_recurrence - 1)
            except RecursionError as e:
                raise e
            except Exception as inst:
                raise inst
        except Exception:
            try:
                return self._smart_get_label_from_wbsearchentities(name, is_property=(not is_property),
                                                                   num_recurrence=num_recurrence - 1)
            except RecursionError as e:
                raise e
            except Exception as inst:
                raise inst

    # This function get way more data than _get_response_from_wbsearchentities
    # but has the benefit of not caring if the id is entity or property.
    # Because Ids should be unique, it should works (obviously): https://www.wikidata.org/wiki/Wikidata:Identifiers
    def _get_response_from_entity_id(self, id: str):
        endpoint = "https://www.wikidata.org/entity/"
        response = requests.get(f"{endpoint}{id}", headers={'User-agent': 'WikidataLLM bot v0'}, allow_redirects=True)
        response.raise_for_status()

        return response

    def _check_and_get_labels_from_entity_response(self, response: requests.Response):
        try:
            data = response.json()
        except:
            raise NameError("The id doesn't exist.")

        results = []

        if 'entities' in data.keys():
            entities = data['entities']
        else:
            raise KeyError("No key entities in data.")

        for entity_id in entities.keys():
            if len(entities[entity_id]['labels']) == 0:
                raise NotImplementedError("The entity doesn't have any label.")

            labels = entities[entity_id]['labels']

            if 'en' in labels.keys():
                label = labels['en']['value']
            else:
                firstLanguage = list(labels.keys())[0]
                label = labels[firstLanguage]['value']
                print(
                    f"The id={entity_id} doesn't have english labels. Taking the first in the list of labels ({firstLanguage} => {label}).")

            results.append((entity_id, label.strip()))

        return results

    @lru_cache(maxsize=32768)
    def _smart_get_labels_from_entity_id(self, name: str):
        try:
            response = self._retry_after_middle_man(self._get_response_from_entity_id, num_retries=3, id=name)
        except HTTPError as inst:
            # Id is incorrect
            return [(name, name)]

        return self._check_and_get_labels_from_entity_response(response)

    def extract_entities_and_properties(self, text: str) -> Tuple[List[str], List[str]]:
        entities = []
        properties = []

        extraction = self.extraction_regex.findall(text)

        if extraction:
            for ttype, label in extraction:
                if ttype == "entity":
                    entities.append(label)
                elif ttype == "property":
                    properties.append(label)

        return entities, properties

    def replace_annotations(self, query: str) -> str:
        entities, properties = self.extract_entities_and_properties(query)
        linked_entities = [(entity, self.find_entities(entity)[0]) for entity in entities]
        linked_properties = [(prop, self.find_properties(prop)[0]) for prop in properties]

        for entity, (entity_id, _) in linked_entities:
            query = re.sub(rf"\[entity:{entity}\]", entity_id, query)
        for prop, (prop_id, _) in linked_properties:
            query = re.sub(rf"\[property:{prop}\]", prop_id, query)

        return query

    def find_entities(self, name: str) -> List[Tuple[str, str]]:
        try:
            return self._smart_get_label_from_wbsearchentities(name, is_property=False)
        except Exception as exception:
            name_parts = name.split(" ")
            if len(name_parts) > 1:
                for i in range(1, len(name_parts)):
                    try:
                        return self._smart_get_label_from_wbsearchentities(" ".join(name_parts[:i]),
                                                                           is_property=False)
                    except Exception as e:
                        return [(name, name)]
            else:
                return [(name, name)]

    def find_properties(self, name: str) -> List[Tuple[str, str]]:
        try:
            return self._smart_get_label_from_wbsearchentities(name, is_property=True)
        except Exception as exception:
            name_parts = name.split(" ")
            if len(name_parts) > 1:
                for i in range(1, len(name_parts)):
                    try:
                        return self._smart_get_label_from_wbsearchentities(" ".join(name_parts[:i]),
                                                                           is_property=True)
                    except Exception as e:
                        return [(name, name)]
            else:
                return [(name, name)]

    def execute_sparql(self, query: str, timeout: int = None) -> SPARQLResponse:
        # url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
        # response = requests.get(url, params={'query': query, 'format': 'json'},
        #                         headers={'User-agent': 'WikidataLLM bot v0'}, timeout=timeout)
        # response.raise_for_status()
        if timeout:
            self.set_timeout(timeout)
        response = self.query(query)

        return SPARQLResponse(response)

    def set_timeout(self, timeout: int):
        self.sparql.setTimeout(timeout)

    @retry(reraise=True, wait=wait_random_exponential(min=60, max=70), stop=stop_after_attempt(3),
           retry=retry_if_exception_type(RetryAfterException))
    def query(self, query):
        """
        Query the endpoint with the given query.

        :param query: The query to execute.
        :type query: str
        :return: The result of the query.
        :rtype: dict
        """
        try:
            self.sparql.setQuery(query)
            return self.sparql.query().convert()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise RetryAfterException("Too many requests. Please try again later.")
            else:
                raise e

    def extract_query(self, text: str) -> str:
        """Extract the query from the LLM processed text"""
        # find the query between the start and end tags
        start_idx = text.find(self.start_tag)
        if start_idx == -1:
            return ""
        start_idx += len(self.start_tag)
        end_idx = text.find(self.end_tag, start_idx)
        if end_idx == -1:
            end_idx = text.find(self.start_tag, start_idx)  # Some models struggle with the end tag
            if end_idx == -1:
                return ""
        return text[start_idx:end_idx].strip()
