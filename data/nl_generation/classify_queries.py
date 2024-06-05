import json
import os
import time

import tqdm

from data.nl_generation import utils


system_prompt = """You are a helpful assistant. Your task is to analyse the given SPARQL query and classify the complexity 
of the query. Return a JSON object with the keys 'complexity_description' and 'complexity' with the value being either 'simple', 'medium', or 'complex'.\n
Here are the conditions:
1. Make the description of the complexity very short with keywords highlighting why the query is simple, medium, or complex.
2. Respond in JSON format with 'complexity_description' and 'complexity' fields, ONLY !important"""

def encode_prompt(query_dict, with_context=True):
    """Encode multiple prompt nl_generation into a list of strings."""
    sys_prompt = system_prompt + "\n"
    messages = [{"role": "system", "content": sys_prompt}]
    query = query_dict["query"]

    prompt = f"SPARQL query: ```{query}```\n"
    messages.append({"role": "user", "content": prompt})
    return messages


def post_process_llama3_response(response):
    """validate that the response is a json and evaluate the json response from Llama3 and return the processed nl_generation."""
    try:
        response = json.loads(response)
        if "complexity" in response and "complexity_description" in response:
            return response, True
        else:
            return response, False
    except json.decoder.JSONDecodeError:
        print("Error: response is not a valid json")

    return response, False

def generate_sparql_classification(output_dir="../processed/",
                                   seed_queries_path="/mnt/ext/Research/Data/final_fq17_with_no_limit/final_fq17-generated_prompt_query_annotated.json",
                                   model_name="llama3",
                                   temperature=0.0,
                                   top_p=1.0,
                                   max_tokens=8192):
    """This function is to generate the nl_generation for the LLMs based on the query, context, description and template."""
    # Load the seed queries
    sparql_queries = json.load(open(seed_queries_path, "r"))

    # Generate the nl_generation
    seed_queries = [
        {"query": sparql_queries["query"][str(i)], "id": str(i)}
        for i in range(len(sparql_queries['query'].keys()))
    ]
    print(f"Loaded {len(seed_queries)} scraped WikiData SPARQL queries.")

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0

    # now let's generate new nl_generation for every query!
    generated_complexities = []
    if os.path.exists(os.path.join(output_dir, "sparql_complexities.json")):
        generated_complexities = utils.jload(os.path.join(output_dir, "sparql_complexities.json"))
        print(f"Loaded {len(generated_complexities)} LLM-processed nl_generation")
    progress_bar = tqdm.tqdm(total=len(seed_queries))
    invalid_responses = []
    if generated_complexities:
        # check invalid responses
        invalid_responses = [i for i, gen in enumerate(generated_complexities) if not gen["valid_gpt_response"]]
        progress_bar.update(len(generated_complexities) - len(invalid_responses))

    try:
        for idx, query_dict in enumerate(seed_queries):
            request_idx += 1
            if idx < len(generated_complexities) and idx not in invalid_responses:
                continue
            # generate complexity for a single query

            messages = encode_prompt(query_dict)

            decoding_args = utils.OpenAIDecodingArguments(
                temperature=temperature,
                n=1,
                max_tokens=max_tokens,
                top_p=top_p,
                # stop=["}\n"]
            )
            request_start = time.time()
            response = utils.openai_completion(
                messages=messages,
                model_name=model_name,
                decoding_args=decoding_args,
                # logit_bias={"50256": -100},
            )
            request_duration = time.time() - request_start
            print(f"Request {request_idx} took {request_duration:.2f} seconds.")

            # post-process the response
            # total_tokens += response["total_tokens"]
            # total_cost = estimate_tokens_cost(total_tokens, model=model_name)
            # print(f"Total tokens: {total_tokens}, Accumulated cost: {total_cost} $")
            complexity, valid = post_process_llama3_response(response.message.content)
            gen_dict = {"id": query_dict["id"], "valid_gpt_response": valid,
                        "query": query_dict["query"] , **complexity}
            if idx in invalid_responses:
                generated_complexities[idx] = gen_dict
            else:
                generated_complexities.append(gen_dict)
            progress_bar.update(1)

            utils.jdump(generated_complexities, os.path.join(output_dir, "sparql_complexities.json"))
    except Exception or KeyboardInterrupt as e:
        print(f"Error: {e}")
        utils.jdump(generated_complexities, os.path.join(output_dir, "sparql_complexities.json"))
        print(f"Saved {len(generated_complexities)} processed queries")
        raise e

def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    # fire.Fire(main)
    generate_sparql_classification()