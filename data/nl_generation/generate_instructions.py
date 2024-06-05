# LLMs Intructions generation based on Sparql queries
# Todo: 1. Generate X re-formulated nl_generation for each query context + description using GPT3.5 and OpenAssistant.
# Todo: 2. Prepare the nl_generation for the LLMs with different templates.
import json
import os
import time
import tqdm

from data.nl_generation import utils


def encode_prompt(query_dict, with_context=True):
    """Encode multiple prompt nl_generation into a list of strings."""
    template = open("prompt.txt").read() + "\n"
    messages = [{"role": "system", "content": template}]
    (context, description, query, _id) = query_dict["context"], query_dict["description"], query_dict["query"], \
        query_dict["id"]
    # prompt = template
    # prompt += f"\n"
    prompt = ""
    if with_context:
        prompt += f'Context: "{context}"\n'
        if description != "":
            prompt += f'Description: "{description}"\n\n'
    else:
        if description != "":
            prompt += f'Context: "{description}"\n\n'
    prompt += f"SPARQL query: ```{query}```\n"
    messages.append({"role": "user", "content": prompt})
    return messages


def estimate_tokens_cost(count, model="gpt-3.5-turbo"):
    if model == "gpt-3.5-turbo":
        return (count / 1000) * 0.002
    elif model == "gpt-4":
        return (count / 1000) * 0.06


def post_process_llama3_response(response):
    """validate that the response is a json and evaluate the json response from GPT3 and return the processed nl_generation."""
    try:
        response = json.loads(response)
        if "nl_generation" in response:
            return response["nl_generation"], True
        else:
            return response, False
    except json.decoder.JSONDecodeError:
        print("Error: response is not a valid json")

    return response, False


def generate_instruction(output_dir="../processed/",
                         seed_queries_path="/mnt/ext/Research/Data/final_fq17_no_limit/final_fq17-generated_prompt_query_annotated.json",
                         model_name="llama3",
                         temperature=0.0,
                         top_p=1.0,
                         max_tokens=8192,
                         estimate=False):
    """This function is to generate the nl_generation for the LLMs based on the query, context, description and template."""
    # Load the seed queries
    sparql_queries = json.load(open(seed_queries_path, "r"))

    # Generate the nl_generation
    seed_queries = [
        {"context": sparql_queries["context"][str(i)], "description": sparql_queries["description"][str(i)],
         "query": sparql_queries["query_templated"][str(i)], "id": str(i)}
        for i in range(len(sparql_queries['query'].keys()))
    ]
    print(f"Loaded {len(seed_queries)} scraped WikiData SPARQL queries.")

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0

    # now let's generate new nl_generation for every query!
    generated_instructions = []
    extended_seed_queries = sparql_queries
    extension_dict = dict()
    if os.path.exists(os.path.join(output_dir, "sparql_instructions.json")):
        generated_instructions = utils.jload(os.path.join(output_dir, "sparql_instructions.json"))
        print(f"Loaded {len(generated_instructions)} LLM-processed nl_generation")
    if os.path.exists(os.path.join(output_dir, seed_queries_path.split("/")[-1])):
        extended_seed_queries = utils.jload(os.path.join(output_dir, seed_queries_path.split("/")[-1]))
        extension_dict = extended_seed_queries.get("nl_generation", dict())
        print(f"Loaded {len(generated_instructions)} LLM-processed nl_generation")
    progress_bar = tqdm.tqdm(total=len(seed_queries))
    invalid_responses = []
    if generated_instructions:
        # check invalid responses
        invalid_responses = [i for i, gen in enumerate(generated_instructions) if not gen["valid_gpt_response"]]
        progress_bar.update(len(generated_instructions) - len(invalid_responses))

    if estimate:
        # estimating total cost
        token_counts = []
        for query_dict in seed_queries:
            # generate nl_generation for a single query
            messages = encode_prompt(query_dict)
            token_counts.append(utils.num_tokens_from_messages(messages, model=model_name))
        print(
            f"Total tokens: {sum(token_counts)}, average tokens: {sum(token_counts) / len(token_counts)}, max tokens: {max(token_counts)}")
        total_cost = estimate_tokens_cost(sum(token_counts), model=model_name)
        print(f"Total cost for model: {total_cost} $")
    else:
        total_tokens = 0
        total_cost = 0
        try:
            for idx, query_dict in enumerate(seed_queries):
                request_idx += 1
                if idx < len(generated_instructions) and idx not in invalid_responses:
                    continue
                # generate nl_generation for a single query
                if idx in invalid_responses:
                    print(f"Re-generating instruction for query {idx} with no context")
                    messages = encode_prompt(query_dict, with_context=True)
                else:
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
                instructions, valid = post_process_llama3_response(response.message.content)
                gen_dict = {"id": query_dict["id"], "nl_generation": instructions, "valid_gpt_response": valid,
                            "query": query_dict["query"]}
                if idx in invalid_responses:
                    generated_instructions[idx] = gen_dict
                else:
                    generated_instructions.append(gen_dict)
                extension_dict[query_dict["id"]] = instructions
                progress_bar.update(1)

                utils.jdump(generated_instructions, os.path.join(output_dir, "sparql_instructions.json"))
                extended_seed_queries["nl_generation"] = extension_dict
                utils.jdump(extended_seed_queries, os.path.join(output_dir, seed_queries_path.split("/")[-1]))
                if total_cost >= 20:
                    print(f"Total tokens: {total_tokens}, Accumulated cost: {total_cost} $")
                    print(f"Saved {len(generated_instructions)} processed nl_generation")
                    raise Exception("Cost limit reached")
        except Exception or KeyboardInterrupt as e:
            print(f"Error: {e}")
            utils.jdump(generated_instructions, os.path.join(output_dir, "sparql_instructions.json"))
            utils.jdump(extended_seed_queries, os.path.join(output_dir, seed_queries_path.split("/")[-1]))
            print(f"Saved {len(generated_instructions)} processed nl_generation")
            print(f"Total tokens: {total_tokens}, Accumulated cost: {total_cost} $")
            raise e


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    # fire.Fire(main)
    generate_instruction(estimate=False)
