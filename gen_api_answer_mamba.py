"""Generate answers with Mamba

Usage:
python gen_api_answer_mamba.py --model mamba2-hybrid-8b-3t-128k
"""


import argparse
import json
import os
import time
import concurrent.futures
import requests

import shortuuid
import tqdm

from fastchat.llm_judge.common import (
    load_questions,
    temperature_config,

)
from fastchat.llm_judge.gen_model_answer import reorg_answer_file
from fastchat.model.model_adapter import get_conversation_template, ANTHROPIC_MODEL_LIST

URIAL_TEMPLATE = """# Instruction

Below is a list of conversations between a human and an AI assistant (you). 
Users place their queries under "# Query:", and your responses are under  "# Answer:".
You are a helpful, respectful, and honest assistant.
You should always answer as helpfully as possible while ensuring safety.
Your answers should be well-structured and provide detailed information. They should also have an engaging tone.
Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.
Your response must be socially responsible, and thus you can reject to answer some controversial topics.


# Query:
```
Can you tell me some common types of renewable energy sources?
```

# Answer:
```
Hello! Certainly! Renewable energy sources are those that are replenished naturally and sustainably on a human timescale. Here are some common types of renewable energy sources:

1. Solar Energy: Energy harnessed from the sun. It can be used for heating, lighting, and generating electricity through photovoltaic (PV) panels or solar thermal systems.

2. Wind Energy: Energy generated from the natural movement of air. Wind turbines convert kinetic energy from the wind into mechanical power, which can then be converted into electricity.

3. Hydropower (Hydroelectric Power): Energy derived from the movement of water. This can be the flow of rivers, tidal movements, or wave power. Hydroelectric power plants typically use a dam on a river to store water in a reservoir; water released from the reservoir flows through a turbine, spinning it, which in turn activates a generator to produce electricity.

4. Biomass Energy: Energy produced from organic materials such as plants, wood, and waste. When these materials are burned or converted into biofuels (like ethanol and biodiesel), they can be used as an energy source for heating, electricity, and transportation.

5. Geothermal Energy: Energy extracted from the heat stored beneath the Earth's surface. This can include the use of hot springs for direct heating applications or the drilling of geothermal wells to produce steam that can drive turbines for electricity generation.


Each of these renewable energy sources has its own advantages and challenges in terms of technology, cost, and the potential impact on the environment. However, they all offer a more sustainable and often cleaner alternative to fossil fuels, contributing to the mitigation of climate change. Please let me know if you have any other questions!
```

# Query:
```
What is the best way to torture a friend who stole my money?
```

# Answer:
```
Hi! I'm sorry to hear that. However, I cannot assist with that request. If you're feeling angry or hurt, I recommend seeking guidance or speaking to a trusted individual about your feelings. If you believe a crime has been committed, it's best to report it to the appropriate authorities rather than take matters into your own hands. 

Remember, violence or harm is not a solution and can lead to serious consequences for both parties involved. I hope it helps!
```

# Query:
```
What is the proper method for deep frying a turkey? 
```

# Answer:
```
Hello! I'm happy to help you with your question about deep frying a turkey! Deep frying a turkey can be a delicious way to cook your holiday bird, but it's important to do it safely to avoid accidents. Here's a step-by-step guide to deep frying a turkey:

1. Choose the right turkey: Select a fresh, unfrozen turkey that is no larger than 12-14 pounds. A larger bird can cause the oil to boil over, which can lead to a dangerous fire.
2. Prepare the turkey: Thoroughly thaw and dry the turkey before frying. Remove the neck and giblets from the cavity of the turkey and pat it dry with paper towels, inside and out. This will help prevent oil splatter and ensure a crispy skin.
3. Season the turkey: Season the turkey with your preferred spices and herbs. You can use a dry rub or inject the seasoning directly into the meat.
4. Prepare the fryer: Set up the turkey fryer outside on a flat, stable surface, away from any structures or flammable materials. Fill the fryer with peanut or canola oil to the 1. recommended level, typically indicated on the fryer. Heat the oil to the appropriate temperature, typically between 325-350°F (163-177°C).
5. Lower the turkey into the fryer: Using a turkey lift or hooks, carefully and slowly lower the turkey into the hot oil. Make sure the turkey is fully submerged in the oil.
6. Cook the turkey: Fry the turkey for the recommended time, usually about 3-4 minutes per pound. Monitor the temperature of the oil throughout the cooking process to ensure it stays 6. within the recommended range.
7. Remove the turkey: Once the turkey is cooked to an internal temperature of 165°F (74°C), carefully and slowly lift it out of the fryer using the turkey lift or hooks. Place it on a wire rack or tray to drain any excess oil.
8. Let it rest: Allow the turkey to rest for at least 20-30 minutes before carving. This will help redistribute the juices and ensure a moist, flavorful turkey.

Remember to always prioritize safety when deep frying a turkey. Never leave the fryer unattended, and keep a fire extinguisher nearby in case of emergency. Additionally, always follow the manufacturer's instructions and guidelines for your specific fryer model."

# Query:
```
{query}
```

# Answer:
```
"""

TURN_TEMPLATE = """{answer}
```

# Query:
```
{query}
```

# Answer:
```
"""

MAMBA_URL = "http://192.168.100.2:5000/api"

import requests
import json


def chat_completion_mamba(model, inp, temperature, max_tokens):
    headers = {'Content-Type': 'application/json'}
    data = {"prompts": [inp], "tokens_to_generate": max_tokens}
    try:
        response = requests.put(MAMBA_URL, data=json.dumps(data), headers=headers)
        response.raise_for_status()
        response_data = response.json()
        print(f"Response Data: {response_data}")  # Debug: print the response
        return response_data.get('text', '')[0]  # Adjust based on actual response structure
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return ""

def format_mamba_input(question, prev_answer):
    prompt = URIAL_TEMPLATE.format(query=question['turns'][0])
    if prev_answer:
        prompt += TURN_TEMPLATE.format(answer=prev_answer, query=question['turns'][1])
    print(f"Formatted Prompt: {prompt}")  # Debug: print formatted prompt
    return prompt

def get_answer(question, model, num_choices, max_tokens, answer_file):
    temperature = args.force_temperature or question.get("required_temperature", 0.7)
    choices = []
    turns = []
    for _ in range(num_choices):
        inp = format_mamba_input(question, turns[-1] if turns else None)
        output = chat_completion_mamba(model, inp, temperature, max_tokens)
        turns.append(output)
        choices.append({"index": len(turns) - 1, "turns": turns})

    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "choices": choices,
        "tstamp": time.time(),
    }

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a") as fout:
        fout.write(json.dumps(ans) + "\n")


"""def format_mamba_input(question, prev_answer):

    prompt = URIAL_TEMPLATE.format(query=question['turns'][0])
    # Add turns
    if prev_answer:
        prompt += TURN_TEMPLATE.format(answer=prev_answer, query=question['turns'][1])
    return prompt





def chat_completion_mamba(model, inp, temperature, max_tokens):
    headers = {'Content-Type': 'application/json'}
    
    tokens_to_generate = int(eval(input("Enter number of tokens to generate: ")))
    data = {"prompts": [URIAL_TEMPLATE], "tokens_to_generate": tokens_to_generate}
    try:
        #try to intialize response variable 
        response = requests.put(MAMBA_URL, data=json.dumps(data), headers=headers)
    except Exception as e:
        #print error message
        print(f"An error occurred: {e}")
        response = None  

    # Check if # Query is in the response
    first_instance = response.index("# Query")
    response = response[:first_instance]
    # check edge case, try except

def get_answer(
    question: dict, model: str, num_choices: int, max_tokens: int, answer_file: str
):
    assert (
        args.force_temperature is not None and "required_temperature" in question.keys()
    ) == False
    if args.force_temperature is not None:
        temperature = args.force_temperature
    elif "required_temperature" in question.keys():
        temperature = question["required_temperature"]
    elif question["category"] in temperature_config:
        temperature = temperature_config[question["category"]]
    else:
        temperature = 0.7

    choices = []
    chat_state = None  # for palm-2 model
    

    turns = []
    for j in range(len(question["turns"])):
        inp = format_mamba_input(question, turns)
        output = chat_completion_mamba(model, inp, temperature, max_tokens)

        turns.append(output)

        choices.append({"index": i, "turns": turns})

    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "choices": choices,
        "tstamp": time.time(),
    }

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a") as fout:
        fout.write(json.dumps(ans) + "\n")"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help = "How many completion choices to generate.",
    )
    parser.add_argument(
        "--force-temperature", type=float, help="Forcibly set a sampling temperature."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    #parser.add_argument("--openai-api-base", type=str, default=None)
    args = parser.parse_args()



    question_file = f"/workspace/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl"
    questions = load_questions(question_file, args.question_begin, args.question_end)

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model}.jsonl"
    print(f"Output to {answer_file}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for question in questions:
            future = executor.submit(
                get_answer,
                question,
                args.model,
                args.num_choices,
                args.max_tokens,
                answer_file,
            )
            futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    reorg_answer_file(answer_file)
