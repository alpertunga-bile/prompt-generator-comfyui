from string import punctuation
from re import sub, compile
from collections import OrderedDict


def get_unique_list(sequence: list) -> list:
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def remove_exact_keywords(line: str) -> list[str]:
    char_blacklist = set(f"{punctuation}0123456789")

    # remove exact prompts
    prompts = get_unique_list(line.split(","))

    pure_prompts = OrderedDict()  # order matters
    extracted_pure_prompts = {}  # order isn't important

    # remove exact keyword
    for prompt in prompts:
        # extract the keyword
        keyword = "".join(c for c in prompt if c not in char_blacklist).lstrip()

        if keyword == "":
            continue

        if prompt in pure_prompts or keyword in extracted_pure_prompts:
            continue

        pure_prompts[prompt] = True
        extracted_pure_prompts[keyword] = True

    return pure_prompts.keys()


def preprocess(line: str, preprocess_mode: str) -> str:
    pattern = compile(r"(,\s){2,}")

    temp_line = line.replace("\xa0", " ")
    temp_line = temp_line.replace("\n", ", ")
    temp_line = temp_line.replace("\t", " ")
    temp_line = temp_line.replace("|", ",")
    temp_line = temp_line.replace("  ", " ")
    temp_line = sub(pattern, ", ", temp_line)

    if preprocess_mode == "exact_keyword":
        temp_line = ",".join(remove_exact_keywords(temp_line))
    elif preprocess_mode == "exact_prompt":
        temp_line = ",".join(get_unique_list(temp_line.split(",")))

    return temp_line
