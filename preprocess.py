from string import punctuation
from re import sub, compile

def get_unique_list(sequence : list) -> list:
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def remove_exact_keywords(line : str) -> list[str]:
    char_blacklist = set(f"{punctuation}0123456789")

    # remove exact prompts
    prompts = get_unique_list(line.split(","))
    pure_prompts = []

    # remove exact keyword
    for prompt in prompts:
        can_add = True
        # extract the keyword
        keyword = "".join(c for c in prompt if c not in char_blacklist).lstrip()

        if keyword == "":
            continue

        for pure_prompt in pure_prompts:
            if prompt == pure_prompt:
                can_add = False
                break

            extracted_pure_prompt = "".join(c for c in pure_prompt if c not in char_blacklist).lstrip()
            if keyword == extracted_pure_prompt:
                can_add = False
                break

        if can_add:
            pure_prompts.append(prompt)

    return pure_prompts

def preprocess(line : str, preprocess_mode : str) -> str:
    pattern = compile(r'(,\s){2,}')

    temp_line = line.replace(u'\xa0', u' ')
    temp_line = temp_line.replace("\n", ", ")
    temp_line = temp_line.replace("\t", " ")
    temp_line = temp_line.replace("|", ",")
    temp_line = temp_line.replace("  ", " ")
    temp_line = sub(pattern, ', ', temp_line)

    if preprocess_mode == "exact_keyword":
        temp_line = ','.join(remove_exact_keywords(temp_line))
    elif preprocess_mode == "exact_prompt":
        temp_line = ','.join(get_unique_list(temp_line.split(",")))

    return temp_line