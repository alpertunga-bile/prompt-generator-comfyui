from re import sub, findall, compile
from collections import OrderedDict


def get_unique_list(sequence: list) -> list:
    seen = set()
    return [x for x in sequence if not (x.strip() in seen or seen.add(x.strip()))]


def remove_exact_keywords(line: str) -> list[str]:
    # remove exact prompts
    prompts = get_unique_list(line.split(","))

    remove_nonprompts_regex = compile(r"[^a-zA-Z()\[\]{}]*")
    remove_nonweighters_regex = compile(r"[()\[\]{}]*")
    remove_inside_regex = compile(r"[^()\[\]{}]*")
    empty_parantheses_remove_regex = compile(r"\(\s*\)")

    pure_prompts = OrderedDict()  # order matters
    extracted_pure_prompts = {}  # order isn't important

    # remove exact keyword
    for prompt in prompts:
        tempPrompt = sub(remove_nonprompts_regex, "", prompt).lstrip()

        if tempPrompt == "":
            continue

        tempPrompt = sub(remove_nonweighters_regex, "", tempPrompt)

        if tempPrompt in extracted_pure_prompts:
            tempPrompt = sub(remove_inside_regex, "", prompt)

            if (
                len(findall(empty_parantheses_remove_regex, tempPrompt)) > 0
                or tempPrompt == ""
            ):
                continue

            pure_prompts[tempPrompt] = True
            continue

        if tempPrompt == "":
            pure_prompts[tempPrompt] = True
        else:
            extracted_pure_prompts[tempPrompt] = True
            pure_prompts[prompt] = True

    return pure_prompts.keys()


def preprocess(line: str, preprocess_mode: str) -> str:

    temp_line = line.encode("ascii", "xmlcharrefreplace").decode()
    temp_line = temp_line.encode(errors="xmlcharrefreplace").decode()

    temp_line = temp_line.replace("\xa0", " ")
    temp_line = temp_line.replace("\n", ", ")
    temp_line = temp_line.replace("\t", " ")
    temp_line = sub(r"\s+", " ", temp_line)
    temp_line = sub(r"(,\s){2,}", ", ", temp_line)  # remove non prompt commas
    temp_line = sub(
        r",\s*:[0-9]*\.?[0-9]+", "", temp_line
    )  # remove non prompt scalar weights

    if preprocess_mode == "exact_keyword":
        temp_line = ", ".join(remove_exact_keywords(temp_line))

        temp_line = temp_line.replace("(,", "(")
        temp_line = temp_line.replace("[,", "[")
        temp_line = temp_line.replace("{,", "{")

        temp_line = sub(
            r"\s+", " ", temp_line
        )  # replace multiple whitespaces with one whitespace
        temp_line = sub(r"\s+,", "", temp_line)  # remove nonprompt commas

        temp_line = temp_line.replace(", )", ")")
        temp_line = temp_line.replace(", ]", "]")
        temp_line = temp_line.replace(", }", "}")

    elif preprocess_mode == "exact_prompt":
        temp_line = ",".join(get_unique_list(temp_line.split(",")))

    return temp_line
