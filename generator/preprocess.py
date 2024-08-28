from re import compile
from collections import OrderedDict, namedtuple


def get_unique_list(sequence: list[str]) -> list[str]:
    seen = set()
    return [
        x.strip() for x in sequence if not (x.strip() in seen or seen.add(x.strip()))
    ]


def remove_exact_keywords(line: str) -> list[str]:
    find_empty_parantheses_regex = compile(r"\(\s*\)")
    remove_nonprompts_regex = compile(r"[^a-zA-Z_\-()\[\]{}]*")
    remove_nonweighters_regex = compile(r"[()\[\]{}]*")
    remove_inside_regex = compile(r"[^()\[\]{}]*")

    # remove exact prompts
    prompts = get_unique_list(line.split(","))

    pure_prompts = OrderedDict()  # order matters, it contains prompts' original forms
    extracted_pure_prompts = set()  # order doesn't matters, it contains prompt keywords

    # remove exact keyword
    for prompt in prompts:
        tempPrompt = remove_nonprompts_regex.sub(
            "", prompt
        ).lstrip()  # from -> ((masterpiece:1.2)) | to -> ((masterpiece))

        if len(tempPrompt) == 0:
            continue

        tempPrompt = remove_nonweighters_regex.sub(
            "", tempPrompt
        )  # from -> ((masterpiece)) | to -> masterpiece

        if tempPrompt in extracted_pure_prompts:
            tempPrompt = remove_inside_regex.sub(
                "", prompt
            )  # from -> ((masterpiece:1.2)) | to -> (())

            if (
                len(find_empty_parantheses_regex.findall(tempPrompt))
                > 0  # find balanced parantheses count
            ):
                # check balanced parantheses
                inner_parant_count = tempPrompt.count("(")
                outer_parant_count = tempPrompt.count(")")

                if inner_parant_count == outer_parant_count:
                    continue

                lowest_count = min(inner_parant_count, outer_parant_count)

                # remove balanced parantheses
                # because it is going to be appended to a string
                for _ in range(lowest_count):
                    tempPrompt = tempPrompt.replace("()", "")

                pure_prompts[tempPrompt] = True

            continue

        extracted_pure_prompts.add(tempPrompt)
        pure_prompts[prompt] = True

    return pure_prompts.keys()


def fix_commas(string: str) -> str:
    remove_multiwhitespaces_regex = compile(r"\s+")
    remove_nonpromptcommas_regex = compile(r"(,\s){2,}")

    temp_str = remove_multiwhitespaces_regex.sub(" ", string)
    return remove_nonpromptcommas_regex.sub(", ", temp_str)


def fix_artifacts(string: str) -> str:
    temp_string = string

    ArtifactFix = namedtuple("ArtifactFix", ["regex", "new_str"])

    replace_list = [
        ArtifactFix(regex=compile(r"\(\s*,"), new_str="("),
        ArtifactFix(regex=compile(r"\[\s*,"), new_str="["),
        ArtifactFix(regex=compile(r"{\s*,"), new_str="{"),
        ArtifactFix(regex=compile(r",\s*\)"), new_str=")"),
        ArtifactFix(regex=compile(r",\s*\]"), new_str="]"),
        ArtifactFix(regex=compile(r",\s*}"), new_str="}"),
    ]

    # fixing the artifacts
    for artifact_fix in replace_list:
        temp_string = artifact_fix.regex.sub(artifact_fix.new_str, temp_string)

    return temp_string


def preprocess(line: str, preprocess_mode: str) -> str:
    remove_scalarweights_regex = compile(r",\s*:[0-9]*\.?[0-9]+")
    remove_emptyprompts_regex = compile(r",\s+[()\[\]{}]+\s*,")
    remove_danglingparantheses_regex = compile(r"\B\s+|\s+\B")

    temp_line = line.encode("ascii", "xmlcharrefreplace").decode()
    temp_line = temp_line.encode("utf-8", "xmlcharrefreplace").decode()

    temp_line = temp_line.replace("\xa0", " ")
    temp_line = temp_line.replace("\n", ", ")

    temp_line = fix_commas(temp_line)

    temp_line = remove_scalarweights_regex.sub(
        "", temp_line
    )  # from -> , 0.6 | to -> *empty string*

    if preprocess_mode == "exact_keyword":
        temp_line = ", ".join(remove_exact_keywords(temp_line))

        temp_line = fix_commas(temp_line)
        temp_line = fix_artifacts(temp_line)

        temp_line = remove_emptyprompts_regex.sub(
            ",", temp_line
        )  # from -> , (((, | to -> ,

        temp_line = remove_danglingparantheses_regex.sub("", temp_line).replace(
            ",", ", "
        )  # from -> (( ((prompt)) | to -> ((prompt))

    elif preprocess_mode == "exact_prompt":
        temp_line = ", ".join(get_unique_list(temp_line.split(",")))

    return temp_line
