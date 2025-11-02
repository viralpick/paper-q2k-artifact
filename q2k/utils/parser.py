import re

URL_OR_DOMAIN = re.compile(r"(https?://|www\.|\w+\.[a-z]{2,6})")


def strip_url_annotations(text: str) -> str:
    """
    Strips URL annotations from a text by removing parentheses that contain URLs or domains.
    """
    result = []
    i = 0
    n = len(text)

    while i < n:
        if text[i] == "(":
            depth = 1
            j = i + 1
            while j < n and depth:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                j += 1

            if depth == 0:
                inner = text[i + 1 : j - 1]
                if URL_OR_DOMAIN.search(inner):
                    i = j
                    continue
                result.append(text[i:j])
                i = j
                continue
            else:
                result.append(text[i])
                i += 1
        else:
            result.append(text[i])
            i += 1

    return "".join(result)
