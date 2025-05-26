import whatthepatch
from difflib import unified_diff

def get_diff(state_1, state_2):
    if isinstance(state_1, str):
        s1 = state_1.split("\n")
    else:
        assert isinstance(state_1, list)
        s1 = state_1

    if isinstance(state_2, str):
        s2 = state_2.split("\n")
    else:
        assert isinstance(state_2, list)
        s2 = state_2
    return list(unified_diff(s1, s2))[2:]

def apply_patch(patch_code, base_code):
    try:
        patch = list(whatthepatch.parse_patch(patch_code))[0]
        return whatthepatch.apply_diff(patch, base_code)
    except:
        return ""
    updated_code = "\n".join(whatthepatch.apply_diff(patch, base_code))
    return updated_code

def convert_to_patch(code_snippet):
    if isinstance(code_snippet, str):
        code_snippet = [code_snippet.strip()]
    code_length = len(code_snippet)
    code_patch = "\n".join([f"@@ -0,0 +1,{code_length} @@"]+[f"+{line}" for line in code_snippet])
    return code_patch

def get_code_snippet(x):
    code_end = "\n```"
    for code_start in ["```python\n", "```\n"]:
        if code_start in x:
            start = x.rfind(code_start)+len(code_start)
            if code_end in x[start:]:
                end = x.find("\n```", start)
            else:
                return ""
            return x[start:end].strip()
    return ""