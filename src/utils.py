from difflib import unified_diff

def get_diff(state_1, state_2):
    if isinstance(state_1, str):
        s1 = [line for line in state_1.split("\n")]
    else:
        assert isinstance(state_1, list)
        s1 = state_1

    if isinstance(state_2, str):
        s2 = [line for line in state_2.split("\n")]
    else:
        assert isinstance(state_2, list)
        s2 = state_2
    return list(unified_diff(s1, s2))[2:]