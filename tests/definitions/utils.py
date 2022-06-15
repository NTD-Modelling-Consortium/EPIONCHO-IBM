from typing import Dict, Union

Flat = Union[float, int]
InnerDictType = Union[
    Flat,
    Dict[str, "InnerDictType"],
]
DictType = Dict[str, InnerDictType]
FlatDict = Dict[str, Flat]


def flatten_dict(input_dict: DictType, prefix: str = "") -> FlatDict:
    if prefix == "":
        effective_prefix = ""
    else:
        effective_prefix = prefix + "_"
    output_dict = {}
    for k, v in input_dict.items():
        if isinstance(v, dict):
            new_dict = flatten_dict(v, prefix=effective_prefix + k)
        else:
            new_dict = {effective_prefix + k: v}
        output_dict.update(new_dict)
    return output_dict
