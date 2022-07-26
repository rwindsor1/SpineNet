from typing import Dict, List, Tuple
from csv import DictWriter
import pandas as pd
VertDicts = List[Dict]

def save_vert_dicts_to_csv(vert_dicts: VertDicts, filename: str) -> None:
    '''
    Saves a list of vert_dicts to a file in csv format.

    Parameters
    ----------
    vert_dicts: List[Dict]
        A list of dictionaries with the following keys: 
        'polys', 'average_polygon', 'slice_nos', 'predicted_label'
        Each entry corresponds to a detected vertebral body
    filename: str
        The name of the file to save the vert_dicts to.
    '''

    required_keys = ['polys', 'average_polygon', 'slice_nos', 'predicted_label']
    check_no_keys_missing(vert_dicts, required_keys)

    with open(filename, 'w') as f:
        w = DictWriter(f, fieldnames=required_keys)
        w.writeheader()
        for vert_dict in vert_dicts:
            w.writerow(vert_dict)
    return


def check_no_keys_missing(vert_dicts, required_keys):
    for vert_dict in vert_dicts:
        for key in required_keys:
            if key not in vert_dict:
                raise KeyError(f'{key} is missing from vert_dict')
    
