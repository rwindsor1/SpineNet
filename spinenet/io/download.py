import os
import warnings

import requests
from tqdm import tqdm


def download(url: str, fname: str, verbose: bool = True) -> None:
    '''
    Download a file from a url and save it to a file.

    Parameters
    ----------
    url : str
        The url to download the file from.
    fname : str
        The path to save the file to.
    verbose : bool, optional
        Whether to print progress information. The default is True.
    '''
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    # Can also replace 'file' with a io.BytesIO object
    if verbose:
        print(f"Downloading {fname} from {url}...")
        with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    else:
        with open(fname, 'wb') as file:
            for data in resp.iter_content(chunk_size=1024):
                file.write(data)
    return



def download_weights(weights_path: str, weight_urls_dict: dict, force: bool = False, verbose: bool = True) -> None:
    '''
    Download weights from the urls in weight_urls_dict to the weights_path.

    Parameters
    ----------
    weights_path : str
        The path to all weights to.
    weight_urls_dict : dict
        The dictionary of urls to download weights from. The key are the filepaths of the weights, and the values are the urls to download the weights from.
    force : bool, optional
        Whether to force download weights even if the weights already exist. The default is False.
    verbose : bool, optional
        Whether to print progress information. The default is True.
    '''
    print('-' * 80)
    print('Downloading weights...')
    print("Note: By downloading these weights and/or adapting SpineNet's source code, you agree to the licence which can be found here: https://github.com/rwindsor1/SpineNet/blob/main/LICENCE.md")
    print('SpineNet is not a diagnostics tool nor a medical device. It should only be used for research.')
    print('-' * 80)
    for path, url in weight_urls_dict.items():
        fname = os.path.join(weights_path, path)
        if not os.path.exists(fname) or force:
            download(url, fname, verbose=verbose)
        else:
            warnings.warn('{} already exists. Skipping download.'.format(fname))
            return
