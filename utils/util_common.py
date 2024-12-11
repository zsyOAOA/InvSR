#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-02-06 10:34:59

import os
import random
import requests
import importlib
from pathlib import Path
from PIL import Image

def mkdir(dir_path, delete=False, parents=True):
    import shutil
    if not isinstance(dir_path, Path):
        dir_path = Path(dir_path)
    if delete:
        if dir_path.exists():
            shutil.rmtree(str(dir_path))
    if not dir_path.exists():
        dir_path.mkdir(parents=parents)

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def get_filenames(dir_path, exts=['png', 'jpg'], recursive=True):
    '''
    Get the file paths in the given folder.
    param exts: list, e.g., ['png',]
    return: list
    '''
    if not isinstance(dir_path, Path):
        dir_path = Path(dir_path)

    file_paths = []
    for current_ext in exts:
        if recursive:
            file_paths.extend([str(x) for x in dir_path.glob('**/*.'+current_ext)])
        else:
            file_paths.extend([str(x) for x in dir_path.glob('*.'+current_ext)])

    return file_paths

def readline_txt(txt_file):
    txt_file = [txt_file, ] if isinstance(txt_file, str) else txt_file
    out = []
    for txt_file_current in txt_file:
        with open(txt_file_current, 'r') as ff:
            out.extend([x[:-1] for x in ff.readlines()])

    return out

def scan_files_from_folder(dir_paths, exts, recursive=True):
    '''
    Scaning images from given folder.
    Input:
        dir_pathas: str or list.
        exts: list
    '''
    exts = [exts, ] if isinstance(exts, str) else exts
    dir_paths = [dir_paths, ] if isinstance(dir_paths, str) else dir_paths

    file_paths = []
    for current_dir in dir_paths:
        current_dir = Path(current_dir) if not isinstance(current_dir, Path) else current_dir
        for current_ext in exts:
            if recursive:
                search_flag = f"**/*.{current_ext}"
            else:
                search_flag = f"*.{current_ext}"
            file_paths.extend(sorted([str(x) for x in Path(current_dir).glob(search_flag)]))

    return file_paths

def write_path_to_txt(
        dir_folder,
        txt_path,
        search_key,
        num_files=None,
        write_only_name=False,
        write_only_stem=False,
        shuffle=False,
        ):
    '''
    Scaning the files in the given folder and write them into a txt file
    Input:
        dir_folder: path of the target folder
        txt_path: path to save the txt file
        search_key: e.g., '*.png'
        write_only_name: bool, only record the file names (including extension),
        write_only_stem: bool, only record the file names (not including extension),
    '''
    txt_path = Path(txt_path) if not isinstance(txt_path, Path) else txt_path
    dir_folder = Path(dir_folder) if not isinstance(dir_folder, Path) else dir_folder
    if txt_path.exists():
        txt_path.unlink()
    if write_only_name:
        path_list = sorted([str(x.name) for x in dir_folder.glob(search_key)])
    elif write_only_stem:
        path_list = sorted([str(x.stem) for x in dir_folder.glob(search_key)])
    else:
        path_list = sorted([str(x) for x in dir_folder.glob(search_key)])
    if shuffle:
        random.shuffle(path_list)
    if num_files is not None:
        path_list = path_list[:num_files]
    with open(txt_path, mode='w') as ff:
        for line in path_list:
            ff.write(line+'\n')

def download_image_from_url(url, dir="./"):
    # Download a file from a given URI, including minimal checks

    # Download
    f = str(Path(dir) / os.path.basename(url))  # filename
    try:
        with open(f, "wb") as file:
            file.write(requests.get(url, timeout=10).content)
    except:
        print(f'Skip the url: {f}!')

    # Rename (remove wildcard characters)
    src = f  # original name
    for c in ["%20", "%", "*", "~", "(", ")"]:
        f = f.replace(c, "_")
    f = f[: f.index("?")] if "?" in f else f  # new name
    if src != f:
        os.rename(src, f)  # rename

    # Add suffix (if missing)
    if Path(f).suffix == "":
        src = f  # original name
        try:
            f += f".{Image.open(f).format.lower()}"
            os.rename(src, f)  # rename
        except:
            Path(f).unlink()
