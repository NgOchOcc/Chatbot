import os
from pathlib import Path

root_path = Path(os.path.dirname(os.path.abspath(__file__))).parent


def join_root_path(path, is_dir=False):
    join_path = os.path.join(root_path, path)
    if is_dir and not os.path.exists(join_path):
        os.makedirs(join_path)
    return join_path

def join_path(src, dest, is_dir=False):
    join_path = os.path.join(src, dest)
    if is_dir and not os.path.exists(join_path):
        os.makedirs(join_path)
    return join_path