"""
Modify the directories of data in the text files.
into "/data1/hanzhongyi/datasets/"
"""
import glob
import os
from types import new_class

txt_file_dir = '/home/hanzhongyi/projects/RDA/data/*/*.txt'
txt_files = glob.glob(txt_file_dir)

old_str = '/home/zhongyi'
new_str = '/data1/hanzhongyi'

def alter(file, old_str, new_str):
    """
    instead string of file
    :param file: file name
    :param old_str: old string
    :param new_str: new string
    :return:
    """
    lines = []
    with open(file, 'r') as f:
        for line in f.read().splitlines():
            if old_str in line:
                line = line.replace(old_str, new_str)
            lines.append(line)
    with open(file, 'w') as f:
        for line in lines:
            f.write('{}\n'.format(line))
    
for txt in txt_files:
    alter(txt, old_str, new_str)