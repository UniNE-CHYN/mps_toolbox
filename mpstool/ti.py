#!/usr/bin/env python3

from mpstool.img import Image
import yaml
import os


class TrainingImageBase:
    """
    Implementation of a database of trainig images
    """

    def __init__(self, base):
        self._base = base

    def get_TI(self, name):
        return self._base[name]

    def get_subset(self, field, value):
        new_base = {}
        for key in self._base:
            if self._base[key].info[field] == value:
                new_base[key] = self._base[key]
        return TrainingImageBase(new_base)

    def as_dict(self):
        return self._base


def build_TrainingImageBase():
    filename = __file__
    module_dir = os.path.dirname(filename)
    ti_dir = os.path.join(module_dir, 'ti')
    filenames = [x for x in os.walk(ti_dir)]
    directories = []
    files = get_yml_files(ti_dir)

    base = {}
    for yml_file, directory in files:
        with open(yml_file, 'r') as stream:
            try:
                ti_info = yaml.load(stream)
                ti_name = ti_info['name']
                ti_file = os.path.join(directory, ti_info['file'])
                ti_Image = Image.fromGslib(ti_file)
                ti = TI(ti_info, ti_Image)
                base[ti_name] = ti
            except yaml.YAMLError as exc:
                print(exc)
    return TrainingImageBase(base)


class TI:
    def __init__(self, info, image):
        self.info = info
        self.image = image


def get_yml_files(directory):
    files_list = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.endswith('.yml'):
                files_list.append((os.path.join(root, name), root))
    return files_list
