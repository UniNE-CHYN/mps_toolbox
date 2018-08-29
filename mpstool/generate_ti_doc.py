#!/usr/bin/env python3

import mpstool.ti
import os


def build_doc(path):
    print("Im generating rst doc")
    ti_base = mpstool.ti.build_TrainingImageBase()
    with open(os.path.join(path, 'ti_list.inc'), 'w') as ti_list_file:
        print("opened " + os.path.join(path, 'ti_list.inc') )
        ti_list_file.write(".. toctree::" + "\n" + "\n")
        ti_base = ti_base.as_dict()
        for key in ti_base:
            ti_list_file.write('   ' + key + '\n')
            with open(os.path.join(path, filename(key)), 'w') as ti_file:
                print("opened " +  os.path.join(path, filename(key)))
                ti_file.write(key + '\n')
                ti_file.write('========================' + '\n' + '\n')
                ti_file.write(
                    '.. image:: ' + os.path.relpath(ti_base[key].info['image'])
                    + '\n' + '\n')
                ti_file.write(ti_base[key].info['type'] + '\n')


def filename(name):
    return name + '.rst'


if __name__ == '__main__':
    build_doc()
