#!/usr/bin/env python3

import mpstool.ti


print("read base")
ti_base = mpstool.ti.build_TrainingImageBase()
print("The complete base: ", ti_base.as_dict())


print(" ")
print(" selection:")
subset_ti = ti_base.get_subset('type', 'categorical')
print("Only categorical images: ", subset_ti.as_dict())
