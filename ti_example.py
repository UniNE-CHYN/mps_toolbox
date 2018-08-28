#!/usr/bin/env python3

import mpstool.ti

ti_base = mpstool.ti.build_TrainingImageBase()
print("The complete base: ", ti_base.as_dict())

subset_ti = ti_base.get_subset('type', 'categorical')
print("Only categorical images: ", subset_ti.as_dict())
