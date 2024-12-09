import sys
sys.path.append('./build')

import PyDepthInpaint
import numpy as np

d1 = PyDepthInpaint.DepthProcess(500,500)
array = np.random.randint(0, 65536, size=(500, 500), dtype=np.uint16)
print(d1.process(array))