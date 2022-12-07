from epioncho_ibm.utils import (
    BlockBinomialGenerator
)
import numpy as np
test = np.array([[1,3,0],[0,3,3]])
gen = BlockBinomialGenerator(
    prob = 0.5
)

gen.generate_row(2)
gen.generate_row(1)
out = gen(test)
print(out)
