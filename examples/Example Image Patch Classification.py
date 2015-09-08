# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from classy import *
from classy import image

# <codecell>

images=image.load_images('data/digits')

# <markdowncell>

# with overlapping patches...

# <codecell>

data=image.images_to_patch_vectors(images,(4,4))

# <markdowncell>

# or with non-overlapping patches...

# <codecell>

data=image.images_to_patch_vectors(images,(4,4),overlap=False)

# <markdowncell>

# then do classification...

# <codecell>

data_train,data_test=split(data)

# <codecell>

C=NaiveBayes()
timeit(reset=True)
C.fit(data_train.vectors,data_train.targets)
print "Training time: ",timeit()

# <codecell>

print "On Training Set:",C.percent_correct(data_train.vectors,data_train.targets)
print "On Test Set:",C.percent_correct(data_test.vectors,data_test.targets)

# <codecell>


