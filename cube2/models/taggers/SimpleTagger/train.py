import sys, subprocess, os
sys.path.append("../../../..")

import numpy as np
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from cube2.models.taggers.SimpleTagger.model import SimpleTagger
from cube2.components.lookup import Lookup, createLookup
from cube2.components.loaders.loaders import getSequenceDataLoader
from cube2.util.utils import pretty_sequences, use_gpu
from cube2.components.trainers.tagger import train

use_gpu() # auto-select GPU if available

print("\n\n\n")
#lookup = createLookup(["../../../../../ud-treebanks-v2.4/UD_Romanian-RRT/ro_rrt-ud-train.conllu"], verbose=True, minimum_word_frequency_cutoff=7)      
#lookup.save("../../../../scratch")
lookup = Lookup("../../../../scratch")


train_dataloader = getSequenceDataLoader(["../../../../../ud-treebanks-v2.4/UD_Romanian-RRT/ro_rrt-ud-train.conllu"], batch_size=32, lookup_object=lookup, num_workers=0, shuffle=True)
dev_dataloader = getSequenceDataLoader(["../../../../../ud-treebanks-v2.4/UD_Romanian-RRT/ro_rrt-ud-dev.conllu"], batch_size=256, lookup_object=lookup, num_workers=0, shuffle=False)
#test_dataloader = getSequenceDataLoader(["d:\\ud-treebanks-v2.4\\UD_Romanian-RRT\\ro_rrt-ud-test.conllu"], batch_size=2, lookup_object=lookup, num_workers=0, shuffle=True)
model = SimpleTagger(lookup)

#print(model)

#optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)#, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=0)

train(model, train_dataloader, dev_dataloader, None, optimizer, criterion, max_epochs=100000, patience=10, model_store_path = "../../../../scratch/SimpleTagger", resume_training = False)
