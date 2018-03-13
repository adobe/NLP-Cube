# NLP-Cube #


# Current status #
* we treat words and character embeddings in a similar fashion 
* we tested with character encodings only (feature cutoff is set at 100)

# ToDO #
- [x] add word embeddings
- [x] find a good network achitecture for POS tagging
- [x] prepare a neural/based language pipeline
- [ ] pre-train models using universal dependencies
- [x] add a parser

# Parser architecture #
```
#   -----------------                    -------------------------- 
#   |word emebddings|----          ------|morphological embeddings|
#   -----------------    |        |      --------------------------
#                        |        |
#                      --------------
#                      |concatenate |
#                      --------------
#                             |
#                     ----------------
#                     |bdlstm_1_layer|
#                     ----------------
#                             |
#                     ----------------                  
#                     |bdlstm_2_layer| 
#                     ----------------                    
#                             |-----------------------------------------------------------------                          
#                     ----------------                                                         |
#                     |bdlstm_3_layer|                                                         |
#                     ----------------                                                         |
#                             |                                                                |
#        ---------------------------------------------                    ---------------------------------------------              
#        |           |                |              |                    |           |                |              |
#        |           |                |              |                    |           |                |              |
#    ---------  -----------       ----------    ------------          ---------  -----------       ----------    ------------
#    |to_link|  |from_link|       |to_label|    |from_label|          |to_link|  |from_link|       |to_label|    |from_label|
#    ---------  -----------       ----------    ------------          ---------  -----------       ----------    ------------
#         |        |                      |       |                       |           |                  |            |
#       --------------                 ---------------                  ------------------            -------------------
#       |softmax link|                 |softmax label|                  |aux softmax link|            |aux softmax label|
#       --------------                 ---------------                  ------------------            -------------------
#
#

```

# Tagger architecture #

```
#   -----------------                    ---------------------- 
#   |word emebddings|----          ------|character embeddings|
#   -----------------    |        |      ----------------------
#                        |        |
#                      --------------
#                      |tanh_1_layer|
#                      --------------
#                             |
#                     ----------------
#                     |bdlstm_1_layer|
#                     ----------------
#                             |
#                      --------------                  
#                      |tanh_2_layer|-------------------
#                      --------------                   |
#                             |                         |
#                     ----------------         -------------------
#                     |bdlstm_2_layer|         |aux_softmax_layer|
#                     ----------------         -------------------
#                             |
#                      ---------------
#                      |softmax_layer|
#                      ---------------
#

```
# Output #
This output is generated for Romanian XPOS tagging using UD (currently we got 97.44% on the test set :D). The best results are obtained by Stanford (96.96)

```Found 364 unique features, 470 unique labels and 212897 word embeddings of size 64
Maximum sequence len is 1630
Train: 193156 examples in 8044 sequences
Dev: 17826 examples in 753 sequences
Creating model/xpos.last
Epoch 1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 evaluating (train loss=0 sacc=0.169816011934 tacc=0.912024477624 dev sacc=0.140770252324 tacc=0.906428811848)
Creating model/xpos.last
Creating model/xpos.bestTerr
Creating model/xpos.bestSerr
Epoch 2 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 evaluating (train loss=0 sacc=0.279960218797 tacc=0.936947337903 dev sacc=0.258964143426 tacc=0.931055761248)
Creating model/xpos.last
Creating model/xpos.bestTerr
Creating model/xpos.bestSerr
Epoch 3 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 evaluating (train loss=0 sacc=0.334037792143 tacc=0.947617469817 dev sacc=0.298804780876 tacc=0.942555817345)
Creating model/xpos.last
Creating model/xpos.bestTerr
Creating model/xpos.bestSerr
Epoch 4 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 evaluating (train loss=0 sacc=0.388736946793 tacc=0.955383213568 dev sacc=0.350597609562 tacc=0.952204644901)
Creating model/xpos.last
Creating model/xpos.bestTerr
Creating model/xpos.bestSerr
Epoch 5 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 evaluating (train loss=0 sacc=0.391098955743 tacc=0.956879413531 dev sacc=0.347941567065 tacc=0.953438797262)
Creating model/xpos.last
Creating model/xpos.bestTerr
Epoch 6 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 evaluating (train loss=0 sacc=0.437344604674 tacc=0.960995257719 dev sacc=0.394422310757 tacc=0.956860765174)
Creating model/xpos.last
Creating model/xpos.bestTerr
Creating model/xpos.bestSerr
Epoch 7 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 evaluating (train loss=0 sacc=0.423794132273 tacc=0.959669904119 dev sacc=0.403718459495 tacc=0.955402221474)
Creating model/xpos.last
Creating model/xpos.bestSerr
Epoch 8 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 evaluating (train loss=0 sacc=0.434982595724 tacc=0.961430139369 dev sacc=0.39176626826 tacc=0.95607539549)
Creating model/xpos.last
Epoch 9 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 evaluating (train loss=0 sacc=0.446171059175 tacc=0.962760670132 dev sacc=0.382470119522 tacc=0.958319308875)
Creating model/xpos.last
Creating model/xpos.bestTerr
Epoch 10 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 evaluating (train loss=0 sacc=0.45822973645 tacc=0.963966948995 dev sacc=0.414342629482 tacc=0.960226635252)
Creating model/xpos.last
Creating model/xpos.bestTerr
Creating model/xpos.bestSerr
Epoch 11 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 evaluating (train loss=0 sacc=0.49564893088 tacc=0.967544368283 dev sacc=0.444887118194 tacc=0.962919331314)
Creating model/xpos.last
Creating model/xpos.bestTerr
Creating model/xpos.bestSerr```
