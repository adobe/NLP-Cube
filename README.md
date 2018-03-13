# NLP-Cube #


# Current status #
* we treat words and character embeddings in a similar fashion 
* we tested with character encodings only (feature cutoff is set at 100)

# ToDO #
- [ ] provide training examples
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

