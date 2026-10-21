Final held-out test comparison:
                                  Accuracy  Balanced Acc  Precision  Recall      F1  ROC-AUC  PR-AUC
System                                                                                              
A - TF-IDF + Logistic Regression    0.9026        0.9026     0.9021  0.9032  0.9027   0.9654  0.9636
B - Frozen distilgpt2 + MLP         0.8280        0.8280     0.8129  0.8522  0.8321   0.9115  0.9114
C - LoRA fine-tuned GPT             0.9026        0.9026     0.8964  0.9105  0.9034   0.9660  0.9650

Highest held-out F1: C - LoRA fine-tuned GPT. This makes the evaluation honest: the most sophisticated system is not automatically the best-performing one.

Bootstrap confidence intervals:
                          System        F1 95% CI   ROC-AUC 95% CI
A - TF-IDF + Logistic Regression (0.8989, 0.9067) (0.9636, 0.9674)
     B - Frozen distilgpt2 + MLP  (0.827, 0.8371) (0.9079, 0.9151)
         C - LoRA fine-tuned GPT (0.8997, 0.9077) (0.9644, 0.9679)

Length-bucket analysis:
System Length bucket     N  Accuracy     F1  ROC-AUC
     A         <=100  3139    0.9057 0.9131   0.9661
     A       101-200 11783    0.9049 0.9046   0.9662
     A       201-400  6947    0.8998 0.8952   0.9639
     A       401-800  2713    0.8953 0.8975   0.9633
     A          >800   418    0.9091 0.9156   0.9717
     B         <=100  3139    0.8261 0.8432   0.9096
     B       101-200 11783    0.8402 0.8415   0.9222
     B       201-400  6947    0.8277 0.8243   0.9088
     B       401-800  2713    0.7866 0.8033   0.8742
     B          >800   418    0.7751 0.8050   0.8551
     C         <=100  3139    0.9169 0.9235   0.9717
     C       101-200 11783    0.9172 0.9171   0.9749
     C       201-400  6947    0.8889 0.8856   0.9550
     C       401-800  2713    0.8629 0.8678   0.9436
     C          >800   418    0.8732 0.8820   0.9421

Error summary:
                               Statistic  Value
                         False positives 1315.0
                         False negatives 1119.0
      Mean length of correct predictions  225.0
Mean length of misclassified predictions  260.9