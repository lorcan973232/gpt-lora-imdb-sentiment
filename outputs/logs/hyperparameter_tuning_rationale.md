Staged tuning decisions:
              stage         axis selected_value  best_F1  best_ROC_AUC  train_time_s
      Design search  max_seq_len            128   0.8296        0.9143          39.9
      Design search      pooling           last   0.8296        0.9143          39.7
      Design search   trunc_side      head_tail   0.8523        0.9304          40.2
Optimisation search           lr         0.0005   0.8527        0.9315          39.8
Optimisation search weight_decay           0.01   0.8527        0.9315          40.0
Optimisation search       lora_r              4   0.8598        0.9370          39.7
Optimisation search lora_dropout           0.05   0.8598        0.9370          40.0
 Final confirmation       epochs              3   0.8640        0.9374          59.6

All recorded tuning runs:
        axis     value  Train time (s)  Trainable params  Accuracy  Balanced Acc  Precision  Recall     F1  ROC-AUC  PR-AUC  val_loss  train_loss  epoch
 max_seq_len       128            39.9            407040     0.834         0.834     0.8523   0.808 0.8296   0.9143  0.9124    0.3795      0.4248      2
 max_seq_len       256            73.8            407040     0.829         0.829     0.9273   0.714 0.8068   0.9334  0.9375    0.4250      0.3764      2
     pooling      last            39.7            407040     0.834         0.834     0.8523   0.808 0.8296   0.9143  0.9124    0.3795      0.4248      2
     pooling      mean            39.5            407040     0.816         0.816     0.7852   0.870 0.8254   0.8934  0.8896    0.4241      0.9792      1
  trunc_side head_tail            40.2            407040     0.860         0.860     0.9018   0.808 0.8523   0.9304  0.9350    0.3575      0.3684      2
  trunc_side      tail            40.1            407040     0.834         0.834     0.8523   0.808 0.8296   0.9143  0.9124    0.3795      0.4248      2
          lr    0.0005            39.8            407040     0.858         0.858     0.8858   0.822 0.8527   0.9315  0.9375    0.3590      0.3573      2
          lr    0.0003            40.1            407040     0.860         0.860     0.9018   0.808 0.8523   0.9304  0.9350    0.3575      0.3684      2
          lr    0.0001            40.5            407040     0.839         0.839     0.8758   0.790 0.8307   0.9022  0.9097    0.4043      0.4634      2
weight_decay      0.01            40.0            407040     0.858         0.858     0.8858   0.822 0.8527   0.9315  0.9375    0.3590      0.3573      2
weight_decay       0.0            40.0            407040     0.849         0.849     0.8802   0.808 0.8425   0.9294  0.9350    0.3638      0.3558      2
      lora_r         4            39.7            204288     0.864         0.864     0.8872   0.834 0.8598   0.9370  0.9415    0.3259      0.3523      2
      lora_r        16            40.0            812544     0.852         0.852     0.8411   0.868 0.8543   0.9343  0.9381    0.3965      0.3634      2
      lora_r         8            40.5            407040     0.858         0.858     0.8874   0.820 0.8524   0.9315  0.9375    0.3593      0.3574      2
lora_dropout      0.05            40.0            204288     0.864         0.864     0.8872   0.834 0.8598   0.9370  0.9415    0.3259      0.3523      2
lora_dropout       0.0            39.3            204288     0.851         0.851     0.8305   0.882 0.8555   0.9313  0.9370    0.3658      0.3636      2
      epochs         3            59.6            204288     0.863         0.863     0.8580   0.870 0.8640   0.9374  0.9422    0.3769      0.3053      3
      epochs         2            39.6            204288     0.864         0.864     0.8872   0.834 0.8598   0.9370  0.9415    0.3259      0.3523      2

Selected final configuration:
model_name  max_seq_len     lr  batch_size  epochs  weight_decay  warmup_ratio  grad_accum_steps  lora_r  lora_alpha  lora_dropout pooling trunc_side
distilgpt2          128 0.0005           8       3          0.01          0.06                 2       4          16          0.05    last  head_tail