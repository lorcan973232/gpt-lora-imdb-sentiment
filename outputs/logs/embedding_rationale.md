Tokenisation coverage summary:
                Statistic  Value
    Mean token/word ratio  1.278
        Median BPE length    216
      90th pct BPE length    581
      95th pct BPE length    770
Reviews <= 512 BPE tokens  87.2%

Probe-term cosine similarity matrix:
           excellent  brilliant  amazing  enjoyable  terrible  awful  boring  dreadful
excellent      0.618      0.223    0.268      0.237     0.258  0.258   0.224     0.262
brilliant      0.223      0.485    0.259      0.236     0.244  0.228   0.253     0.239
amazing        0.268      0.259    0.599      0.241     0.224  0.271   0.262     0.237
enjoyable      0.237      0.236    0.241      0.482     0.225  0.270   0.240     0.260
terrible       0.258      0.244    0.224      0.225     0.597  0.244   0.245     0.251
awful          0.258      0.228    0.271      0.270     0.244  0.612   0.258     0.362
boring         0.224      0.253    0.262      0.240     0.245  0.258   0.585     0.275
dreadful       0.262      0.239    0.237      0.260     0.251  0.362   0.275     0.492

Interpretation: token embeddings alone are insufficient for document-level sentiment because negation, reversal, discourse structure, and late verdicts require contextualisation. That motivates LoRA adaptation rather than frozen lexical geometry alone.