# Problem choice justification

Research question:
How effective is parameter-efficient fine-tuning of decoder-only GPT models for long-form
IMDb sentiment classification, relative to a strong classical baseline and a weaker frozen-GPT
baseline, when truncation policy, embedding behaviour, model choice, and hyperparameters are
examined explicitly?

Why this is not a trivial classroom reproduction:
1. The task is framed as long-form decoder-only GPT adaptation, not generic IMDb sentiment.
2. The documents are long enough that context budget and truncation strategy become design variables.
3. Decoder-only GPTs are less natural classifiers than encoder models, so adaptation choices matter.
4. The study separates pretrained representation quality from task-specific adaptation.
5. The code includes baselines, candidate selection, tuning, calibration/error analysis, and
   review-length analysis rather than a single unexamined run.
