# Hyperparameters

All experiments were run with the following common hyperparameters:
  - optimizer: 
    - AdamW with default betas and eps set in transformers library
    - warmup_proportion: 0.0
    - weight_decay: 0.0
    - gradient_accumulation_steps: 1
  - validation frequency: We perform validation on the validation set after every training epoch. 
      The model with the best validation score (micro-f1) is used to report results on the test set.
  - seeds: All scores are reported as a mean over runs initialed by 4 different seeds. 
           We used {42, 1234 1729, 7777} as the set of seeds for all our experiments. 

The hyperparameters which we did vary across tasks and models were:  
  - learning_rate
  - num_train_epochs
  - max_seq_length
  - train_batch_size 

### Hyperparameters for xlm-roberta-large

| Task           | learning_rate | num_train_epochs| max_seq_length | train_batch_size |  
| -------------- | ------------- | --------------- | -------------- | ---------------- |
| Abstract       | 5e-5          | 5               | 128            | 16               |
| Better Phase-1 | 5e-5          | 50              | 128            | 16               |
| Better Phase-2 | 5e-5          | 50              | 128            | 16               | 
| ACE            | 5e-6          | 20              | 128            | 16               | 
| MINION         | 5e-6          | 20              | 256            | 16               | 

### Hyperparameters for lanwuwei/GigaBERT-v4-Arabic-and-English

| Task           | learning_rate | num_train_epochs| max_seq_length | train_batch_size |  
| -------------- | ------------- | --------------- | -------------- | ---------------- |
| Abstract       | 5e-5          | 5               | 128            | 32               |
