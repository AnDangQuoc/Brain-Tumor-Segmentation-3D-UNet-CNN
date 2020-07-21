# Create Data For training

```bash
$ python create_and_save_cropped_images.py
```

# Shuffle train and validation data

```bash
$ python create_new_train_val_test_split.py
```

# Train model

```bash
$ python train.py
```

Based on the model you want to train answer the questions

`What model do you want to train?`

- first_model: full branch model
- second_model: segment only model

`How many branch do you want to train?`

- 1: train 1 branch
- 2: train 2 branch
- 3: train 3 branch
