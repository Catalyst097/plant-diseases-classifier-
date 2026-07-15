# Plant Disease Classifier

A MobileNetV2 transfer-learning image classifier that identifies plant
diseases from leaf photos across 38 classes spanning 14 crop species,
trained on the PlantVillage dataset.

## Approach

- **Backbone**: MobileNetV2 pretrained on ImageNet, frozen initially
- **Head**: `GlobalAveragePooling2D → Dropout(0.2) → Dense(128, relu) → Dense(38, softmax)`
- **Pipeline**: `tf.data` with on-the-fly augmentation (flip, rotation, zoom)
  applied only to training data, never validation
- **Fine-tuning**: top ~55 layers of MobileNetV2 unfrozen after head training,
  retrained with a reduced learning rate (1e-5)

## Results

- **89% validation accuracy** after initial head training
- Strong performance on visually distinct classes (e.g. `Grape___healthy`,
  `Squash___Powdery_mildew`, both near 0.99 f1-score)
- **Key finding**: tomato disease classes were the model's weakest area
  (`Tomato___Early_blight` recall as low as 0.13, several others under 0.75
  f1-score). Confusion matrix analysis shows these classes are being
  confused with each other rather than with unrelated crop diseases —
  consistent with the fact that early blight, Septoria leaf spot, and
  target spot all present similarly on tomato leaves (brownish/dark
  lesions), making them genuinely hard to distinguish even visually.

## Dataset

[PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
(not included in this repo — downloaded via `kagglehub` at runtime)

## Requirements

```bash
pip install tensorflow kagglehub scikit-learn matplotlib seaborn numpy
```

## Usage

```bash
python train.py
```

## What I'd Improve With More Time

- Targeted data augmentation or class-weighting specifically for the
  weak tomato classes
- A second-stage classifier specialized only on tomato diseases, given
  how visually distinct they are from other crops but similar to each other
- Deploy as an interactive demo (Gradio / Hugging Face Spaces) for live testing
