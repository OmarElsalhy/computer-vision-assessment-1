# Computer Vision – Assessment 1

Two small computer vision tasks done with TensorFlow/Keras and OpenCV.

## Repository structure

```
.
├── README.md
├── task1-image-classification/
│   ├── task image classification.ipynb
│   ├── test dog.jpeg
│   └── test cat.jpeg
└── task2-object-counting/
    ├── object Counting.ipynb
    └── test coins.jpeg
```

---

## Task 1 – Cat vs Dog Image Classification

**Goal:** Classify an image as a cat or a dog.

**Dataset:** [Microsoft Cats vs Dogs (Kaggle)](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset) – not included in this repo because of its size. Download it and extract it so the `PetImages` folder sits next to the notebook (`task1-image-classification/PetImages`).

**Approach**
- Images resized to 150×150 and rescaled to [0, 1]
- 80/20 train/validation split (20,000 train / 4,998 validation images)
- Simple CNN: 3 × (Conv2D + MaxPooling) → Flatten → Dense(128) → Dense(1, sigmoid)
- Adam optimizer, binary cross-entropy, 10 epochs

**Results**

| Metric | Final epoch |
|---|---|
| Training accuracy | ~99.0% |
| Validation accuracy | ~81.8% |

The gap between training and validation accuracy shows the model **overfits**. Data augmentation, dropout, or transfer learning would likely improve generalization.

**Prediction:** the notebook includes a `predict_and_show()` helper that is run on `test dog.jpeg` and `test cat.jpeg`.

---

## Task 2 – Object Counting (Coins)

**Goal:** Count the coins in an image.

**Approach**
- Convert to grayscale and apply a median blur (kernel 7)
- Detect circles with the Hough Circle Transform (`cv2.HoughCircles`)
- Keep only circles fully inside the image (ignore ones cut off by the border)
- Draw the detected circles and print the count

**Result:** 4 full circles detected on `test coins.jpeg`.

**Limitation:** Hough parameters (`param2`, `minRadius`, `maxRadius`, ...) are tuned by hand for this image and may need adjusting for others.

---

## How to run

1. Clone the repo
   ```bash
   git clone https://github.com/OmarElsalhy/computer-vision-assessment-1.git
   cd computer-vision-assessment-1
   ```
2. Install dependencies
   ```bash
   pip install tensorflow numpy matplotlib opencv-python jupyter
   ```
3. Open the notebooks with Jupyter and run them from inside their own folder.

## Tools
Python · TensorFlow/Keras · OpenCV · NumPy · Matplotlib
