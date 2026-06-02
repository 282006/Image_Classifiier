# 🔍 Image Classifier

(https://huggingface.co/spaces/Vidhanjain28/img_classification_gradio) (Link to access)

A deep learning web app that classifies images as **Cat 🐱**, **Dog 🐶**, or **Person 🧑** — built with a custom CNN in PyTorch and served via a Gradio UI.

---

## Demo

> Upload any image → click **✨ Classify** → get an instant prediction with the label overlaid on the image.

---

## Project Structure

```
image-classifier/
│
├── app.py                  # Gradio UI entry point
├── requirements.txt        # Python dependencies
├── README.md
│
├── core/
│   ├── __init__.py
│   └── predict.py          # CNN model definition + ImageClassifier class
│
├── model/
│   └── cnn_128_model-100.pth   # Trained model weights (not tracked by git)
│
└── samples/                # Optional sample images for the UI
    ├── cat.jpg
    ├── dog.jpg
    └── person.jpg
```

---

## Model Architecture

A custom CNN trained on 128×128 RGB images across 3 classes.

```
Input (3 × 128 × 128)
   │
   ├─ Conv2d(3→32)  + BatchNorm + ReLU + MaxPool
   ├─ Conv2d(32→64) + BatchNorm + ReLU + MaxPool
   ├─ Conv2d(64→128)+ BatchNorm + ReLU + MaxPool
   ├─ Conv2d(128→256)+BatchNorm + ReLU + MaxPool
   │
Flatten
   │
   ├─ Linear(→512) + ReLU
   ├─ Linear(→128) + ReLU
   └─ Linear(→3)   → [CAT, DOG, Person]
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/image-classifier.git
cd image-classifier
```

### 2. Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your model weights

Place your trained `.pth` file at:

```
model/cnn_128_model-100.pth
```

> The model file is excluded from git via `.gitignore` due to its size.  
> Download link / instructions: *(add your own link here)*

### 5. Run the app

```bash
python app.py
```

Then open the link in your browser.

---

## Requirements

| Package | Purpose |
|---|---|
| `gradio` | Web UI |
| `torch` | Model inference |
| `torchvision` | Image transforms |
| `opencv-python` | Label overlay on image |
| `pillow` | Image loading |

---

## Usage

1. Upload a photo using the input panel (or click a sample image)
2. Click **✨ Classify**
3. The prediction card shows the detected class and inference time
4. The annotated image shows the label drawn directly on the photo
5. Click **Clear** to reset

---

## Known Limitations

- Only classifies into 3 classes: Cat, Dog, Person
- Input images are resized to 128×128 — very high-res images may lose detail
- Works best on clear, well-lit photos with a single subject

---

## License

MIT
