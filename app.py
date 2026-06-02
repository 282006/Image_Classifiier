import gradio as gr
import os
import time
from core.predict import ImageClassifier
from PIL import Image

cwd = os.getcwd()
model_path = os.path.join(cwd, "model", "cnn_128_model-100.pth")
classifier = ImageClassifier(model_path=model_path, class_name=None)

LABEL_EMOJI = {
    "CAT":    "🐱",
    "DOG":    "🐶",
    "Person": "🧑",
}

LABEL_COLOR = {
    "CAT":    "#f59e0b",
    "DOG":    "#3b82f6",
    "Person": "#10b981",
}


def classify(image):
    if image is None:
        raise gr.Error("Please upload an image first.")

    image = image.convert("RGB")

    image_path = "uploaded_img.jpg"
    image.save(image_path)

    start = time.time()
    label, output_path = classifier.predict(image_path)
    elapsed = round((time.time() - start) * 1000)

    emoji = LABEL_EMOJI.get(label, "❓")
    color = LABEL_COLOR.get(label, "#6366f1")

    label_html = f"""
    <div style="
        font-family: 'DM Sans', sans-serif;
        background: #12152b;
        border: 1px solid {color}55;
        border-left: 4px solid {color};
        border-radius: 10px;
        padding: 18px 22px;
        display: flex;
        align-items: center;
        gap: 14px;
    ">
        <span style="font-size: 38px;">{emoji}</span>
        <div>
            <div style="font-size: 11px; color: #8891b4; text-transform: uppercase; letter-spacing: 0.1em;">Detected</div>
            <div style="font-size: 28px; font-weight: 700; color: {color};">{label}</div>
            <div style="font-size: 12px; color: #8891b4; margin-top: 2px;">inference: {elapsed} ms</div>
        </div>
    </div>
    """

    return label_html, Image.open(output_path)


def clear_outputs():
    return None, None


css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: radial-gradient(ellipse at 60% 10%, #1a1f3a 0%, #0d0f1c 55%, #050608 100%) !important;
    font-family: 'DM Sans', sans-serif !important;
    min-height: 100vh;
}

.gr-prose h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #ffffff, #a5b4fc);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    letter-spacing: -0.02em;
}
.gr-prose p { color: #8891b4 !important; font-size: 0.95rem !important; }

.gr-panel, .gr-box {
    background: #0e1120 !important;
    border: 1px solid #1e2340 !important;
    border-radius: 14px !important;
}

.gr-image-upload {
    border: 2px dashed #2a2f52 !important;
    border-radius: 12px !important;
    background: #0b0d1a !important;
    transition: border-color 0.25s;
}
.gr-image-upload:hover { border-color: #4f46e5 !important; }

button.primary, #predict-btn {
    background: linear-gradient(135deg, #6366f1, #818cf8) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.01em;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s !important;
    box-shadow: 0 4px 18px rgba(99, 102, 241, 0.40) !important;
}
button.primary:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
button.primary:active { transform: scale(0.98) !important; }

button.secondary {
    background: transparent !important;
    border: 1px solid #2a2f52 !important;
    color: #8891b4 !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s, color 0.2s !important;
}
button.secondary:hover { border-color: #6366f1 !important; color: #a5b4fc !important; }

label span, .gr-form label { color: #8891b4 !important; font-size: 0.82rem !important; letter-spacing: 0.05em; }

.output-image img { border-radius: 10px !important; }

.gr-accordion { background: #0e1120 !important; border: 1px solid #1e2340 !important; border-radius: 10px !important; }

footer { display: none !important; }
"""

with gr.Blocks(css=css, title="Image Classifier") as demo:

    gr.Markdown(
        """
# 🔍 Image Classifier
Upload a photo and the model will detect whether it shows a **Cat**, **Dog**, or **Person**.
        """
    )

    with gr.Row():

        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Input Image",
                elem_classes=["gr-image-upload"],
            )

            with gr.Row():
                predict_btn = gr.Button("✨ Classify", variant="primary", elem_id="predict-btn")
                clear_btn   = gr.Button("Clear", variant="secondary")

            gr.Markdown("### 🗃️ Try a sample")
            

        with gr.Column(scale=1):
            label_output = gr.HTML(label="Prediction")
            image_output = gr.Image(label="Annotated Image", elem_classes=["output-image"])

    with gr.Accordion("ℹ️ How it works", open=False):
        gr.Markdown(
            """
**Model**: Custom CNN trained on 128×128 images across 3 classes — Cat, Dog, Person.

**Architecture**: 4 convolutional blocks (Conv2d → BatchNorm → ReLU → MaxPool) followed by
3 fully-connected layers (512 → 128 → 3).

**Pipeline**:
1. Image resized to 128×128 and normalised
2. Forward pass through the CNN
3. `argmax` over logits → predicted class
4. OpenCV overlays the label on the original image
            """
        )

    predict_btn.click(
        fn=classify,
        inputs=image_input,
        outputs=[label_output, image_output],
    )

    clear_btn.click(
        fn=clear_outputs,
        inputs=None,
        outputs=[label_output, image_output],
    )

if __name__ == "__main__":
    demo.launch()