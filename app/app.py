"""
app/app.py

Gradio demo app cho Flood Detection project.

Cach chay:
    python app/app.py
"""

import os
import sys

import gradio as gr
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.infer import FloodPredictor

# Load model 1 lan khi khoi dong app
predictor = FloodPredictor(
    checkpoint_path="checkpoints/best_model.pt",
    tfidf_path="checkpoints/tfidf_vectorizer.pkl",
)


def predict(image, text):
    if image is None:
        return "Vui long upload anh.", None

    pil_image = Image.fromarray(image)
    result = predictor.predict(pil_image, text or "")

    label = result["label"]
    conf  = result["confidence"]
    probs = result["probs"]

    # Format output
    emoji = "🌊" if label == "Flood" else "✅"
    summary = f"{emoji} **{label}** ({conf:.1%} confidence)"

    bar = {
        "No Flood": probs["No Flood"],
        "Flood":    probs["Flood"],
    }

    return summary, bar


# --- UI ---
with gr.Blocks(title="Flood Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌊 Flood Detection
    **Multimodal image classifier** — ConvNeXt-Tiny + TF-IDF text branch  
    Upload an image and (optionally) add a description or tags to improve accuracy.
    
    *Test accuracy: **94.6%** | F1: **92.8%** on MediaEval flood dataset*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Image", type="numpy")
            text_input  = gr.Textbox(
                label="Description / Tags (optional)",
                placeholder="e.g. flood disaster water submerged street...",
                lines=2,
            )
            submit_btn = gr.Button("Predict", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Markdown(label="Prediction")
            prob_output  = gr.Label(label="Confidence", num_top_classes=2)

    submit_btn.click(
        fn=predict,
        inputs=[image_input, text_input],
        outputs=[label_output, prob_output],
    )

    gr.Markdown("""
    ---
    **Model**: ConvNeXt-Tiny (pretrained ImageNet-12k, fine-tuned) + TF-IDF text branch  
    **Dataset**: [MediaEval Flood-related Multimedia](http://www.multimediaeval.org/) — 5,280 images  
    **Training**: RTX 2050 4GB VRAM, ~15 epochs, mixed precision (FP16)
    """)


if __name__ == "__main__":
    demo.launch(share=True)