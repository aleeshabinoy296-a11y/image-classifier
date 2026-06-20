import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="Image Classifier", page_icon="🖼️", layout="centered")

st.title("Image Classifier")
st.write("Upload any photo and the AI will tell you what it thinks is in it.")


@st.cache_resource
def load_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")


classifier = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your uploaded image", use_container_width=True)

    if st.button("Classify image"):
        with st.spinner("Analysing image..."):
            results = classifier(image)

        st.subheader("Top 5 predictions")

        for i, result in enumerate(results[:5], start=1):
            label_clean = result["label"].replace("_", " ").title()
            confidence = result["score"]

            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{i}. {label_clean}")
                st.progress(float(confidence))
            with col2:
                st.write(f"{confidence * 100:.1f}%")

        top_label = results[0]["label"].replace("_", " ").title()
        top_confidence = results[0]["score"] * 100
        st.success(f"Best guess: **{top_label}** ({top_confidence:.1f}% confident)")

st.divider()
st.caption("Built with Streamlit and a Vision Transformer (ViT) model | A beginner computer vision project")
st.caption("Model: google/vit-base-patch16-224, pre-trained on ImageNet (1000 everyday object categories)")