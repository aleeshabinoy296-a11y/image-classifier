# Image Classifier

A web app that identifies what's in a photo using a pre-trained Vision Transformer (ViT) model.

## Demo

![App screenshot](screenshot.png)

Live app: [Add your deployed Streamlit link here once deployed]

## What it does

- Upload any photo (JPG or PNG)
- The app identifies what's in the image using a Vision Transformer trained on ImageNet (1000 everyday object categories)
- Shows the top 5 predictions with confidence percentages

## Tech stack

- Python
- Streamlit — for the web interface
- PyTorch — deep learning framework
- Hugging Face Transformers — for loading the pre-trained Vision Transformer model
- Pillow (PIL) — for image handling

## How to run this locally

1. Clone this repository
   ```
   git clone https://github.com/YOUR-USERNAME/image-classifier.git
   cd image-classifier
   ```

2. Install the required packages
   ```
   pip install -r requirements.txt
   ```

3. Run the app
   ```
   streamlit run app.py
   ```

4. The first time you run it, the model will download automatically (a few hundred MB). This only happens once and is then cached locally.

5. Your browser will open at `http://localhost:8501`

## What I learned building this

- How to use a pre-trained Vision Transformer (ViT) for image classification via Hugging Face's `pipeline` API
- The difference between CNNs (older approach) and Transformers (modern approach) for computer vision
- Image preprocessing and confidence scoring
- Caching a model in Streamlit with `@st.cache_resource` so it doesn't reload on every interaction

## Limitations

This model is trained on ImageNet, which contains real-world photos across 1000 everyday categories (animals, vehicles, food, household objects, etc). It has never seen anime, cartoons, illustrations, or fictional characters, so it will misclassify these by matching the closest visual patterns it does know (e.g. shapes, colours, textures). It works best and most accurately on real photographs of everyday objects.

**Why this limitation exists, and why it wasn't "fixed":**

- **Training data scope** — ImageNet was built from real-world photographs only, so the model has no learned concept of illustrated, animated, or stylised content. This isn't a bug to patch, it's a direct consequence of what the model was shown during training. Expanding it to recognise anime or cartoon imagery would require retraining on a different dataset entirely.
- **Identity vs object recognition** — this model classifies generic object categories (e.g. "dog," "car," "umbrella"), not specific named people or characters. It was deliberately kept this way rather than extended into facial or character recognition, since identifying specific real people from photos is a facial recognition capability with genuine privacy and safety risks (e.g. enabling stalking or unauthorised surveillance), and identifying specific fictional characters from copyrighted media risks reproducing protected artwork.
- **A known limitation is more useful than a hidden one** — rather than presenting the model as universally accurate, this README documents exactly where it breaks down, which is good practice for any deployed ML system.

## Possible improvements

- Let users upload multiple images and classify them in a batch
- Try a different model from Hugging Face's model hub and compare accuracy
- Add a confidence threshold warning when the model isn't sure about its prediction
- Deploy as a mobile-friendly app