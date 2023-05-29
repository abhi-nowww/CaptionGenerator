# CaptionGenerator
This is a Python script that generates descriptions and captions for images using the Salesforce BLiP Image Captioning model and OpenAI's GPT-3 language model. The script provides two options for generating captions: uploading custom images or using pre-defined sample images.

# Requirements
To run this script, you need to have the following dependencies installed:
1.streamlit
2.transformers
3.openai
4.tqdm
5.PIL
6.torch

# Usage
To use the image caption generator, run the script and open the generated Streamlit web application in your browser. The application provides two tabs: "Upload Images" and "Sample".

# Upload Images
In the "Upload Images" tab, you can upload custom images from your local machine. The script accepts images in JPG, PNG, and JPEG formats. Once you have uploaded the images, click the "Generate" button to generate descriptions and captions for the images. The descriptions will be displayed, and three creative captions will be generated using OpenAI's GPT-3 model.

# Sample
In the "Sample" tab, the script provides three pre-defined sample images. Clicking the "Generate" button below each sample image will generate descriptions and captions in the same way as described above.

# Acknowledgments
The Salesforce BLiP Image Captioning model is used for generating descriptions of the images.
OpenAI's GPT-3 language model is used for generating creative captions based on the image descriptions.
