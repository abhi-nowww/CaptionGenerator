import streamlit as st
from transformers import AutoProcessor, BlipForConditionalGeneration, AutoTokenizer
import openai
from itertools import cycle
from tqdm import tqdm
from PIL import Image
import torch

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

openai.api_key = "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"
openai_model = "text-davinci-002"


def caption_generator(des):
    caption_prompt = ('''Please Generate Three Unique And Creative Caption to use on instagram for a photo
    that shows '''+des+'''.The Caption should be Fun and creative.
    Captions:
    1.
    2.
    3.
    ''')
    response = openai.Completion.create(
        engine=openai_model,
        prompt = caption_prompt,
        max_tokens=(175 * 3),
        n=1,
        stop=None,
        temperature=0.7

    )
    caption = response.choices[0].text.strip().split("\n")
    return caption


def prediction(img_list):
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    img = []

    for images in tqdm(img_list):
        i_image = Image.open(images)
        st.image(i_image, width=200)

        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        img.append(i_image)
    pixel_val = processor(images=img, return_tensors="pt").pixel_values
    pixel_val = pixel_val.to(device)
    output = model.generate(pixel_val, **gen_kwargs)

    predict = tokenizer.batch_decode(output, skip_special_tokens=True)
    predict = [pred.strip() for pred in predict]

    return predict


def sample():
    sp_images = {'sample 1': "images/Image1.png",
                 'sample 2 ': "images/Image2.png",
                 'sample 3': "images/Image3.png"}
    colms = cycle(st.columns(3))

    for sp in sp_images.values():
        next(colms).image(sp, width=150)

    for i, sp in enumerate(sp_images.values()):
        if next(colms).button("Generate ", key=i):
            description = prediction([sp])
            st.subheader("Description of The Image :")
            st.write(description[0])

            st.subheader("Captions For The Image")
            captions = caption_generator(description[0])
            for caption in captions:
                st.write(caption)


def upload():
    with st.form("Uploader"):
        image = st.file_uploader("Upload Images ", accept_multiple_files=True, type=["jpg", "png", "jpeg"])
        submit = st.form_submit_button("Generate")

        if submit:
            description = prediction(image)

            st.subheader("Description of The Image :")
            for i, caption in enumerate(description):
                st.write(caption)

            st.subheader("Caption of the Images Are:")
            captions = caption_generator(description[0])
            for caption in captions:
                st.write(caption)


def main():
    st.set_page_config(page_title="Caption Generator")
    st.title("Description And Caption Generator")
    st.subheader("By Abhinav")

    tab1, tab2 = st.tabs(["Upload Images ", "Sample"])

    with tab1:
        upload()
    with tab2:
        sample()


if __name__ == "__main__":
    main()