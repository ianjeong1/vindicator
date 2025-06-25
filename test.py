from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# Load the processor and model
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base", 
    trust_remote_code=True,
    use_safetensors=True
)

# Load an image from a URL
url="https://t4.ftcdn.net/jpg/02/24/86/95/360_F_224869519_aRaeLneqALfPNBzg0xxMZXghtvBXkfIA.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Preprocess the image
inputs = processor(images=image, return_tensors="pt")

# Generate a caption
output = model.generate(**inputs)

# Decode the output
caption = processor.decode(output[0], skip_special_tokens=True)
print("Generated Caption:", caption)