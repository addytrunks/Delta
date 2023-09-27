from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from PIL import Image

image_path = 'sky.webp'

# preprocessing images to make them suitable for the ResNet-50 model.
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

# pre-trained ResNet-50 image classification model we will use for predictions
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

# PIL: the format the processor is expecting
image = Image.open(image_path)

inputs = processor(images=image, return_tensors="pt")

# prediction happens over here
with torch.no_grad():
    # output scores : represent the model's confidence in each of the possible classes
    logits = model(**inputs).logits

# finding the label with the highest output score.
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])