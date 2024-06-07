from diffusers import StableDiffusionImg2ImgPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import torch
import torch.nn as nn
import timm
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionInpaintPipeline
import random
import torchvision
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
import requests
from PIL import Image
from io import BytesIO


def fetch_image_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        raise Exception("Failed to fetch image from URL")


def inference(model, data_path):
    # placeholder when we have access to the real vial dataset
    pass


def train(config):
    pipe, model = config['gen_model'], config['model']
    optimizer = torch.optim.Adam(config['model'].parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()

    transform = transforms.ToTensor()

    ref_image = load_image(config['image_path'])

    for i in range(config['iter']):
        images = None
        for j in range(config['batch_size']):
            choose_list = config['good_catalog'] if j % 2 == 0 else config['defect_catalog']
            edited_image = \
                pipe(prompt=random.choice(choose_list), image=ref_image).images[0]
            tensor_image = transform(edited_image).unsqueeze(0)
            images = torch.cat((images, tensor_image), dim=0) if images is not None else tensor_image

        images = images.to(config['device'])
        label = torch.tensor([1 if j % 2 == 0 else 0 for j in range(config['batch_size'])]).to(config['device'])

        # optimize
        optimizer.zero_grad()
        pred = model(images)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        print(f"iter {i}/{config['iter']}, loss: {loss.item():.3f}")

    return model


def load_model(model_name, device, pretrained=True):
    if model_name == 'resnet18':
        model = timm.create_model('resnet18', pretrained)
    elif model_name == 'resnet50':
        model = timm.create_model('resnet50', pretrained)
    elif model_name == 'resnet101':
        model = timm.create_model('resnet101', pretrained)
    elif model_name == 'mobilenet':
        model = torchvision.models.mobilenet_v2(pretrained)
    elif model_name == 'sd_xl':
        model = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16",
            use_safetensors=True)
    elif model_name == 'sd_inpainting':
        model = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        )
    else:
        raise ValueError('Model not supported')
    model = model.to(device)
    return model


def show_gen_image(config, display_or_save='display', save_path='./gen_images'):
    pipe = config['gen_model']
    ref_image = fetch_image_from_url(config['image_path'])

    print(ref_image)

    gen_list = config['defect_catalog'] + config['good_catalog']

    if display_or_save == 'save':
        if not os.path.exists(save_path):
            print(f"create the directory {save_path}")
            os.makedirs(save_path)
        else:
            print(f"directory exists, delete everything in {save_path}")
            for file in os.listdir(save_path):
                os.remove(os.path.join(save_path, file))

    for i, prompt in enumerate(gen_list):
        edited_image = pipe(prompt, image=ref_image).images[0]
        plt.figure()
        plt.imshow(edited_image)
        plt.title(prompt)
        if display_or_save == 'display':
            plt.show()
        else:
            plt.savefig(os.path.join(save_path, 'image_' + str(i) + '.png'))
    return None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        'device': device,
        'gen_model': load_model('sd_xl', device),
        'image_path': 'https://github.com/zhengqigao/reference_image/blob/main/gen_vial/ref_vial.jpg?raw=true',
        'defect_catalog': [
            'a photo of a glass vial with a vertical crack on the sidewall of the glass vial',
            'a photo of a glass vial with a contamination spot on the sidewall of the glass vial',
        ],
        'good_catalog': [
            'a photo of a glass vial containing a white substance, possibly a powder or crystalline material'
        ],
        'model': load_model('resnet18', device),
        'lr': 0.001,
        'iter': 2,
        'batch_size': 4
    }

    show_gen_image(config, display_or_save='save', save_path='./gen_images')

    # config['model'] = train(config)


main()
