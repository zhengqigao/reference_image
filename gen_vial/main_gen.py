from diffusers import StableDiffusionImg2ImgPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import torch
import torch.nn as nn
import timm
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionInpaintPipeline
import random
from torchvision import models
import torchvision
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
import requests
from PIL import Image, ImageDraw
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

    ref_image = fetch_image_from_url(config['image_path'])

    mask_image = Image.new("L", ref_image.size, 0)
    draw = ImageDraw.Draw(mask_image)
    width, height = ref_image.size
    draw.rectangle([(width // 4, height // 4), (3 * width // 4, 3 * height // 4)], fill=255)

    for i in range(config['iter']):
        images = None
        for j in range(config['batch_size']):
            choose_list = config['good_catalog'] if j % 2 == 0 else config['defect_catalog']
            edited_image = \
                pipe(prompt=random.choice(choose_list), image=ref_image,
                     mask_image=mask_image).images[0]
            tensor_image = transform(edited_image).unsqueeze(0)
            images = torch.cat((images, tensor_image), dim=0) if images is not None else tensor_image

        images = images.to(config['device'])
        label = torch.tensor([1 if j % 2 == 0 else 0 for j in range(config['batch_size'])]).to(config['device']).to(
            torch.long)

        optimizer.zero_grad()
        pred = model(images)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        print(f"iter {i}/{config['iter']}, loss: {loss.item():.3f}")

    return model


def load_model(model_name, device, num_classes=2, pretrained=True):
    # see: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = torch.nn.Linear(512, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = torch.nn.Linear(2048, num_classes)
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=pretrained)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    elif model_name == 'vgg13_bn':
        model = models.vgg13_bn(pretrained=pretrained)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    elif model_name == 'vgg11_bn':
        model = models.vgg11_bn(pretrained=pretrained)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    elif model_name == 'squeezenet':
        model = models.squeezenet1_0(pretrained=pretrained)
        model.features[0] = torch.nn.Conv2d(3, 96, kernel_size=(3, 3), stride=(2, 2))
        model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))

    # the following is for generative diffusion models
    elif model_name == 'sd_xl':
        model = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16",
            use_safetensors=True)
    elif model_name == 'sd_inpainting':
        model = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        )
    elif model_name == 'sd_base':
        model = StableDiffusionImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-base",
            torch_dtype=torch.float16,
        )
    else:
        raise ValueError('Model not supported')
    model = model.to(device)
    return model


def show_gen_image(config, num_image = 1, display_or_save='display', save_path='./gen_images'):
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

    # Create a mask image of the same size as the input image
    mask_image = Image.new("L", ref_image.size, 0)
    draw = ImageDraw.Draw(mask_image)
    width, height = ref_image.size
    draw.rectangle([(width // 4, height // 4), (3 * width // 4, 3 * height // 4)], fill=255)

    if display_or_save == 'save':
        mask_image.save(os.path.join(save_path, 'mask_image.png'))
    else:
        plt.figure()
        plt.imshow(mask_image)
        plt.title('mask image')
        plt.show()

    for i, prompt in enumerate(gen_list):
        for j in range(num_image):
            edited_image = pipe(prompt, image=ref_image, mask_image=mask_image,
                                # num_inference_steps=50,
                                # guidance_scale=7.5,
                                # strength=0.8
                                ).images[0]
            plt.figure()
            plt.imshow(edited_image)
            plt.title(prompt)
            if display_or_save == 'display':
                plt.show()
            else:
                plt.savefig(os.path.join(save_path, 'image_' + str(j) + '_' + str(i) + '.png'))
    return None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        'device': device,
        'gen_model': load_model('sd_inpainting', device),
        'image_path': 'https://github.com/zhengqigao/reference_image/blob/main/gen_vial/ref_vial.jpg?raw=true',
        'defect_catalog': [
            'a very large crack on the transparent glass',
            'A single strand of black human hair on the transparent glass',
            'a piece of dirt or dust on the transparent glass',
            'a clear large scratch on the transparent glass',
            'a cat face on the sidewall of the glass vial',  # for debugging purpose, or even add more severe 'outliers/defects'
        ],
        'good_catalog': [
            'a clean and clear transparent glass vial',
            'the color of vial cap is blue',
        ],
        'model': load_model('resnet18', device),
        'lr': 0.001,
        'iter': 2,
        'batch_size': 4
    }

    # show_gen_image(config, num_image=2, display_or_save='save', save_path='./gen_images')

    config['model'] = train(config)


main()
