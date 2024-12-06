import gradio
import cv2
from PIL import Image
import numpy as np

import spaces
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import accelerate
import transformers
from random import randrange


from transformers.utils.hub import move_cache
move_cache()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# base_model_id = "runwayml/stable-diffusion-v1-5"
base_model_id = "botp/stable-diffusion-v1-5"
model_id = "LuyangZ/FloorAI"
# model_id = "LuyangZ/controlnet_Neufert4_64_100"

# controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
# controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype="auto")
# controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float32, force_download=True)
controlnet = ControlNetModel.from_pretrained(model_id, force_download=True)

controlnet.to(device)
torch.cuda.empty_cache()


# pipeline = StableDiffusionControlNetPipeline.from_pretrained(base_model_id , controlnet=controlnet, torch_dtype=torch.float32,  force_download=True)
# pipeline = StableDiffusionControlNetPipeline.from_pretrained(base_model_id , controlnet=controlnet, torch_dtype="auto")
# pipeline = StableDiffusionControlNetPipeline.from_pretrained(base_model_id , controlnet=controlnet, torch_dtype=torch.float16)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(base_model_id, controlnet=controlnet, force_download=True)
pipeline.safety_checker = None
pipeline.requires_safety_checker = False

pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)



# pipeline.enable_xformers_memory_efficient_attention()
# pipeline.enable_model_cpu_offload()
# pipeline.enable_attention_slicing()


pipeline = pipeline.to(device)
torch.cuda.empty_cache()




def expand2square(ol_img, background_color):
    width, height = ol_img.size
    
    if width == height:
        pad = int(width*0.2)
        width_new = width + pad
        halfpad = int(pad/2)
        
        ol_result = Image.new(ol_img.mode, (width_new, width_new), background_color)
        ol_result.paste(ol_img, (halfpad, halfpad))

        return ol_img
    
    elif width > height:
        
        pad = int(width*0.2)
        width_new = width + pad
        halfpad = int(pad/2)

        ol_result = Image.new(ol_img.mode, (width_new, width_new), background_color)
        ol_result.paste(ol_img, (halfpad, (width_new - height) // 2))

        return ol_result
    
    else:
        pad = int(height*0.2)
        height_new = height + pad
        halfpad = int(pad/2)

        ol_result = Image.new(ol_img.mode, (height_new, height_new), background_color)
        ol_result.paste(ol_img, ((height_new - width) // 2, halfpad))

        return ol_result

def clean_img(image, mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY_INV)[1]
    
    image[mask<250]=(255,255,255)
    image = Image.fromarray(image).convert('RGB')
    return image

# @spaces.GPU
@spaces.GPU(duration=40)

def floorplan_generation(outline, num_of_rooms):
    new_width = 512
    new_height = 512

    outline = cv2.cvtColor(outline, cv2.COLOR_RGB2BGR)
    outline_original = outline.copy()

    gray = cv2.cvtColor(outline, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)[1]

    x,y,w,h = cv2.boundingRect(thresh)
    n_outline = outline_original[y:y+h, x:x+w]
    n_outline = cv2.cvtColor(n_outline, cv2.COLOR_BGR2RGB)
    n_outline = Image.fromarray(n_outline).convert('RGB')
    n_outline = expand2square(n_outline, (255, 255, 255))
    n_outline = n_outline.resize((new_width, new_height))

    num_of_rooms = str(num_of_rooms)
    validation_prompt = "floor plan, " + num_of_rooms + " rooms"
    validation_image = n_outline

    image_lst = []
    for i in range(5):
        seed = randrange(5000)
        generator = torch.Generator(device=device).manual_seed(seed)
    
    
        image = pipeline(validation_prompt, 
                         validation_image, 
                         num_inference_steps=20, 
                         generator=generator).images[0]
    
        image = np.array(image)
        mask = np.array(n_outline)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        image = clean_img(image, mask)

        image_arr = np.array(image)
        mask_bgr = cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)
        mask_gray = cv2.cvtColor(mask_bgr, cv2.COLOR_RGB2GRAY)
        mask_white = cv2.threshold(mask_gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
        image_arr[mask_white<250]=(255,255,255)
        image_arr_copy = image_arr.copy()

        gray = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)[1]
        
        x,y,w,h = cv2.boundingRect(thresh)
        image_final = image_arr_copy[y:y+h, x:x+w]


        src = image_final
        tmp = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        _,alpha = cv2.threshold(tmp,250,255,cv2.THRESH_BINARY_INV)
        b, g, r = cv2.split(src)
        rgba = [b,g,r, alpha]
        dst = cv2.merge(rgba,4)
        image = Image.fromarray(dst)
        image_lst.append(image)
        
    return image_lst[0], image_lst[1], image_lst[2], image_lst[3], image_lst[4]

# from datetime import datetime
# print(str(datetime.now()))

gradio_interface = gradio.Interface(
  fn=floorplan_generation,
  inputs=[gradio.Image(label="Floor Plan Outline, Entrance"),
          gradio.Textbox(type="text", label="Number of Rooms", placeholder="Number of Rooms")],
  outputs=[gradio.Image(label="Generated Floor Plan 1"), 
           gradio.Image(label="Generated Floor Plan 2"),
           gradio.Image(label="Generated Floor Plan 3"),
           gradio.Image(label="Generated Floor Plan 4"),
           gradio.Image(label="Generated Floor Plan 5")],
  title="FloorAI",
  examples=[["example_1.png", "4"], ["example_2.png", "3"], ["example_3.png", "2"], ["example_4.png", "4"], ["example_5.png", "4"]])

#max_size=10, 
gradio_interface.queue(status_update_rate="auto", api_open=True)
gradio_interface.launch(share=True, show_api=True, show_error=True)
