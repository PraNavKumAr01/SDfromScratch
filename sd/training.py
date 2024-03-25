import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from pipeline import rescale, get_time_embedding
import model_loader
from datasets import load_dataset
from ddpm import DDPMSampler
from transformers import CLIPTokenizer
from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion

HEIGHT = 512
WIDTH = 512
LATENT_HEIGHT = HEIGHT // 8
LATENT_WIDTH = WIDTH // 8

DEVICE = "CPU"

generator = torch.Generator(device = DEVICE)

latents_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

ckpt_path = "data/v1-5-pruned-emaonly.ckpt"

models = model_loader.preload_model_from_standard_weights(ckpt_path, DEVICE)

dataset_name = "m1guelpf/nouns"
dataset = load_dataset(dataset_name, split="train")
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size = 16, shuffle=True)

scheduler = DDPMSampler(generator = generator)

tokenizer = CLIPTokenizer("data/tokenizer_vocab.json", merges_file = "data/tokenizer_merges.txt")

def train(models, train_dataloader, scheduler, num_epochs, device, tokenizer):

    vae_encoder = VAE_Encoder
    unet = Diffusion
    clip = CLIP

    for epoch in range(num_epochs):
        train_loss = 0.0
        progress_bar = tqdm(train_dataloader, total=len(train_dataloader))

        for step, batch in enumerate(progress_bar):

            image, text = batch["image"].to(device), batch["text"]

            input_image = input_image.resize((HEIGHT, WIDTH))
            input_image_array = np.array(input_image)
            input_image_tensor = torch.tensor(input_image_array, dtype = torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            input_image_tensor = input_image_tensor.unsqueeze(0)
            input_image_tensor = input_image_tensor.permute(0,3,1,2)

            encoder_noise = torch.randn(latents_shape, generator = generator, device = device)

            # Convert images to latent space
            latents = vae_encoder(image, encoder_noise)

            # Sample a random timestep for each image
            bsz = latents.shape[0]
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (bsz,), device=device)
            timesteps = timesteps.long()

            timesteps_embedding = get_time_embedding(timesteps).to(device)

            # Add noise to the latents (forward diffusion process)
            noisy_latents, noise = scheduler.add_noise(latents, timesteps)

            # Encode text with CLIP
            text_tokens = tokenizer.batch_encode_plus([text], padding="max_length", max_length=77, return_tensors="pt").to(device)
            text_embeddings = clip(text_tokens)

            # Predict the noise residual and compute loss
            noise_pred = unet(noisy_latents, text_embeddings,timesteps_embedding)
            
            target = noise

            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

            # Backpropagate
            optimizer = optim.AdamW(unet.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=train_loss / (step + 1))

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss / len(train_dataloader)}")