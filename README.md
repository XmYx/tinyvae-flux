# TinyVAE Training Repository

This repository contains the implementation of a Tiny Variational Autoencoder (VAE) trained using a cyclical annealing schedule. The model is trained on images and saves the encoder and decoder weights periodically.

## Repository Structure

```
├── .gitignore
├── LICENSE
├── decoded_image_epoch_999.png
├── input.jpg
├── requirements.txt
├── tiny_decoder_epoch_1000.pth
├── tiny_encoder_epoch_1000.pth
└── train_flux_tinyvae.py
```

## Installation

1. Clone the repository:

```sh
git clone https://github.com/XmYx/tinyvae-flux
cd tinyvae-flux
```

2. Create and activate a virtual environment (optional but recommended):

```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required packages:

```sh
pip install -r requirements.txt
```

## Training the Model

To start training the TinyVAE model, use the following command. Make sure you have a folder with training images.

```sh
python train_flux_tinyvae.py <data_folder> <output_folder>
```

Replace `<data_folder>` with the path to your folder containing the images, and `<output_folder>` with the path to the folder where you want to save the model checkpoints and generated images.

### Example

```sh
python train_flux_tinyvae.py ./data ./output
```

## Testing the Model

After training, you can use the saved encoder and decoder weights to test the model. Below is an example of how to load the model weights and generate an image from a sample input.

```python
import torch
from torchvision import transforms
from PIL import Image
from train_flux_tinyvae import TinyAutoEncoder, VaeImageProcessor, postprocess

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyAutoEncoder(size_variant='tiny').to(device)
model.encoder.load_state_dict(torch.load('output/tiny_encoder_epoch_1000.pth'))
model.decoder.load_state_dict(torch.load('output/tiny_decoder_epoch_1000.pth'))
model.eval()

# Load the input image
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
input_image = Image.open('input.jpg').convert("RGB")
input_tensor = transform(input_image).unsqueeze(0).to(device)

# Process the image
processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=16)
preprocessed = processor.preprocess(input_tensor, width=512, height=512)
encoded_sample = model.encoder(preprocessed)
decoded_sample = model.decoder(encoded_sample)

# Postprocess and save the output image
output_image = postprocess(decoded_sample[0])
output_image.save('output_image.png')
```

## Results

Sample output image after training for 1000 epochs:

![Decoded Image Epoch 999](decoded_image_epoch_999.png)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to open an issue or a pull request if you have any questions or suggestions!
```
