import torch

import comfy

from .tinyvae import TinyAutoEncoder


size_variants = {
    'zepto': [1 ,2 ,2 ,4],
    'yocto': [16 ,8 ,8 ,16],
    'zeptometer': [2, 4, 8, 16],
    'yoctometer': [4, 8, 8, 16],
    'angstrom': [4, 8, 16, 32],
    'attometer': [8, 16, 16, 32],
    'nanometer': [8, 16, 32, 64],
    'micrometer': [16, 32, 32, 64],
    'pico': [16, 32, 64, 64],
    'nano': [16, 32, 64, 128],
    'micro': [24, 48, 96, 192],
    'tiny': [256, 128, 64, 32],
    'small': [64, 128, 256, 512],
    'normal': [128, 256, 512, 1024],
    'large': [256, 512, 1024, 2048],
}

var_names = [k for k, _ in size_variants.items()]

class TAEFLUX:
    def __init__(self, model):
        self.first_stage_model = model
        self.patcher = comfy.model_patcher.ModelPatcher(self.first_stage_model, load_device=torch.device('cuda'),
                                                        offload_device=torch.device('cuda'))
        self.process_input = lambda image: image * 2.0 - 1.0
        self.process_output = lambda image: torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)

        self.vae_dtype = torch.float16
        self.device = torch.device('cuda')
        self.output_device = torch.device('cuda')
        self.downscale_ratio = 8
        self.upscale_ratio = 8
        self.latent_channels = 16
        self.output_channels = 3
    def decode(self, samples_in):
        # try:
        #     # memory_used = self.memory_used_decode(samples_in.shape, self.vae_dtype)
        #     # model_management.load_models_gpu([self.patcher], memory_required=memory_used)
        #     # free_memory = model_management.get_free_memory(self.device)
        #     # batch_number = int(free_memory / memory_used)
        #     # batch_number = max(1, batch_number)
        batch_number = 1
        pixel_samples = torch.empty((samples_in.shape[0], self.output_channels) + tuple(map(lambda a: a * self.upscale_ratio, samples_in.shape[2:])), device=self.output_device)
        for x in range(0, samples_in.shape[0], batch_number):
            samples = samples_in[x:x+batch_number].to(self.vae_dtype).to(self.device)
            pixel_samples[x:x+batch_number] = self.process_output(self.first_stage_model.decoder(samples).to(self.output_device).float())
        # except model_management.OOM_EXCEPTION as e:
        #     logging.warning("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
        #     if len(samples_in.shape) == 3:
        #         pixel_samples = self.decode_tiled_1d(samples_in)
        #     else:
        #         pixel_samples = self.decode_tiled_(samples_in)

        pixel_samples = pixel_samples.to(self.output_device).movedim(1,-1)
        return pixel_samples

class TAEFLUXLoader:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"variant": (var_names,), }}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_taeflux"

    CATEGORY = "loaders"

    #TODO: scale factor?
    def load_taeflux(self, variant):
        channels = size_variants.get(variant, size_variants['yocto'])
        print(channels)
        model = TinyAutoEncoder(channels=channels).half().to('cuda')
        model.decoder.load_state_dict(
            torch.load('/home/mix/Playground/flux_base/flux_tiny_vae_versions/yocto_decoder_flux_150.pth', weights_only=True))

        return (TAEFLUX(model),)
