import torch
import pytorch_lightning as L
from .components import ConvEncoder2D, ConvDecoder2D

class LitVAE(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        # required for lightning to parse the config dictionary from the checkpoint
        self.save_hyperparameters(config) 
        
        model_cfg = config['model']
        enc_cfg = model_cfg['encoder']
        dec_cfg = model_cfg['decoder']

        enc_kwargs = {k: v for k, v in enc_cfg.items() if k != 'type'}
        self.encoder = ConvEncoder2D(
            latent_dim=model_cfg['latent_dim'], 
            input_shape=config['data']['input_shape'],
            **enc_kwargs
        )
        
        dec_kwargs = {k: v for k, v in dec_cfg.items() if k != 'type'}
        self.decoder = ConvDecoder2D(
            latent_dim=model_cfg['latent_dim'],
            conv_output_shape=self.encoder.conv_output_shape,
            **dec_kwargs
        )


    def forward(self, x):
        mu, logvar = self.encoder(x)
        outputs = self.decoder(mu)
        return outputs, mu, logvar