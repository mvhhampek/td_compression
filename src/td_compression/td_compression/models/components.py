import torch
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "leaky_relu": lambda: nn.LeakyReLU(0.2),
    "linear": nn.Identity,
    "identity": nn.Identity,
    "sigmoid": nn.Sigmoid
}

def get_activation(name):
    if name not in ACTIVATIONS:
        raise ValueError(f"Activation '{name}' not found!")
    return ACTIVATIONS[name]()

def get_norm(norm_type, channels):
    if norm_type == 'group_norm':
        return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
    return nn.Identity()

class ConvEncoder2D(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_sizes, strides, padding, activation, latent_dim, input_shape, norm='group_norm', dropout=0.0):
        super().__init__()
        
        modules = []
        current_channels = in_channels
        
        for h_dim, k_size, stride, pad in zip(hidden_channels, kernel_sizes, strides, padding):
            layers = [
                nn.Conv2d(current_channels, h_dim, kernel_size=k_size, stride=stride, padding=pad),
                get_norm(norm, h_dim), 
                get_activation(activation)
            ]
            
            if dropout > 0.0:
                layers.append(nn.Dropout2d(p=dropout))
                
            modules.append(nn.Sequential(*layers))
            current_channels = h_dim
            
        self.feature_extractor = nn.Sequential(*modules)
        
        dummy_input = torch.zeros(1, *input_shape)
        with torch.no_grad():
            dummy_output = self.feature_extractor(dummy_input)
            
        self.conv_output_shape = dummy_output.shape[1:] 
        self.flattened_size = dummy_output.numel()
        
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        features_flat = torch.flatten(features, start_dim=1)
        mu = self.fc_mu(features_flat)
        logvar = self.fc_logvar(features_flat)
        return mu, logvar


class ConvDecoder2D(nn.Module):
    def __init__(self, latent_dim, out_channels, hidden_channels, kernel_sizes, strides, padding, output_padding, activation, final_activation, conv_output_shape, norm='group_norm', dropout=0.0, upsample_mode='nearest'):
        super().__init__()
        self.conv_output_shape = conv_output_shape 
        
        flattened_size = conv_output_shape[0] * conv_output_shape[1] * conv_output_shape[2]
        self.fc_expand = nn.Linear(latent_dim, flattened_size)
        
        modules = []
        current_channels = conv_output_shape[0] 
                
        for h_dim, k_size, stride, out_pad, pad in zip(hidden_channels, kernel_sizes, strides, output_padding, padding):
            s = stride[0] if isinstance(stride, (list, tuple)) else stride
            p = pad[0] if isinstance(pad, (list, tuple)) else pad
            k = k_size[0] if isinstance(k_size, (list, tuple)) else k_size
            o_pad = out_pad[0] if isinstance(out_pad, (list, tuple)) else out_pad

            layers = []
            
            if s > 1 and upsample_mode != 'transpose':
                layers.append(nn.Upsample(scale_factor=s, mode=upsample_mode))
                layers.append(nn.Conv2d(current_channels, h_dim, kernel_size=k, stride=1, padding=p))
            else:
                layers.append(nn.ConvTranspose2d(current_channels, h_dim, kernel_size=k, stride=s, padding=p, output_padding=o_pad))
                
            layers.append(get_norm(norm, h_dim))
            layers.append(get_activation(activation))
            
            if dropout > 0.0:
                layers.append(nn.Dropout2d(p=dropout))
                
            modules.append(nn.Sequential(*layers))
            current_channels = h_dim
            
        self.hidden_layers = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(
            nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
            get_activation(final_activation)
        )

    def forward(self, z):
        x = self.fc_expand(z)
        x = x.view(-1, *self.conv_output_shape)
        x = self.hidden_layers(x)
        return self.final_layer(x)