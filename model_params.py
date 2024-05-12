# model_params.py

import torch
from torchsummary import summary
from basicsr.archs.ddcolor_arch import DDColor

# Load the configuration
config = {
    'encoder_name': 'convnext-l',
    'decoder_name': 'MultiScaleColorDecoder',
    'num_queries': 100,
    'num_scales': 3,
    'dec_layers': 9,
    'encoder_from_pretrain': True,
    'num_output_channels': 2,
    'do_normalize': False,
    'input_size': (256, 256)  # Add this line to specify the input size
}

# Instantiate the model with the configuration
model = DDColor(**config)

# Move the model to the appropriate device (e.g., GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Create a sample input tensor
sample_input = torch.randn(1, 3, 256, 256).to(device)  # Sample input with shape (1, 3, 256, 256)

# Print the model summary
print("Model Summary:")
summary(model, (3, 256, 256))  # Pass the sample input tensor to the summary function

# Calculate the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Number of Parameters: {total_params:,}")
