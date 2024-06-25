import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Positional Encoding
def positional_encoding(x, L=10):
    rets = [x]
    for i in range(L):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(2.0 ** i * x))
    return torch.cat(rets, dim=-1)

# NeRF Model
class NeRF(nn.Module):
    def __init__(self, L=10, hidden_dim=256):
        super(NeRF, self).__init__()
        self.L = L
        self.hidden_dim = hidden_dim
        self.pts_linears = nn.ModuleList(
            [nn.Linear(3 + 6 * L, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for _ in range(7)]
        )
        self.views_linears = nn.ModuleList([nn.Linear(3 + 6 * L, hidden_dim // 2)])
        self.feature_linear = nn.Linear(hidden_dim, hidden_dim)
        self.alpha_linear = nn.Linear(hidden_dim, 1)
        self.rgb_linear = nn.Linear(hidden_dim // 2, 3)

    def forward(self, x):
        input_pts, input_views = x[..., :3], x[..., 3:]
        h = positional_encoding(input_pts, self.L)
        for i, l in enumerate(self.pts_linears):
            h = l(h)
            if i in {4}:
                h = h + positional_encoding(input_pts, self.L)
            h = torch.relu(h)
        alpha = self.alpha_linear(h)
        h = self.feature_linear(h)

        h = torch.cat([h, positional_encoding(input_views, self.L)], -1)
        for l in self.views_linears:
            h = l(h)
            h = torch.relu(h)
        rgb = self.rgb_linear(h)

        return torch.cat([rgb, alpha], -1)

# Volume Rendering
def volume_rendering(nerf_model, rays, n_samples=64):
    rendered_image = torch.zeros((rays.shape[0], 3))
    for i, ray in enumerate(rays):
        t_vals = torch.linspace(0, 1, n_samples)
        points = ray['origin'] + t_vals[:, None] * ray['direction']
        input_views = ray['direction'].expand_as(points)
        inputs = torch.cat([points, input_views], dim=-1)
        
        rgb_sigma = nerf_model(inputs)
        
        rgb = rgb_sigma[:, :3]
        sigma = rgb_sigma[:, 3]
        
        delta = t_vals[1] - t_vals[0]
        alpha = 1.0 - torch.exp(-sigma * delta)
        
        weights = alpha * torch.cumprod(torch.cat([torch.ones((1,)), 1.0 - alpha + 1e-10], dim=0), dim=0)[:-1]
        
        rendered_image[i] = torch.sum(weights[:, None] * rgb, dim=0)
    return rendered_image

# Training NeRF
def train_nerf(nerf_model, dataset, epochs=1000, batch_size=1024, lr=1e-4):
    optimizer = optim.Adam(nerf_model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataset.get_batches(batch_size):
            rays, target_images = batch
            optimizer.zero_grad()
            predictions = volume_rendering(nerf_model, rays)
            loss = loss_fn(predictions, target_images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs} Loss: {epoch_loss / len(dataset)}')

# Dummy Dataset
class DummyDataset:
    def __init__(self, n_samples=10000):
        self.n_samples = n_samples

    def get_batches(self, batch_size):
        for _ in range(self.n_samples // batch_size):
            rays = {
                'origin': torch.randn(batch_size, 3),
                'direction': torch.randn(batch_size, 3)
            }
            target_images = torch.rand(batch_size, 3)
            yield rays, target_images

# Main function
def main():
    L = 10
    hidden_dim = 256
    epochs = 100
    batch_size = 1024
    lr = 1e-4

    nerf_model = NeRF(L=L, hidden_dim=hidden_dim)
    dataset = DummyDataset(n_samples=10000)

    train_nerf(nerf_model, dataset, epochs=epochs, batch_size=batch_size, lr=lr)

    # Render a test image
    test_rays = {
        'origin': torch.randn(1, 3),
        'direction': torch.randn(1, 3)
    }
    rendered_image = volume_rendering(nerf_model, [test_rays])
    plt.imshow(rendered_image.detach().numpy().reshape(1, 3))
    plt.show()

if __name__ == "__main__":
    main()
