import unittest
from calcium_avae.snDGM.sndgm.avae import VAE  # Adjust the import based on your actual class/function names

class TestVAE(unittest.TestCase):

    def setUp(self):
        # Initialize the VAE model with test parameters
        self.vae = VAE(latent_dim=32, hidden_dim=256)

    def test_vae_initialization(self):
        # Test if the VAE model initializes correctly
        self.assertIsNotNone(self.vae)

    def test_forward_pass(self):
        # Test the forward pass of the VAE
        test_input = torch.randn(16, 10, 3)  # Example input shape
        output = self.vae(test_input)
        self.assertEqual(output.shape, test_input.shape)

    def test_loss_function(self):
        # Test the loss function implementation
        test_input = torch.randn(16, 10, 3)
        recon_batch, mu, logvar = self.vae(test_input)
        loss = self.vae.loss_function(recon_batch, test_input, mu, logvar)
        self.assertIsInstance(loss, float)

if __name__ == '__main__':
    unittest.main()