import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F # Add this import

# Assuming the ContradictionLoss class and its helper functions are in SAAF-OS.losses.contradiction_loss
# For testing purposes, we might need to adjust the import path based on the test execution environment
# or duplicate/mock the helper functions if they are not easily importable.

# If SAAF-OS is a package and tests are run from the root or a specific test directory:
from losses.contradiction_loss import ContradictionLoss, vae_kl_divergence, info_nce_loss, energy_phi, constraint_penalties, imine_penalty

# Dummy constraint function for testing
def dummy_norm_constraint(z, max_norm=1.0):
    return torch.relu(torch.norm(z, p=2, dim=-1) - max_norm).mean()

def another_dummy_constraint(z):
    return torch.abs(z.mean() - 0.5) # Example: penalize if mean is not 0.5

# Dummy IMINE estimator for testing
class DummyIMINE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Ensure the layer can handle the concatenated dimensions
        self.fc = nn.Linear(input_dim * 2, 1)
    def forward(self, z1, z2):
        # Ensure z1 and z2 are correctly shaped for concatenation
        if z1.ndim == 1: z1 = z1.unsqueeze(0)
        if z2.ndim == 1: z2 = z2.unsqueeze(0)
        
        # If batch sizes are different and one is 1, expand it
        if z1.size(0) == 1 and z2.size(0) > 1:
            z1 = z1.expand(z2.size(0), -1)
        elif z2.size(0) == 1 and z1.size(0) > 1:
            z2 = z2.expand(z1.size(0), -1)
            
        combined = torch.cat([z1, z2], dim=-1)
        return self.fc(combined)

class TestContradictionLoss(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.latent_dim = 8
        self.num_negative_samples_infonce = 3

        # Synthetic inputs
        self.z_t = torch.randn(self.batch_size, self.latent_dim, requires_grad=True)
        self.z_t_plus_1_predicted = torch.randn(self.batch_size, self.latent_dim, requires_grad=True)
        self.z_t_plus_1_adjusted = torch.randn(self.batch_size, self.latent_dim)

        # VAE components (optional)
        self.vae_mu = torch.randn(self.batch_size, self.latent_dim)
        self.vae_logvar = torch.randn(self.batch_size, self.latent_dim) # log(sigma^2)

        # InfoNCE components (optional)
        self.infonce_positive_key = torch.randn(self.batch_size, self.latent_dim)
        self.infonce_negative_keys = torch.randn(self.batch_size, self.num_negative_samples_infonce, self.latent_dim)

        # Constraint components (optional)
        self.constraint_fns = [
            lambda z: dummy_norm_constraint(z, max_norm=float(self.latent_dim**0.5)), # Avg norm per dim = 1
            another_dummy_constraint
        ]
        self.constraint_weights = [0.5, 0.3]

        # IMINE component (optional)
        self.imine_model = DummyIMINE(self.latent_dim)

        # Initialize the loss function with all components
        self.loss_fn_all_terms = ContradictionLoss(
            lambda_kl=0.1,
            lambda_infonce=0.2,
            lambda_energy=0.3,
            lambda_constraint=0.4,
            lambda_imine=0.5,
            constraint_lambdas=self.constraint_weights,
            constraint_functions=self.constraint_fns,
            imine_estimator_model=self.imine_model,
            temperature_infonce=0.07
        )
        
        # Initialize a minimal loss function (only MSE)
        self.loss_fn_mse_only = ContradictionLoss(
            lambda_kl=0.0, lambda_infonce=0.0, lambda_energy=0.0,
            lambda_constraint=0.0, lambda_imine=0.0
        )

    def test_forward_pass_all_terms(self):
        total_loss, individual_losses = self.loss_fn_all_terms(
            self.z_t,
            self.z_t_plus_1_predicted,
            self.z_t_plus_1_adjusted,
            vae_mu=self.vae_mu,
            vae_logvar=self.vae_logvar,
            infonce_positive_key=self.infonce_positive_key,
            infonce_negative_keys=self.infonce_negative_keys
        )
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertGreaterEqual(total_loss.item(), 0) # Loss should be non-negative generally
        self.assertIn('mse', individual_losses)
        self.assertIn('kl_divergence', individual_losses)
        self.assertIn('infonce', individual_losses)
        self.assertIn('energy_phi', individual_losses)
        self.assertIn('constraint_penalties', individual_losses)
        self.assertIn('imine_penalty', individual_losses)
        self.assertIn('total_loss', individual_losses)
        self.assertEqual(total_loss, individual_losses['total_loss'])

    def test_forward_pass_mse_only(self):
        total_loss, individual_losses = self.loss_fn_mse_only(
            self.z_t,
            self.z_t_plus_1_predicted,
            self.z_t_plus_1_adjusted
            # No optional args
        )
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertGreaterEqual(total_loss.item(), 0)
        self.assertIn('mse', individual_losses)
        self.assertEqual(total_loss.item(), individual_losses['mse'].item())
        self.assertEqual(individual_losses['kl_divergence'].item(), 0)
        self.assertEqual(individual_losses['infonce'].item(), 0)
        self.assertEqual(individual_losses['energy_phi'].item(), 0)
        self.assertEqual(individual_losses['constraint_penalties'].item(), 0)
        self.assertEqual(individual_losses['imine_penalty'].item(), 0)


    def test_backward_pass_all_terms(self):
        # Simulate a model whose parameters we want to optimize
        model = nn.Linear(self.latent_dim, self.latent_dim)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Get model prediction
        z_t_plus_1_model_pred = model(self.z_t)

        total_loss, _ = self.loss_fn_all_terms(
            self.z_t,
            z_t_plus_1_model_pred, # Use model's prediction
            self.z_t_plus_1_adjusted,
            vae_mu=self.vae_mu,
            vae_logvar=self.vae_logvar,
            infonce_positive_key=self.infonce_positive_key,
            infonce_negative_keys=self.infonce_negative_keys
        )
        
        optimizer.zero_grad()
        try:
            total_loss.backward()
        except Exception as e:
            self.fail(f"backward() call failed with exception: {e}")

        # Check if gradients are computed for model parameters
        for param in model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertNotEqual(torch.sum(param.grad**2).item(), 0) # Grads should not be all zero

        # Check if gradients are computed for input tensors that require grad
        # Note: z_t_plus_1_predicted is now z_t_plus_1_model_pred, which is intermediate.
        # We need to check grad for z_t if it's part of the graph leading to loss.
        if self.z_t.grad is not None: # Grad might be None if z_t is not part of any term that has a gradient path
             self.z_t.grad = None # Reset for next test if any
        # If z_t_plus_1_predicted itself was a leaf tensor requiring grad and part of loss:
        # self.assertIsNotNone(self.z_t_plus_1_predicted.grad)


    def test_backward_pass_mse_only(self):
        model = nn.Linear(self.latent_dim, self.latent_dim)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        z_t_plus_1_model_pred = model(self.z_t)

        total_loss, _ = self.loss_fn_mse_only(
            self.z_t,
            z_t_plus_1_model_pred,
            self.z_t_plus_1_adjusted
        )
        optimizer.zero_grad()
        try:
            total_loss.backward()
        except Exception as e:
            self.fail(f"backward() call failed with exception: {e}")
        
        for param in model.parameters():
            self.assertIsNotNone(param.grad)

    def test_individual_loss_components(self):
        # Test MSE
        mse_val = F.mse_loss(self.z_t_plus_1_predicted, self.z_t_plus_1_adjusted)
        _, losses = self.loss_fn_all_terms(self.z_t, self.z_t_plus_1_predicted, self.z_t_plus_1_adjusted)
        self.assertAlmostEqual(losses['mse'].item(), mse_val.item(), places=6)

        # Test KL
        if self.vae_mu is not None and self.vae_logvar is not None:
            kl_val = vae_kl_divergence(self.vae_mu, self.vae_logvar) * self.loss_fn_all_terms.lambda_kl
            _, losses = self.loss_fn_all_terms(self.z_t, self.z_t_plus_1_predicted, self.z_t_plus_1_adjusted, vae_mu=self.vae_mu, vae_logvar=self.vae_logvar)
            self.assertAlmostEqual(losses['kl_divergence'].item(), kl_val.item(), places=6)

        # Test InfoNCE
        if self.infonce_positive_key is not None and self.infonce_negative_keys is not None:
            infonce_val = info_nce_loss(self.z_t_plus_1_predicted, self.infonce_positive_key, self.infonce_negative_keys, self.loss_fn_all_terms.temperature_infonce) * self.loss_fn_all_terms.lambda_infonce
            _, losses = self.loss_fn_all_terms(self.z_t, self.z_t_plus_1_predicted, self.z_t_plus_1_adjusted, infonce_positive_key=self.infonce_positive_key, infonce_negative_keys=self.infonce_negative_keys)
            # Detach predicted before passing to info_nce_loss if comparing, as original call uses it with grad
            infonce_val_no_grad = info_nce_loss(self.z_t_plus_1_predicted.detach(), self.infonce_positive_key, self.infonce_negative_keys, self.loss_fn_all_terms.temperature_infonce) * self.loss_fn_all_terms.lambda_infonce
            self.assertAlmostEqual(losses['infonce'].item(), infonce_val_no_grad.item(), places=6)


        # Test Energy Phi
        energy_val = energy_phi(self.z_t, self.z_t_plus_1_predicted).mean() * self.loss_fn_all_terms.lambda_energy
        _, losses = self.loss_fn_all_terms(self.z_t, self.z_t_plus_1_predicted, self.z_t_plus_1_adjusted)
        self.assertAlmostEqual(losses['energy_phi'].item(), energy_val.item(), places=6)


        # Test Constraint Penalties
        if self.constraint_fns:
            constraint_val = constraint_penalties(self.z_t, self.constraint_fns, self.constraint_weights) * self.loss_fn_all_terms.lambda_constraint
            _, losses = self.loss_fn_all_terms(self.z_t, self.z_t_plus_1_predicted, self.z_t_plus_1_adjusted)
            self.assertAlmostEqual(losses['constraint_penalties'].item(), constraint_val.item(), places=6)

        # Test IMINE Penalty
        if self.imine_model is not None:
            # imine_penalty returns 0 for now, so this test will be trivial unless placeholder changes
            imine_val = imine_penalty(self.z_t, self.z_t_plus_1_predicted, self.imine_model) * self.loss_fn_all_terms.lambda_imine
            _, losses = self.loss_fn_all_terms(self.z_t, self.z_t_plus_1_predicted, self.z_t_plus_1_adjusted)
            self.assertAlmostEqual(losses['imine_penalty'].item(), imine_val.item(), places=6)
            
    def test_no_optional_components(self):
        # Test ContradictionLoss when no optional components (KL, InfoNCE, Constraints, IMINE) are provided
        loss_fn_minimal = ContradictionLoss(lambda_kl=0.0, lambda_infonce=0.0, lambda_energy=0.0, lambda_constraint=0.0, lambda_imine=0.0)
        
        total_loss, individual_losses = loss_fn_minimal(
            self.z_t,
            self.z_t_plus_1_predicted,
            self.z_t_plus_1_adjusted
            # No vae_mu, vae_logvar, infonce_positive_key, infonce_negative_keys
        )
        
        expected_mse = F.mse_loss(self.z_t_plus_1_predicted, self.z_t_plus_1_adjusted)
        # Energy phi is calculated on z_t and z_t_plus_1_predicted, but its lambda is 0
        # expected_energy = energy_phi(self.z_t, self.z_t_plus_1_predicted).mean() * 0.0
        
        self.assertAlmostEqual(total_loss.item(), expected_mse.item(), places=6)
        self.assertAlmostEqual(individual_losses['mse'].item(), expected_mse.item(), places=6)
        self.assertEqual(individual_losses['kl_divergence'].item(), 0.0)
        self.assertEqual(individual_losses['infonce'].item(), 0.0)
        # self.assertEqual(individual_losses['energy_phi'].item(), expected_energy.item()) # Will be non-zero if lambda_energy > 0
        self.assertEqual(individual_losses['energy_phi'].item(), 0.0) # Since lambda_energy is 0.0 for loss_fn_minimal
        self.assertEqual(individual_losses['constraint_penalties'].item(), 0.0)
        self.assertEqual(individual_losses['imine_penalty'].item(), 0.0)

if __name__ == '__main__':
    unittest.main()

