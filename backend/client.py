import torch
from torch import nn, optim
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Dict
import copy
import logging
from opacus import PrivacyEngine

class Client:
    """
    Client class for handling local training in federated learning.
    """

    def __init__(self, client_id: int, client_loader: DataLoader, dp: bool = False):
        """
        Initialize the Client.

        Args:
            client_id (int): Unique identifier for the client.
            client_loader (DataLoader): DataLoader for the client's dataset.
            dp (bool): Whether to use differential privacy.
        """
        self.client_id = client_id
        self.client_loader = client_loader
        self.privacy_engine = PrivacyEngine() if dp else None
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Client {self.client_id} initialized.")

    def _create_optimizer(self, model_parameters, optimizer_type: str, learning_rate: float, 
                         momentum: float = 0.0, weight_decay: float = 0.0):
        """
        Create optimizer based on type and parameters.
        
        Args:
            model_parameters: Model parameters
            optimizer_type (str): Type of optimizer ('SGD', 'Adam', 'AdamW')
            learning_rate (float): Learning rate
            momentum (float): Momentum factor (0 means no momentum)
            weight_decay (float): Weight decay factor
            
        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        optimizer_type = optimizer_type.upper()
        
        if optimizer_type == 'SGD':
            if momentum == 0.0:
                # SGD without momentum
                optimizer = optim.SGD(
                    model_parameters, 
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
            else:
                # SGD with momentum
                optimizer = optim.SGD(
                    model_parameters, 
                    lr=learning_rate,
                    momentum=momentum,
                    weight_decay=weight_decay
                )
                
        elif optimizer_type == 'ADAM':
            optimizer = optim.Adam(
                model_parameters, 
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Supported: SGD, Adam, AdamW")
            
        return optimizer

    def train(
        self,
        global_model: Module,
        local_epochs: int,
        learning_rate: float,
        device: str,
        optimizer: str = 'SGD',
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Train the global model on the client's local data.

        Args:
            global_model (Module): The global model to be trained.
            local_epochs (int): Number of local epochs for training.
            learning_rate (float): Learning rate for the optimizer.
            device (str): Device to run the training on ('cpu' or 'cuda').
            optimizer (str): Type of optimizer ('SGD', 'Adam', 'AdamW').
            momentum (float): Momentum factor (0.0 means no momentum for SGD).
            weight_decay (float): Weight decay for regularization.

        Returns:
            Dict[str, torch.Tensor]: The updated model parameters after training.
        """
        self.logger.info(f"Client {self.client_id} started training with {optimizer} optimizer.")
        

        client_model = global_model.to(device)
        client_model.train()
        
        
        optimizer_instance = self._create_optimizer(
            client_model.parameters(), 
            optimizer, 
            learning_rate, 
            momentum, 
            weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()

        # Conditional AMP setup
        use_amp = device == 'cuda'
        scaler = torch.amp.GradScaler('cuda') if use_amp else None
        
        if self.privacy_engine:
            client_model, optimizer_instance, self.client_loader = self.privacy_engine.make_private(
                module=client_model,
                optimizer=optimizer_instance,
                data_loader=self.client_loader,
                noise_multiplier=1.1,
                max_grad_norm=10.0,
            )

        initial_params = {key: param.clone().detach() for key, param in client_model.named_parameters()}
        
        for epoch in range(local_epochs):
            self.logger.debug(f"Client {self.client_id} - Local Epoch {epoch + 1}/{local_epochs}")
            for batch_idx, batch in enumerate(self.client_loader):
                # Check if the batch is a dict (images) or a tuple/list (tabular data)
                if isinstance(batch, dict):
                    inputs = batch['image'].to(device)
                    targets = batch['label'].to(device)
                elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    if targets.ndim == 2 and targets.size(1) == 1:
                        targets = targets.squeeze(1)
                    targets = targets.to(device)
                else:
                    raise ValueError("Unexpected batch format. Expected dict with keys ('image','label') or tuple (features, labels).")
                    
                optimizer_instance.zero_grad(set_to_none=True)

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = client_model(inputs)
                        loss = criterion(outputs, targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer_instance)
                    scaler.update()
                else:
                    outputs = client_model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer_instance.step()
        
        final_params = {key: param.clone().detach() for key, param in client_model.named_parameters()}
        param_differences = {
            key: (final_params[key] - initial_params[key]) for key in final_params.keys()
        }
        del initial_params, final_params
        torch.cuda.empty_cache()
        gradient_like_diffs = torch.cat([diff.view(-1) for diff in param_differences.values()]).unsqueeze(1)

        return {
            'flattened_diffs': gradient_like_diffs,
            'data_size': len(self.client_loader.dataset)        
        }


class FedProxClient(Client):
    """
    Client class for FedProx: Federated Learning with a Proximal term.
    It extends the Client class and overrides the training process to include
    a proximal term that restricts the deviation of local model weights from the global model.
    """

    def __init__(self, client_id: int, client_loader, dp: bool = False, fedprox_mu: float = 0.01):
        super().__init__(client_id, client_loader, dp)
        self.mu = fedprox_mu  

    def train(
        self, 
        global_model: nn.Module, 
        local_epochs: int, 
        learning_rate: float, 
        device: str,
        optimizer: str = 'SGD',
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        mu: float = 0.01
    ) -> Dict[str, torch.Tensor]:
        """
        Train with FedProx algorithm including proximal term.
        
        Args:
            global_model (nn.Module): The global model to be trained.
            local_epochs (int): Number of local epochs for training.
            learning_rate (float): Learning rate for the optimizer.
            device (str): Device to run the training on.
            optimizer (str): Type of optimizer ('SGD', 'Adam', 'AdamW').
            momentum (float): Momentum factor (0.0 means no momentum for SGD).
            weight_decay: Weight decay for regularization.
            mu (float): FedProx regularization parameter.
            
        Returns:
            Dict[str, torch.Tensor]: The updated model parameters after training.
        """
        
            
        # Save a copy of the global model parameters to serve as a reference
        global_params = {name: param.clone().detach() for name, param in global_model.named_parameters()}

        client_model = global_model.to(device)
        client_model.train()
        
        # Create optimizer with proper configuration
        optimizer_instance = self._create_optimizer(
            client_model.parameters(), 
            optimizer, 
            learning_rate, 
            momentum, 
            weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()

        use_amp = device == 'cuda'
        scaler = torch.amp.GradScaler('cuda') if use_amp else None

        if self.privacy_engine:
            client_model, optimizer_instance, self.client_loader = self.privacy_engine.make_private(
                module=client_model,
                optimizer=optimizer_instance,
                data_loader=self.client_loader,
                noise_multiplier=1.1,
                max_grad_norm=10.0,
            )

        for epoch in range(local_epochs):
            self.logger.debug(f"Client {self.client_id} - Local Epoch {epoch + 1}/{local_epochs} (FedProx)")
            for batch_idx, batch in enumerate(self.client_loader):
                if isinstance(batch, dict):
                    inputs = batch['image'].to(device)
                    targets = batch['label'].to(device)
                elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    if targets.ndim == 2 and targets.size(1) == 1:
                        targets = targets.squeeze(1)
                    targets = targets.to(device)
                else:
                    raise ValueError("Unexpected batch format. Expected dict or tuple.")

                optimizer_instance.zero_grad(set_to_none=True)
                
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = client_model(inputs)
                        loss = criterion(outputs, targets)
                        # Add proximal term: restrict local model from straying too far from global model.
                        prox_term = sum(((param - global_params[name])**2).sum()
                                        for name, param in client_model.named_parameters())
                        loss += 0.5 * mu * prox_term
                    scaler.scale(loss).backward()
                    scaler.step(optimizer_instance)
                    scaler.update()
                else:
                    outputs = client_model(inputs)
                    loss = criterion(outputs, targets)
                    prox_term = sum(((param - global_params[name])**2).sum()
                                    for name, param in client_model.named_parameters())
                    loss += 0.5 * mu * prox_term
                    loss.backward()
                    optimizer_instance.step()

        
        final_params = {name: param.clone().detach() for name, param in client_model.named_parameters()}
        param_differences = {
            name: final_params[name] - global_params[name]
            for name in final_params
        }
        torch.cuda.empty_cache()
        flattened_diffs = torch.cat([diff.view(-1) for diff in param_differences.values()]).unsqueeze(1)
        return {
            'flattened_diffs': flattened_diffs,
            'data_size': len(self.client_loader.dataset)
        }