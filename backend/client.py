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

    def __init__(self, client_id: int, client_loader: DataLoader , dp: bool = False):
        """
        Initialize the Client.

        Args:
            client_id (int): Unique identifier for the client.
            client_loader (DataLoader): DataLoader for the client's dataset.
        """
        self.client_id = client_id
        self.client_loader = client_loader
        self.privacy_engine = PrivacyEngine() if dp else None
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Client {self.client_id} initialized.")

    def train(
        self,
        global_model: Module,
        local_epochs: int,
        learning_rate: float,
        device: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Train the global model on the client's local data.

        Args:
            global_model (Module): The global model to be trained.
            local_epochs (int): Number of local epochs for training.
            learning_rate (float): Learning rate for the optimizer.
            device (str): Device to run the training on ('cpu' or 'cuda').

        Returns:
            Dict[str, torch.Tensor]: The updated model parameters after training.
        """
        self.logger.info(f"Client {self.client_id} started training.")
        

        # Create a new model and load global state
        client_model = copy.deepcopy(global_model).to(device)
        client_model.load_state_dict(global_model.state_dict())
        
        
        client_model.train()
        # Define optimizer and loss function
        optimizer = optim.SGD(client_model.parameters(), lr=learning_rate , momentum=0.9) 
        #optimizer = optim.Adam(client_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Conditional AMP setup
        use_amp = device == 'cuda'
        scaler = torch.amp.GradScaler('cuda') if use_amp else None
        
        if self.privacy_engine:
            client_model, optimizer, self.client_loader = self.privacy_engine.make_private(
                module=client_model,
                optimizer=optimizer,
                data_loader=self.client_loader,
                noise_multiplier=1.1,
                max_grad_norm=10.0,
            )

        # Clone initial parameters for computing updates
        initial_params = {key: param.clone() for key, param in client_model.named_parameters()}
        
        for epoch in range(local_epochs):
            self.logger.info(f"Client {self.client_id} - Local Epoch {epoch + 1}/{local_epochs}")
            for batch_idx, batch in enumerate(self.client_loader):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = client_model(images)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = client_model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
        
        final_params = {key: param.clone() for key, param in client_model.named_parameters()}
        param_differences = {
            key: (final_params[key] - initial_params[key]) for key in final_params.keys()
        }

        gradient_like_diffs = torch.cat([diff.view(-1) for diff in param_differences.values()]).unsqueeze(1)

        
        return {
            'flattened_diffs': gradient_like_diffs,
            'data_size': len(self.client_loader.dataset)        
        }