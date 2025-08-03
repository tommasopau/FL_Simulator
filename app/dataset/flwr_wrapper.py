from flwr_datasets.federated_dataset import FederatedDataset

class CustomFederatedDataset(FederatedDataset):
    def __init__(self, dataset, partitioners, **kwargs):
        # Call parent with a dummy dataset name (won't be used)
        dummy_dataset_name = "dummy"
        super().__init__(dataset=dummy_dataset_name, partitioners=partitioners, **kwargs)
        self._dataset = dataset  # Use your pre-loaded dataset
        self._dataset_prepared = True  # Mark as prepared

    def _prepare_dataset(self) -> None:
        # Override to skip downloading/processing since we already have the dataset.
        pass