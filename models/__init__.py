from typing import Any
import pytorch_lightning as pl
from torch.nn import Module
class BrainModel(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
class PartialModel(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
class BrainEncoder(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
class BrainQuantizer(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
class BrainDecider(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
