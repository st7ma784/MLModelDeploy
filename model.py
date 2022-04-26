from cog import BasePredictor, Input, Path
import torch

import os, torch, datasets
from typing import List
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint,LearningRateMonitor
from torch.nn import CrossEntropyLoss
from sklearn.metrics import classification_report
from pytorch_lightning import seed_everything
from transformers import AutoModelForTokenClassification,AutoConfig,AutoTokenizer
config_class, model_class, tokenizer_class = AutoConfig, AutoModelForTokenClassification, AutoTokenizer         
from datamod import get_labels,MODEL_CLASSES
from datetime import datetime
from typing import Optional

class HuggingFace(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        labels: List[str] = None,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        model_type:str ="bert",
        eval_output_file:str = None,
        frompretrained=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pad_token_label_id=CrossEntropyLoss().ignore_index
        self.config_class, self.model_class = AutoConfig, AutoModelForTokenClassification#MODEL_CLASSES[self.model_type]
        if frompretrained is not None:
            self.config_class, self.model_class,_ = MODEL_CLASSES[model_type]
            model_name_or_path=frompretrained#set model name to be the pretrained model dir
        self.config = self.config_class.from_pretrained(
          model_name_or_path,
          num_labels=len(labels),
          id2label={i: label for i, label in enumerate(labels)},
          label2id= {label: i for i, label in enumerate(labels)},)
        self.model = self.model_class.from_pretrained(
          model_name_or_path,
          from_tf=model_name_or_path.endswith(".ckpt"),
          config=self.config)
        self.eval_output_file = eval_output_file or os.path.join(os.getcwd(), "eval_results.txt")
        self.metric = datasets.load_metric("seqeval")
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.hparams.weight_decay,},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, } ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        '''At point of coding, schedulers are being rewritten in PL and need manually stepping '''
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.hparams.warmup_steps,
        #     num_training_steps=self.hparams.max_steps,
        # )
        #scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer]#, [scheduler]
    def forward(self, batch):
        return self.model(**batch)
    def training_step(self, batch, batch_idx):
        outputs= self(batch)
        loss = outputs[0]
        return loss


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = torch.load("./weights.pth")

    # The arguments and types the model takes as input
    def predict(self,
          input: Path = Input(title="Grayscale input image")
    ) -> Path:
        """Run a single prediction on the model"""
        processed_input = preprocess(input)
        output = self.model(processed_input)
        return postprocess(output)
