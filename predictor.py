
from DataModule import Preprocessor
from model.py import GLUETransformer
class Predictor(BasePredictor):
      glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        model_name_or_path="albert-base-v2"
        task_name="qnli"
        self.model = GLUETransformer(
            model_name_or_path=model_name_or_path,
            num_labels=self.glue_task_num_labels[task_name],
            task_name=task_name,
        )
        self.preprocessor=Preprocessor(model_name_or_path)
    
    # The arguments and types the model takes as input
    
    def predict(self,
          input: Path = Input(title="Grayscale input image")
    ) -> Path:
        """Run a single prediction on the model"""
        processed_input = self.preprocessor.preprocess(
                batched=True,
                remove_columns=["label"],
            )(input)
        output = self.model(processed_input)
        return self.preprocessor.postprocess(output)
