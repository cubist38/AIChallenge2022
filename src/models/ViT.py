from src.models.base import *

class VIT(BaseModel):
    def __init__(self):
        self.model, preprocess = clip.load("ViT-B/16", device="cpu")

        self.model.eval()
        self.input_resolution = self.model.visual.input_resolution
        self.context_length = self.model.context_length
        self.vocab_size = self.model.vocab_size

    def encode_text(self, text):
        text_tokens = clip.tokenize([text])

        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens).float()

        return text_features

