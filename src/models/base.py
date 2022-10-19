# models For embedding text & images
import clip
import torch
from abc import abstractmethod


class BaseModel():
    @abstractmethod
    def encode_text(self):
        raise NotImplementedError

    @abstractmethod
    def encode_image(self):
        raise NotImplementedError