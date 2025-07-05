from .train_and_evaluate import main as train_and_evaluate
from .train_bert import main as train_bert
from .train_sbert import main as train_sbert
from .train_word2vec import main as train_word2vec

__all__ = ['train_and_evaluate', 'train_bert', 'train_sbert', 'train_word2vec'] 