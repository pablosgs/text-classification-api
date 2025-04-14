import re
from tqdm import tqdm
from typing import Optional, Set, List
import unicodedata

class Preprocessing:
    def __init__(
            self,
            remove_special: bool = True,
    ):
        """
        remove_numbers: eliminar o no números del texto
        remove_special: eliminar o no caracteres especiales
        pos_to_remove: lista de POS tags para eliminar
        lemmatize: aplicar o no lematización
        """
        self._remove_special = remove_special

    def preprocess_text(self, text: str) -> str:
        """Preprocesa el texto dependiendo del idioma detectado."""

        text = str(text)
        text = re.sub(r"[\"']", "", text)
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_dates(text)
        text = self.remove_accents(text)

        return self.__clean(text)

    def preprocess_text_list(self, texts: List[str]) -> List[str]:
        """Preprocesa una lista de textos."""
        clean_texts = []
        for text in tqdm(texts):
            clean_texts.append(self.preprocess_text(text))
        return clean_texts

    def remove_dates(self, text: str) -> str:
        """Elimina fechas del texto en formatos como YYYY-MM-DD o YYYY/MM/DD."""
        date_pattern = re.compile(r'\b\d{4}[-/]\d{2}[-/]\d{2}\b')
        return date_pattern.sub(r'', text)
    

    def remove_urls(self, text: str) -> str:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    
    def remove_emails(self, text: str) -> str:
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        return email_pattern.sub(r'', text)

    def remove_multiple_spaces(self, text: str) -> str:
        return re.sub(' +', ' ', text)

    def remove_leading_trailing_spaces(self, text: str) -> str:
        return text.strip()

    def remove_accents(self, text: str) -> str:
        """Elimina acentos del texto."""
        return ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )

    def __clean(self, text) -> str:

        if self._remove_special:
            text = re.sub(r"[^a-zA-ZáéíóúñüÁÉÍÓÚÑÜ']", " ", text)

        text = text.lower()
        text = self.remove_multiple_spaces(text)
        text = self.remove_leading_trailing_spaces(text)

        return text


if __name__ == "__main__":
    preprocessor = Preprocessing(
        remove_numbers=False,
    )
    texts = [
        "¡Hola, mundo! Este es un ejemplo en español.",
        "Hello world! This is an example in English."
    ]
    clean_texts = preprocessor.preprocess_text_list(texts)
    for original, clean in zip(texts, clean_texts):
        print(f"Original: {original}\nPreprocessed: {clean}\n")
