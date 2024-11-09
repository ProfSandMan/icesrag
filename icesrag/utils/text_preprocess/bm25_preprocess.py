from typing import List
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from icesrag.utils.text_preprocess.strategy_pattern import TextPreprocessingStrategy

# Prerequirement!
# nltk.download('punkt')  # For word tokenization
# nltk.download('stopwords')  # For stopwords list
# nltk.download('wordnet')  # For lemmatization

class BM25PreProcess(TextPreprocessingStrategy):

    def preprocess(self, text: str, **kwargs) -> str:
        # Step 1: Convert all words to lowercase
        text = text.lower()
        
        # Step 2: Remove all punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        
        # Step 3: Tokenize the text into words
        words = word_tokenize(text)
        
        # Step 4: Remove stop words
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        
        # Step 5: Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        
        # Return the processed text as a single string
        return ' '.join(lemmatized_words)
    
    def batch_preprocess(self, texts: List[str], **kwargs) -> List[str]:
        
        processed = []
        for t in texts:
            processed.append(self.preprocess_batch(t))
        return processed
    


    
import nltk

# Manually add the NLTK data path
nltk.data.path.append('C:\\Users\\sandidgeh\\AppData\\Roaming\\nltk_data')

PP = BM25PreProcess()
print(PP.preprocess(text="The very happy cow jumped over the delicious cheese flavored moon."))