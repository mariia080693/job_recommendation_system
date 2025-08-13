import re
from nltk.stem import PorterStemmer
import pandas as pd
stemmer = PorterStemmer()

def clean_text(text): #tokenize text
    if not text or pd.isna(text):
        return ""
    
    # Combine values from dict Metadata into string except for the 'additionalSalaryText' values
    if isinstance(text, dict):
        text = ' '.join(str(value) for key, value in text.items() if key != 'additionalSalaryText' and value is not None)
    
    text = text.replace('IT', 'specialtokenIT')
    
    # Convert to lowercase
    text = text.lower()
    
    # Protect special terms (more items can be added as needed)
    text = text.replace('c++', 'specialtokencplusplus')
    text = text.replace('c#', 'specialtokencsharp')
    text = text.replace('r&d', 'specialtokenrandd')
    text = text.replace('.net', 'specialtokendotnet')
    text = text.replace('r', 'specialtokenR')
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text, flags=re.DOTALL)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Stem the text
    tokens = text.split()
    stems = [stemmer.stem(token) for token in tokens]
    return ' '.join(stems)

if __name__ == "__main__":
    sample_text = "This is a sample text with <b>HTML</b> tags and some punctuation! C++ and C# are programming languages. R&D is important. .NET is a framework."
    cleaned_text = clean_text(sample_text)
    print(cleaned_text)