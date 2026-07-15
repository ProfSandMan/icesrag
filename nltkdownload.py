import nltk

# Download all required NLTK resources for the application
resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']

for resource in resources:
    print(f"Downloading {resource}...")
    nltk.download(resource, download_dir='nltk_data_local')
    print(f"Downloaded {resource}")
