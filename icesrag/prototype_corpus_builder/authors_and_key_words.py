import pandas as pd
from sqlite3 import connect
import json

def clean_keyword(keyword):
    keyword = keyword.title().replace("-", " ").replace("\n", " ")
    while "  " in keyword:
        keyword = keyword.replace("  ", " ")
    return keyword

# Load data
conn = connect("./ices.db")
data = pd.read_sql("SELECT * FROM abstracts", conn)

# Transform authors and keywords to lists
data['authors'] = data['authors'].apply(lambda x: json.loads(x))
data['keywords'] = data['keywords'].apply(lambda x: json.loads(x))

# Replace "Not Found" with None
data['authors'] = data['authors'].apply(lambda x: x if isinstance(x, str) == False else [])
data['keywords'] = data['keywords'].apply(lambda x: x if isinstance(x, str) == False else [])


# Build new dataframes
authors_paper_id = []
authors_name = []

keywords_paper_id = []
keywords_word = []

# Need to replace \n with a space
for index, row in data.iterrows():
    for author in row['authors']:
        authors_paper_id.append(row['id'])
        authors_name.append(author)
    for keyword in row['keywords']:
        keywords_paper_id.append(row['id'])
        keywords_word.append(clean_keyword(keyword))

authors_df = pd.DataFrame({'paper_id': authors_paper_id, 'name': authors_name})
keywords_df = pd.DataFrame({'paper_id': keywords_paper_id, 'keyword': keywords_word})

# Push to database
authors_df.to_sql("authors", conn, if_exists='replace', index=False)
keywords_df.to_sql("keywords", conn, if_exists='replace', index=False)

# Close connection
conn.close()

