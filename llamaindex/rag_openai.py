from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import os
# import logging
# import sys

os.environ["OPENAI_API_KEY"] = "<openapi-api-key>"

'''
# # enable DEBUG / INFO logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
'''

# load data and build an index.
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Query data.
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)


