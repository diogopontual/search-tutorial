import json
from pprint import pprint
import os
import time


from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME="my_documents"
class Search:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.es = Elasticsearch('http://localhost:9200', basic_auth=('elastic', 'changeme'))
        client_info = self.es.info()
        print('Connected to Elasticsearch')
        pprint(client_info.body)
        
    def create_index(self):
        self.es.indices.delete(index=INDEX_NAME, ignore_unavailable=True)
        self.es.indices.create(index=INDEX_NAME, mappings={
            'properties': {
                'embedding': {'type': 'dense_vector'}
            }
        })
        print(f'Index {INDEX_NAME} created')
        
    def get_embedding(self, text):
        return self.model.encode(text)
    
    def insert_document(self, doc):
        return self.es.index(index=INDEX_NAME, document={
            **doc,
            'embedding': self.get_embedding(doc['summary'])
        })
        print(f'Document {doc} inserted')
        
        
    def insert_documents(self, docs):
        operations = []
        for doc in docs:
            operations.append({'index': {'_index': INDEX_NAME}})
            operations.append({
            **doc,
            'embedding': self.get_embedding(doc['summary'])
        })
        return self.es.bulk(operations=operations)
    
    def reindex(self):
        self.create_index()
        with open('data.json', 'rt') as f:
            docs = json.loads(f.read())
        return self.insert_documents(docs)
    
    def search(self, **query_args):
        return self.es.search(index=INDEX_NAME, **query_args)
    
    def retrieve_document(self, id):
        return self.es.get(index=INDEX_NAME, id=id)