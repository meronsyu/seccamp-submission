# Install required libraries
!pip install datasets huggingface_hub sentence_transformers lancedb

from datasets import load_dataset
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from lancedb import lancedb
from lancedb.rerankers import ColbertReranker
from transformers import AutoModelForSeq2SeqLM

# Load dataset
dataset = load_dataset("jamescalam/ai-arxiv-chunked", split="train")

# Tokenize and chunk documents
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def chunk_text(text, chunk_size=512, overlap=64):
   tokens = tokenizer.encode(text, return_tensors="pt", truncation=True)
   chunks = tokens.split(chunk_size - overlap)
   texts = [tokenizer.decode(chunk) for chunk in chunks]
   return texts

chunked_data = []
for doc in dataset:
   text = doc["chunk"]
   chunked_texts = chunk_text(text)
   chunked_data.extend(chunked_texts)

# Load Sentence Transformer model and create LanceDB vector store
model = SentenceTransformer('all-MiniLM-L6-v2')
db = lancedb.lancedb('/path/to/store')
db.create_collection('docs', vector_dimension=model.get_sentence_embedding_dimension())

# Index documents
for text in chunked_data:
   vector = model.encode(text).tolist()
   db.insert_document('docs', vector, text)

# Perform initial retrieval (not shown in the snippets)
initial_docs = ...

# Rerank initial documents
reranker = ColbertReranker()
reranked_docs = reranker.rerank(query, initial_docs)

# Augment query with reranked documents and generate response
tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

augmented_query = query + " " + " ".join(reranked_docs[:3])

input_ids = tokenizer.encode(augmented_query, return_tensors="pt")
output_ids = model.generate(input_ids, max_length=500)
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)
