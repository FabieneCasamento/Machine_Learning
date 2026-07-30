# %% [markdown]
# # Step 1: Setup prerequisites
# Aula MongoDay
#https://play.instruqt.com/mongo-devrel/invite/g64hrvyj7qs2/tracks/ai-labs-vector-search/challenges/vector-search-lab/assignment#tab-0
# vector_search_lab

# %% [markdown]
# # Step 1: Setup prerequisites

# %%
import os
import sys
from pymongo import MongoClient

# Add parent directory to path to import from utils
sys.path.append(os.path.join(os.path.dirname(os.getcwd())))
from utils import set_env

# %%
# If you are using your own MongoDB Atlas cluster, use the connection string for your cluster here
MONGODB_URI = os.getenv("MONGODB_URI")
# Initialize a MongoDB Python client
mongodb_client = MongoClient(MONGODB_URI, appname="devrel-workshop-vector-search")
# Check the connection to the server
mongodb_client.admin.command("ping")

# %%
# Set the passkey provided by your workshop instructor
PASSKEY = "ynMkXg"

# %%
# Obtain a Voyage API key from our AI model proxy and set it as an environment variable-- DO NOT CHANGE
set_env(["voyageai"], PASSKEY)

# %% [markdown]
# # Step 2: Import data into MongoDB

# %%
import json

# %% [markdown]
# ### **Do not change the values assigned to the variables below**

# %%
# Database name
DB_NAME = "mongodb_genai_devday_vs"
# Collection name
COLLECTION_NAME = "books"
# Name of the vector search index
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

# %%
# Connect to the `COLLECTION_NAME` collection.
collection = mongodb_client[DB_NAME][COLLECTION_NAME]

# %%
with open("../data/books.json", "r") as data_file:
    json_data = data_file.read()

data = json.loads(json_data)

print(f"Deleting existing documents from the {COLLECTION_NAME} collection.")
collection.delete_many({})
collection.insert_many(data)
print(
    f"{collection.count_documents({})} documents ingested into the {COLLECTION_NAME} collection."
)

# %% [markdown]
# # Step 3: Generating embeddings

# %%
from PIL import Image
import requests
import voyageai

# %%
# Initialize the Voyage AI client
vo = voyageai.Client()

# %% [markdown]
# ### For images
#
# 📚 https://www.mongodb.com/docs/voyageai/models/multimodal-embeddings/?client=python#example

# %%
image_url = "https://images.isbndb.com/covers/4318463482198.jpg"
# Load the image from the URL above
image = Image.open(requests.get(image_url, stream=True).raw)
# Use the `multimodal_embed` method of the Voyage API with the following arguments to embed the image:
# inputs: The image wrapped in a list of lists
# model: `voyage-multimodal-3.5`
#embedding = <CODE_BLOCK_1>

embedding =vo.multimodal_embed(inputs=[[image]], model="voyage-multimodal-3.5")



# %%
# Get the embeddings as a list from the `embedding` object
#<CODE_BLOCK_2>
embedding.embeddings[0]



# %% [markdown]
# ### For text

# %%
text = "Puppy Preschool: Raising Your Puppy Right---Right from the Start!"
# Use the `multimodal_embed` method to embed a piece of text
embedding = vo.multimodal_embed(inputs=[[text]], model="voyage-multimodal-3.5")

# %%
# Get the embeddings as a list from the `embedding` object
embedding.embeddings[0]

# %% [markdown]
# # Step 4: Adding embeddings to existing data in Atlas

# %%
# You might see a warning after running this cell-- You can ignore it
from typing import List, Dict, Optional
from tqdm import tqdm

# %%
# Field in the documents to embed-- in this case, the book cover
field_to_embed = "cover"
# Name of the embedding field to add to the documents
embedding_field = "embedding"

# %%
# Define a function to generate multimodal embeddings using the Voyage API
def get_embeddings(content: str, mode: str, input_type: str) -> List[float]:
    """
    Generate embeddings

    Args:
        content (str): Content to embed
        mode (str): Content mode (Can be one of "image" or "text")
        input_type (str): Type of input, either "document" or "query"

    Returns:
        List[float]: Embedding of the content as a list.
    """
    # If the input is an image, first load the image content
    if mode == "image":
        if content.startswith("http"):
            content = Image.open(requests.get(content, stream=True).raw)
        else:
            content = Image.open(content)
    return vo.multimodal_embed(inputs=[[content]], model="voyage-multimodal-3.5", input_type=input_type).embeddings[0]

# %% [markdown]
# 📚 https://www.mongodb.com/docs/manual/tutorial/query-documents/#select-all-documents-in-a-collection

# %%
# Query for all documents in the `collection` collection
#results = <CODE_BLOCK_3>

results= collection.find({})



# %% [markdown]
# 📚 **$set:** https://www.mongodb.com/docs/manual/reference/operator/update/set/#syntax
#
# 📚 **update_one():** https://pymongo.readthedocs.io/en/stable/api/pymongo/collection.html#pymongo.collection.Collection.update_one

# %%
result

# %%
# Update each document in the `collection` collection with embeddings
for result in tqdm(results):
    content = result[field_to_embed]
    #content = result
    # Use the `get_embeddings` function defined above to embed the `content`
    # Note that `content` is the cover image of the book, so set the `mode` accordingly
    # `input_type` should be set to "document" since we are embedding the "documents" we want to search
    embedding = get_embeddings(content, "image", "document")
    # Filter for the document where the `_id` field is equal to the `_id` of the current document
    filter = {"_id": result["_id"]}
    # Set the `embedding_field` field to the value `embedding` using the `$set` operator
    update = {"$set": {embedding_field: embedding}}
    # Update the documents in the `collection` collection inplace using the `update_one()` operation
    # Get the right document `_id` using the `filter` and apply the `update`
    collection.update_one(filter, update)

# %% [markdown]
# # Step 5: Create a vector search index

# %%
from utils import create_search_index, check_index_ready

# %%
# Create vector index definition specifying:
# path: Path to the embeddings field
# numDimensions: Number of embedding dimensions- depends on the embedding model used
# similarity: Similarity metric. One of cosine, euclidean, dotProduct.
model = {
    "name": ATLAS_VECTOR_SEARCH_INDEX_NAME,
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 1024,
                "similarity": "cosine",
            }
        ]
    },
}

# %%
# Use the `create_search_index` function from the `utils` module to create a vector search index with the above definition for the `collection` collection
create_search_index(collection, ATLAS_VECTOR_SEARCH_INDEX_NAME, model)

# %%
# Use the `check_index_ready` function from the `utils` module to verify that the index was created and is in READY status before proceeding
check_index_ready(collection, ATLAS_VECTOR_SEARCH_INDEX_NAME)

# %% [markdown]
# # Step 6: Perform vector search queries
#
# 📚 https://www.mongodb.com/docs/vector-search/query/aggregation-stages/vector-search-stage/?deployment-type=atlas&embedding=byo&interface=driver&language=python#simple-query-5

# %%
# Define a function to retrieve relevant documents for a user query using vector search
def vector_search(
    user_query: str, mode: str, filter: Optional[Dict] = {}
) -> None:
    """
    Retrieve relevant documents for a user query using vector search.

    Args:
    user_query (str): The user's query (can be a piece of text or a link to an image)
    mode (str): Query mode (image or text)
    filter (Optional[Dict], optional): Optional vector search pre-filter
    """
    # Generate embedding for the `user_query` using the `get_embeddings` function defined in Step 4
    # `input_type` should be set to "query" since we are embedding the query
    query_embedding = get_embeddings(user_query, mode, "query")

    # Define an aggregation pipeline consisting of a $vectorSearch stage, followed by a $project stage
    # Set the number of candidates to 20 and only return the top 5 documents from the vector search
    # Set the `filter` field in the $vectorSearch stage to the value `filter` passed to the function
    # In the $project stage, exclude the `_id` field, include these fields: `title`, `cover`, `year`, `pages`, and the `vectorSearchScore`
    # NOTE: Use variables defined previously for the `index`, `queryVector` and `path` fields in the $vectorSearch stage
    pipeline = [
    {
        "$vectorSearch": {
            "index": ATLAS_VECTOR_SEARCH_INDEX_NAME,
            "queryVector": query_embedding,
            "path": "embedding",
            "numCandidates": 20,
            "filter": filter,
            "limit": 5,
        }
    },
    {"$project": {"_id": 0, "title": 1, "cover": 1, "year":1, "pages":1, "score": {"$meta": "vectorSearchScore"}}},
]

    # Execute the aggregation `pipeline` and store the results in `results`
    results = collection.aggregate(pipeline)

    # Print book title, score, and cover image
    for book in results:
        cover = Image.open(requests.get(book.get("cover"), stream=True).raw).resize((100,150))
        print(f"{book.get('title')}({book.get('year')}, {book.get('pages')} pages): {book.get('score')}")
        display(cover)

# %%
# Test the vector search with a text query
vector_search("A man wearing a golden crown", "text")

# Also try these text queries:
# - A rainbow of lively colors
# - Creatures wondrous or familiar
# - A boy and the ocean
# - Houses

# %%
# Test the vector search with an image query
vector_search("https://images.isbndb.com/covers/10835953482746.jpg", "image")

# Also try these image queries:
# - ../data/images/salad.jpg
# - ../data/images/kitten.png
# - ../data/images/barn.png

# %% [markdown]
# # Step 7: Adding pre-filters to your vector search

# %% [markdown]
# ### Filter for books that were published after the year `2002`
#
# 📚 https://www.mongodb.com/docs/vector-search/index/vector-search-type/#about-the-filter-type

# %%
# Modify the vector search index `model` from Step 5 to include the `year` field as a `filter` field
model = {
    "name": ATLAS_VECTOR_SEARCH_INDEX_NAME,
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 1024,
                "similarity": "cosine",
            },
            {"type": "filter", "path": "year"},
        ]
    },
}

# %%
# Use the `create_search_index` function from the `utils` module to re-create the vector search index with the modified model
create_search_index(collection, ATLAS_VECTOR_SEARCH_INDEX_NAME, model)

# %%
# Use the `check_index_ready` function from the `utils` module to verify that the index has the right filter fields and is in READY status before proceeding
check_index_ready(collection, ATLAS_VECTOR_SEARCH_INDEX_NAME)

# %% [markdown]
# 📚 https://www.mongodb.com/docs/manual/reference/operator/query/gte/#syntax

# %%
# Create a filter definition to filter for books where the `year` field is greater than `2002` using the `$gte` operator
filter = {"year": {"$gte": 2002}}
# Pass the `filter` as an argument to the `vector_search` function.
# Notice how this filter is incorporated in the `pipeline` in the `vector_search` function.
vector_search("A boy and the ocean", "text", filter)

# %% [markdown]
# ### Filter for books that were published after the year `2002` and under `250` pages
#
# 📚 https://www.mongodb.com/docs/vector-search/index/vector-search-type/#about-the-filter-type

# %%
# Modify the vector search index `model` from Step 5 to include `year` and `pages` as filter fields
model = {
    "name": ATLAS_VECTOR_SEARCH_INDEX_NAME,
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 1024,
                "similarity": "cosine",
            },
            {"type": "filter", "path": "year"},
            {"type": "filter", "path": "pages"},
        ]
    },
}

# %%
# Use the `create_search_index` function from the `utils` module to re-create the vector search index with the modified model
create_search_index(collection, ATLAS_VECTOR_SEARCH_INDEX_NAME, model)

# %%
# Use the `check_index_ready` function from the `utils` module to verify that the index has the right filter fields and is in READY status before proceeding
check_index_ready(collection, ATLAS_VECTOR_SEARCH_INDEX_NAME)

# %% [markdown]
# 📚 https://www.mongodb.com/docs/manual/reference/operator/query/lte/#mongodb-query-op.-lte

# %%
# Create a filter definition to filter for books where the `year` field is greater than or equal to `2002` and the `pages` field is less than or equal to 250
# Use the `$gte` and `$lte` operators
filter = {"$and": [{"year": {"$gte": 2002}}, {"pages": {"$lte": 250}}]}
# Pass the `filter` as an argument to the `vector_search` function.
# Notice how this filter is incorporated in the `pipeline` in the `vector_search` function.
vector_search("A boy and the ocean", "text", filter)

# %% [markdown]
# # Step 8: Enable vector quantization
#
# 📚 https://www.mongodb.com/docs/vector-search/index/vector-search-type/?deployment-type=atlas&embedding=byo&interface=driver&language=python#syntax

# %%
# Modify the vector search index `model` from Step 5 to use `scalar` quantization
model = {
    "name": ATLAS_VECTOR_SEARCH_INDEX_NAME,
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 1024,
                "similarity": "cosine",
                "quantization": "scalar",
            },
        ]
    },
}

# %%
# Use the `create_search_index` function from the `utils` module to re-create the vector search index with the modified model
create_search_index(collection, ATLAS_VECTOR_SEARCH_INDEX_NAME, model)

# %%
# Use the `check_index_ready` function from the `utils` module to verify the index was created with quantization enabled
check_index_ready(collection, ATLAS_VECTOR_SEARCH_INDEX_NAME)

# %% [markdown]
# # 🦹‍♀️ Hybrid search

# %%
# Name of the full-text search index
ATLAS_FTS_INDEX_NAME = "fts_index"

# %%
# Create full-text search index definition specifying the field mappings
model = {
    "name": ATLAS_FTS_INDEX_NAME,
    "type": "search",
    "definition": {
        "mappings": {"dynamic": False, "fields": {"synopsis": {"type": "string"}}}
    },
}

# %%
# Use the `create_search_index` function from the `utils` module to create a full-text search index with the above definition for the `collection` collection
create_search_index(collection, ATLAS_FTS_INDEX_NAME, model)

# %%
# Reset the vector search index to the original vector search index definition
model = {
    "name": ATLAS_VECTOR_SEARCH_INDEX_NAME,
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 1024,
                "similarity": "cosine",
            }
        ]
    },
}

# Use the `create_search_index` function from the `utils` module to reset the vector search index to its initial definition.
create_search_index(collection, ATLAS_VECTOR_SEARCH_INDEX_NAME, model)

# %%
# Use the `check_index_ready` function from the `utils` module to verify that both the indexes were created and are in READY status before proceeding
check_index_ready(collection, ATLAS_VECTOR_SEARCH_INDEX_NAME)
check_index_ready(collection, ATLAS_FTS_INDEX_NAME)

# %% [markdown]
# 📚 Refer to our [documentation](https://www.mongodb.com/docs/manual/reference/operator/aggregation/rankFusion/) for a detailed explanation of the hybrid search query below.

# %%
# Define a function to retrieve relevant documents for a user query using hybrid search
def hybrid_search(
    user_query: str, vector_weight: float, full_text_weight: float
) -> None:
    """
    Retrieve relevant documents for a user query using hybrid search.

    Args:
        user_query (str): User query string
        vector_weight (float): Weight of vector search in the final search results
        full_text_weight (float): Weight of full-text search in the final search results
    """
    pipeline = [
        {
            "$rankFusion": {
                "input": {
                    "pipelines": {
                        # Vector search pipeline
                        "vector_pipeline": [
                            {
                                "$vectorSearch": {
                                    "index": ATLAS_VECTOR_SEARCH_INDEX_NAME,
                                    "path": "embedding",
                                    "queryVector": get_embeddings(user_query, "text", "query"),
                                    "numCandidates": 20,
                                    "limit": 10,
                                }
                            }
                        ],
                        # Full-text search pipeline
                        "fts_pipeline": [
                            {
                                "$search": {
                                    "index": ATLAS_FTS_INDEX_NAME,
                                    "text": {"query": user_query, "path": "synopsis"},
                                }
                            },
                            {"$limit": 10},
                        ],
                    }
                },
                # Combining the scores from the vector search and full-text search pipelines
                "combination": {
                    "weights": {"vector_pipeline": vector_weight, "fts_pipeline": full_text_weight}
                },
                "scoreDetails": True,
            }
        },
        {
            "$project": {
                "_id": 0,
                "title": 1,
                "cover": 1,
                "score": {"$getField": {"field": "value", "input": {"$meta": "scoreDetails"}}},
            }
        },
        {"$limit": 5},
    ]

    results = collection.aggregate(pipeline)
    # Print book title, hybrid search score, and cover image
    for book in results:
        cover = Image.open(requests.get(book["cover"], stream=True).raw).resize(
            (100, 150)
        )
        print(
            f"{book.get('title')}, Hybrid Search Score: {book.get('score')}"
        )
        display(cover)

# %%
# Test the hybrid search query with a weight of 1.0 for vector search and 0.0 for full-text search
hybrid_search(
    user_query="My Favorite Summer",
    vector_weight=1.0,
    full_text_weight=0.0,
)

# %%
# Test the hybrid search query with a weight of 0.3 for vector search and 0.7 for full-text search
hybrid_search(
    user_query="My Favorite Summer",
    vector_weight=0.3,
    full_text_weight=0.7,
)

# %%