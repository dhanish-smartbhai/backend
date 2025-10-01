import os
from utils import create_vector_store
from dotenv import load_dotenv
from pymongo import MongoClient, errors

load_dotenv()

def connect_db():
    """
    Connects to MongoDB Atlas and returns a client.
    Handles errors and prints connection status.
    """
    uri = os.getenv("MONGO_DB_URI")

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")  # test connection
        print("✅ Successfully connected to MongoDB Atlas")
        return client
    except errors.ServerSelectionTimeoutError as e:
        print("❌ Connection timed out. Check your URI and internet connection.")
        print("Error:", e)
        return None
    except Exception as e:
        print("❌ Failed to connect to MongoDB Atlas.")
        print("Error:", e)
        return None


def get_collection(client, collection: str):
    if client is None:
        print("⚠️ No MongoDB client available. Returning None.")
        return None
    
    db = client[os.getenv("DB_NAME")]
    collection = db[collection]
    return collection


def insert_vector_data(collection:str, csv_file:str):
    mongo_client = connect_db()
    collection = get_collection(mongo_client, collection)
    create_vector_store.insert_csv_with_embeddings(csv_file, collection)