import os
import pandas as pd
from dotenv import load_dotenv
from pymongo.errors import PyMongoError
from langchain_aws import BedrockEmbeddings
from langchain.docstore.document import Document
from langchain_mongodb import MongoDBAtlasVectorSearch


load_dotenv()

def insert_csv_with_embeddings(csv_file: str, collection):
    """
    Reads a CSV, generates embeddings, and inserts documents into MongoDB.
    Handles errors with try/except.
    """
    if collection is None:
        print("⚠️ Skipping insert because MongoDB connection failed.")
        return

    try:
        embeddings = BedrockEmbeddings(
                model_id= os.getenv("EMBEDDING_MODEL_ID"),
                region_name= os.getenv("AWS_DEFAULT_REGION"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name="vector_index",
            relevance_score_fn="cosine",
        )
        # Load CSV
        df = pd.read_csv(csv_file)
        docs = []
        for _, row in df.iterrows():
            text_to_embed = generate_offer_string(row)
            emi = 1 if row['emi'].strip()=='y' else 0
            metadata = {
                "platform":row['platform'].strip(),
                "title":row['title'].strip(),
                "offer":row['offer'].strip(),
                "coupon_code":row['coupon_code'].strip(),
                "bank":row["bank"].strip(),
                "payment_mode":row["payment_mode"].strip(),
                "emi":emi,
                "url":row['url'].strip(),
                "flight_type":row['flight_type'].strip(),
            }

            docs.append(Document(page_content=text_to_embed, metadata=metadata))

            
        # Insert into MongoDB
        vector_store.add_documents(documents=docs)
        # collection.insert_many(docs)
        print(f"✅ Inserted {len(docs)} documents into MongoDB.")

    except FileNotFoundError:
        print(f"❌ CSV file '{csv_file}' not found.")
    except pd.errors.EmptyDataError:
        print(f"❌ CSV file '{csv_file}' is empty.")
    except PyMongoError as e:
        print("❌ Failed to insert documents into MongoDB.")
        print("Error:", e)
    except Exception as e:
        print("❌ An unexpected error occurred.")
        print("Error:", e)



def generate_offer_string(row):
    """
    Generate a descriptive, human-friendly embedding string using
    platform, title, offers, bank, payment_mode, emi, and flight_type.
    Optional fields are handled gracefully.
    """
    platform = row.get("platform", "").strip()
    title = row.get("title", "").strip()
    offers = row.get("offer", "").strip()
    bank = row.get("bank", "").strip()
    payment_mode = row.get("payment_mode", "").strip()
    emi = row.get("emi", "").strip()
    flight_type = row.get("flight_type", "").strip()

    # Optional payment info
    payment_info = ""
    if bank and payment_mode:
        payment_info = f" for customers using {bank} with {payment_mode}"
    elif bank:
        payment_info = f" for customers using {bank}"
    elif payment_mode:
        payment_info = f" for customers with {payment_mode}"

    # Optional EMI info
    emi_info = " EMI options are available" if emi.lower() == 'y' else ""

    # Smooth descriptive sentence
    offer_string = (
        f"Take advantage of '{title}' on {platform}, which provides {offers}{payment_info}.{emi_info}. "
        f"This exclusive deal is valid for {flight_type} flights and lets you save while traveling comfortably."
    )

    return offer_string
