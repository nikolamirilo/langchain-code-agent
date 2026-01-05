"""
Data Pipeline for CSV Processing and Supabase Vector Storage

This module processes CSV files from the uploads folder:
1. Reads each CSV file
2. Converts each row to a document chunk (with headers)
3. Generates metadata from column key-value pairs
4. Embeds using OpenAI text-embedding-3-small
5. Stores in Supabase PostgreSQL with pgvector
"""

import os
import glob
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Configuration
UPLOADS_FOLDER = Path(__file__).parent.parent / "uploads"
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TABLE_NAME = "documents"
EMBEDDING_MODEL = "text-embedding-3-small"


def get_supabase_client() -> Client:
    """Create and return a Supabase client."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def get_embeddings() -> OpenAIEmbeddings:
    """Create and return OpenAI embeddings instance."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY must be set in .env")
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY  # type: ignore
    )


def csv_row_to_document(
    row: pd.Series,
    headers: List[str],
    source_file: str,
    row_index: int
) -> Document:
    """
    Convert a CSV row to a LangChain Document with metadata.
    
    Args:
        row: A pandas Series representing a single CSV row
        headers: List of column headers
        source_file: Name of the source CSV file
        row_index: Index of the row in the original file
        
    Returns:
        A Document with page_content containing the row as text
        and metadata containing all column key-value pairs
    """
    # Create text content with headers (CSV-like format)
    content_parts = []
    for header in headers:
        value = row[header]
        content_parts.append(f"{header}: {value}")
    
    page_content = "\n".join(content_parts)
    
    # Create metadata with all column values
    metadata: Dict[str, Any] = {
        "_source_file": source_file,
        "_row_index": row_index,
    }
    
    # Add each column as metadata for filtering
    for header in headers:
        value = row[header]
        # Convert to appropriate type
        if pd.isna(value):
            metadata[header] = None
        elif isinstance(value, (int, float)):
            metadata[header] = value
        else:
            metadata[header] = str(value)
    
    return Document(page_content=page_content, metadata=metadata)


def process_csv_file(file_path: Path) -> List[Document]:
    """
    Process a single CSV file and return list of documents.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of Document objects, one per row
    """
    print(f"Processing: {file_path.name}")
    
    df = pd.read_csv(file_path)
    headers = df.columns.tolist()
    documents = []
    
    for idx, row in df.iterrows():
        doc = csv_row_to_document(
            row=row,
            headers=headers,
            source_file=file_path.name,
            row_index=int(idx)  # type: ignore
        )
        documents.append(doc)
    
    print(f"  → Created {len(documents)} documents from {file_path.name}")
    return documents


def store_documents_in_supabase(documents: List[Document]) -> int:
    """
    Embed documents and store them in Supabase.
    
    Args:
        documents: List of Document objects to store
        
    Returns:
        Number of documents successfully stored
    """
    if not documents:
        return 0
    
    supabase = get_supabase_client()
    embeddings = get_embeddings()
    
    # Get embeddings for all documents
    texts = [doc.page_content for doc in documents]
    print(f"Generating embeddings for {len(texts)} documents...")
    vectors = embeddings.embed_documents(texts)
    
    # Prepare records for insertion
    records = []
    for doc, vector in zip(documents, vectors):
        records.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "embedding": vector
        })
    
    # Insert into Supabase
    print(f"Inserting {len(records)} records into Supabase...")
    
    # Insert in batches of 100 to avoid payload limits
    batch_size = 100
    inserted = 0
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        result = supabase.table(TABLE_NAME).insert(batch).execute()
        inserted += len(batch)
        print(f"  → Inserted batch {i // batch_size + 1}: {len(batch)} records")
    
    return inserted


def process_all_csv_files(folder_path: Optional[Path] = None) -> int:
    """
    Process all CSV files in the uploads folder.
    
    Args:
        folder_path: Optional custom folder path (defaults to uploads/)
        
    Returns:
        Total number of documents processed
    """
    folder = folder_path or UPLOADS_FOLDER
    
    # Ensure folder exists
    folder.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files
    csv_files = list(folder.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {folder}")
        return 0
    
    print(f"Found {len(csv_files)} CSV file(s) in {folder}")
    print("-" * 50)
    
    all_documents = []
    
    for csv_file in csv_files:
        try:
            docs = process_csv_file(csv_file)
            all_documents.extend(docs)
        except Exception as e:
            print(f"  ✗ Error processing {csv_file.name}: {e}")
    
    if not all_documents:
        print("No documents to store.")
        return 0
    
    print("-" * 50)
    print(f"Total documents to store: {len(all_documents)}")
    
    # Store in Supabase
    try:
        stored = store_documents_in_supabase(all_documents)
        print("-" * 50)
        print(f"✓ Successfully stored {stored} documents in Supabase")
        return stored
    except Exception as e:
        print(f"✗ Error storing documents: {e}")
        raise


def search_vectors(
    query: str,
    filter_metadata: Optional[Dict[str, Any]] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Search the vector database for similar documents.
    
    Args:
        query: Search query text
        filter_metadata: Optional dict of metadata filters
        top_k: Number of results to return
        
    Returns:
        List of matching documents with similarity scores
    """
    supabase = get_supabase_client()
    embeddings = get_embeddings()
    
    # Generate embedding for query
    query_vector = embeddings.embed_query(query)
    
    # Call the match_documents function in Supabase
    # This requires a SQL function to be set up (see implementation_plan.md)
    response = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_vector,
            "match_count": top_k,
            "filter": filter_metadata or {}
        }
    ).execute()
    
    return response.data


if __name__ == "__main__":
    print("=" * 50)
    print("CSV Data Pipeline - Supabase Vector Storage")
    print("=" * 50)
    print()
    
    try:
        total = process_all_csv_files()
        print()
        print(f"Pipeline completed. Total documents: {total}")
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        exit(1)