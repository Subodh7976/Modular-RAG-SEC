from llama_index.core import Document
import pandas as pd


def create_documents_with_metadata(data: pd.DataFrame, text_column: str, 
                                   metadata_columns: list[str]) -> list[Document]:
    '''
    creates the list of documents representing each data entry with corresponding metadata
    
    Args:
        data: pd.DataFrame - dataset to be used for creating documents with metadata
        text_column: str - data column name which stores the content
        metadata_columns: list[str] - columns in dataset to be used for metadata
    Returns:
        list[Document] - documents form llama_index with metadata attached
    '''
    documents = [
        Document(
            text=document[text_column], 
            metadata={
                key: document[key] for key in metadata_columns
            }
        )
        for _, document in data.iterrows()
    ]
    
    return documents 

