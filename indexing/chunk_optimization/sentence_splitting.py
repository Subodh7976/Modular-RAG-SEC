from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
from llama_index.core import Document
from llama_index.core.schema import TextNode


DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 128


def split_chunks_from_documents(documents: list[Document], chunk_size: int = DEFAULT_CHUNK_SIZE, 
                                chunk_overlap: int = DEFAULT_CHUNK_OVERLAP, window_size: int = 0, 
                                split_chunk: bool = False) -> list[TextNode]:
    '''
    splits the documents into multiple chunks, given the parameters
    
    Args:
        documents: list[Documents] - the list of llama_index Document class, which will be used to convert into chunks of text nodes
        chunk_size: int - token chunk size for each chunk
        chunk_window: int - token overlap of each chunk when splitting
        window_size: int - the window size, representing the number of sentences to be selected as context window above and below the target setence 
                            0 means regular splitting without any window.
        split_chunk: bool - whether to use the chunk based or default splitting for window splitter (default is single sentence splitting)
    Returns:
        list[TextNode] - the chunks in the llama_index TextNode class
    '''
    if not window_size:
        sentence_splitter = SentenceSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        ) if split_chunk else None 
        splitter = SentenceWindowNodeParser.from_defaults(
            sentence_splitter,
            window_size=window_size, 
            window_metadata_key="window", 
            original_text_metadata_key="original_text"
        )
    else:
        splitter = SentenceSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
    
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
    return nodes 
