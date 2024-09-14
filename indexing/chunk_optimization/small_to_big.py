from llama_index.core.schema import TextNode, NodeRelationship
from llama_index.core import Document


def extract_parent_node(node: TextNode, documents: list[Document], verbose: bool = True) -> Document | None:
    '''
    extracts parent document (source) from the child chunks
    
    Args:
        node: TextNode - the chunk text node, for which parent document is to be extracted
        documents: list[Document] - the search list for the parent document
        verbose: bool - whether to display status of search
    Returns:
        document - the document node
    '''
    if node.relationships:
        parent_id = node.relationships[NodeRelationship.SOURCE].node_id
        for document in documents:
            if document.id_ == parent_id:
                return document 
        if verbose:
            print("Document not in the list")
        return None
    
    if verbose:
        print("TextNode doesn't have parent document.")
    return None
    