from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core import Document
from llama_index.core.schema import TextNode, NodeRelationship


def get_hierarchical_nodes(documents: list[Document], chunk_sizes: list[int] = None) -> list[TextNode]:
    '''
    performs hierarchical node parsing with tree like relationship
    
    Args:
        documents: list[Document] - document nodes which needs to be chunked
        chunk_sizes: list[int] - sizes of chunks if not defined default values will be taken
    Returns:
        list[TextNode] - the chunked nodes with relationships
    '''
    parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = parser.get_nodes_from_documents(documents)
    return nodes 
    
def get_parent_node(node: TextNode, nodes: list[TextNode], verbose: bool = True) -> TextNode | None:
    '''
    fetches parent node from the nodes list if exists
    
    Args:
        node: TextNode - node for which parent nodes needs to be fetched
        nodes: list[TextNode] - list of nodes to search for parent
        verbose: bool - whether to display search status
    Returns:
        TextNode - the parent node
    '''
    if node.relationships and node.relationships[NodeRelationship.PARENT]:
        parent_id = node.relationships[NodeRelationship.PARENT].node_id
        for node_ in nodes:
            if node_.id_ == parent_id:
                return node_ 
        
        if verbose:
            print("Could not find parent node in given list of nodes")
        return None 
    
    if verbose:
        print("Node relationship for Parent not found")
    return None 

def get_next_node(node: TextNode, nodes: list[TextNode], verbose: bool = True) -> TextNode | None:
    '''
    fetches next node from the nodes list if exists
    
    Args:
        node: TextNode - node for which next nodes needs to be fetched
        nodes: list[TextNode] - list of nodes to search for next
        verbose: bool - whether to display search status
    Returns:
        TextNode - the next node
    '''
    if node.relationships and node.relationships[NodeRelationship.NEXT]:
        next_id = node.relationships[NodeRelationship.NEXT].node_id
        for node_ in nodes:
            if node_.id_ == next_id:
                return node_ 
        
        if verbose:
            print("Could not find next node in given list of nodes")
        return None 
    
    if verbose:
        print("Node relationship for NEXT not found")
    return None 

def get_previous_node(node: TextNode, nodes: list[TextNode], verbose: bool = True) -> TextNode | None:
    '''
    fetches previous node from the nodes list if exists
    
    Args:
        node: TextNode - node for which previous nodes needs to be fetched
        nodes: list[TextNode] - list of nodes to search for previous
        verbose: bool - whether to display search status
    Returns:
        TextNode - the previous node
    '''
    if node.relationships and node.relationships[NodeRelationship.PREVIOUS]:
        previous_id = node.relationships[NodeRelationship.PREVIOUS].node_id
        for node_ in nodes:
            if node_.id_ == previous_id:
                return node_ 
        
        if verbose:
            print("Could not find previous node in given list of nodes")
        return None 
    
    if verbose:
        print("Node relationship for Previous not found")
    return None 

def get_child_nodes(node: TextNode, nodes: list[TextNode], verbose: bool = True) -> list[TextNode] | None:
    '''
    fetches child node from the nodes list if exists
    
    Args:
        node: TextNode - node for which child nodes needs to be fetched
        nodes: list[TextNode] - list of nodes to search for child
        verbose: bool - whether to display search status
    Returns:
        TextNode - the child node
    '''
    if node.relationships and node.relationships[NodeRelationship.CHILD]:
        childs = node.relationships[NodeRelationship.CHILD]
        if not isinstance(childs, (list, tuple)):
            childs = [childs]
        child_nodes = []
        for child in childs:
            child_id = child.node_id
            for node_ in nodes:
                if node_.id_ == child_id:
                    child_nodes.append(node_)
            
            if verbose:
                print("Could not find child node in given list of nodes")
        return child_nodes
        
    if verbose:
        print("Node relationship for Child not found")
    return None 
