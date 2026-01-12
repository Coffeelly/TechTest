from langgraph.graph import StateGraph, END
from document import BaseDocumentStore
import time
from typing import Dict, Any

class RagWorkFlow:
    """
    RAG (Retrieval-Augmented Generation) pipeline using LangGraph.    
    """
    def __init__(self, document_store: BaseDocumentStore):
        """
        Initializes the RAG workflow with a specific document store.

        Args: document_store (BaseDocumentStore) backend (Memory or Qdrant) used for retrieving context.
        """

        self.document_store = document_store

        # Initialize the state graph with a simple dictionary state
        workflow = StateGraph(dict)

        # Define nodes (steps in the workflow)
        workflow.add_node("retrieve", self.simple_retrieve)
        workflow.add_node("answer", self.simple_answer)

        # Define the flow (edges)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "answer")
        workflow.add_edge("answer", END)

        # Compile the graph into an executable chain
        self.chain = workflow.compile()

    def simple_retrieve(self, state) -> Dict[str, Any]:
        """
        Retrieve relevant context based on the user's question.

        Args    : state (Dict[str, Any]) containing the question.
        Returns : Dict[str, Any] Updated state with retrieved context.
        """
        query = state["question"]
        results = self.document_store.search_query(query)
        state["context"] = results
        return state
    
    def simple_answer(self, state) -> Dict[str, Any]:
        """
        Generate an answer based on the retrieved context.

        Args    : state (Dict[str, Any]) containing context.
        Returns : Dict[str, Any]: Updated state with the final answer.
        """
        ctx = state["context"]
        if ctx:
            answer = f"I found this: '{ctx[0][:100]}...'"
        else:
            answer = "Sorry, I don't know."
        state["answer"] = answer
        return state
    
    def run_query(self, question: str)-> Dict[str, Any]:
        """
        Executes the full RAG pipeline for a given question.

        Args    : question (str) from user's input.
        Returns : Dict[str, Any] containing the answer, context used, and latency.
        """
        start = time.time()

        # Invoke the LangGraph chain
        result = self.chain.invoke({"question": question})
        return {
            "question": question,
            "answer": result["answer"],
            "context_used": result.get("context", []),
            "latency_sec": round(time.time() - start, 3)
        }