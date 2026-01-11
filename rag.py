from langgraph.graph import StateGraph, END
from document import DocumentStore
import time


class RagWorkFlow:
    def __init__(self, document_store: DocumentStore):
        self.document_store = document_store

        workflow = StateGraph(dict)
        workflow.add_node("retrieve", self.simple_retrieve)
        workflow.add_node("answer", self.simple_answer)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "answer")
        workflow.add_edge("answer", END)
        self.chain = workflow.compile()

    def simple_retrieve(self, state):
        query = state["question"]
        results = self.document_store.search_query(query)
        state["context"] = results
        return state
    
    def simple_answer(self, state):
        ctx = state["context"]
        if ctx:
            answer = f"I found this: '{ctx[0][:100]}...'"
        else:
            answer = "Sorry, I don't know."
        state["answer"] = answer
        return state
    
    def run_query(self, question: str):
        start = time.time()
        result = self.chain.invoke({"question": question})
        return {
            "question": question,
            "answer": result["answer"],
            "context_used": result.get("context", []),
            "latency_sec": round(time.time() - start, 3)
        }