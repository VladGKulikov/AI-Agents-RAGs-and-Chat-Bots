from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from dataclasses import dataclass
from agent.relevance_grader import RelevanceGrader
from agent.retrieval_grader import RetrievalGrader
from agent.rag_chain_generator import RagChainGenerator
from agent.hallucination_grader import HallucinationGrader
from agent.answer_grader import AnswerGrader
from agent.question_rewriter import QuestionRewriter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


@dataclass
class SelfRAGConfig:
    model: str
    embed_model: str
    chroma_db_path: str


# class AgentState(TypedDict):
#     messages: Annotated[list[AnyMessage], operator.add]

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]


class SelfRAGAgent:
    def __init__(self, config: SelfRAGConfig, system=""):  # tools, system=""):

        self.config = config
        self.system = system

        # import chromadb
        # client = chromadb.HttpClient(host="localhost", port="8000")
        # https://abhishektatachar.medium.com/run-chroma-db-on-a-local-machine-and-as-a-docker-container-a9d4b91d2a97
        self.embedding = OpenAIEmbeddings(model=config.embed_model)
        self.vectorstore = Chroma(persist_directory=config.chroma_db_path, embedding_function=self.embedding)
        self.retriever = self.vectorstore.as_retriever()

        self.relevance_grader = RelevanceGrader().get_relevance_grader()
        self.retrieval_grader = RetrievalGrader().get_retrieval_grader()
        self.rag_chain = RagChainGenerator().get_rag_chain()
        self.hallucination_grader = HallucinationGrader().get_hallucination_grader()
        self.answer_grader = AnswerGrader().get_answer_grader()
        self.question_rewriter = QuestionRewriter().get_question_rewriter()

        graph = StateGraph(GraphState)

        # Define the nodes
        graph.add_node("relevance", self.relevance) # grade relevance
        graph.add_node("retrieve", self.retrieve)  # retrieve
        graph.add_node("grade_documents", self.grade_documents)  # grade documents
        graph.add_node("generate", self.generate)  # generate
        graph.add_node("transform_query", self.transform_query)  # transform_query

        # Build graph
        graph.add_edge(START, "relevance")
        graph.add_conditional_edges(
            "relevance",
            self.grade_relevance,
            {
                "not_relevant": END,
                "relevant": "retrieve",
            },
        )
        # graph.add_edge("relevance", "retrieve")
        graph.add_edge("retrieve", "grade_documents")
        graph.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        graph.add_edge("transform_query", "retrieve")
        graph.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not_supported": "generate",
                "useful": END,
                "not_useful": "transform_query",
            },
        )
        # Compile
        self.graph = graph.compile()

    ### Nodes

    def relevance(self, state):    
        """
        If the question (state["question"]) isn't relevant (self.grade_relevance) 
        then len(state) == 1 and in main.py:

            if len(query_response) == 1:
                out = "I'm sorry, but your question doesn't seem relevant to our topic. 
                    Could you please try again or rephrase your message?"
            else:            
                out=query_response['generation']
        """

        return state 


    def retrieve(self, state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        # print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = self.retriever.get_relevant_documents(question)
        return {"documents": documents, "question": question}

    def generate(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        # print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = (self.rag_chain.invoke({"context": documents, "question": question}))
        return {"documents": documents, "question": question, "generation": generation}

    def grade_relevance(self, state):

        question = state["question"]

        score = self.relevance_grader.invoke(
            {"question": question}
        )

        grade = score.binary_score

        if grade == "yes":
            return 'relevant'
        else:
            return 'not_relevant'



    def grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        # print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                # print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                # print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}

    def transform_query(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        # print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = self.question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    ### Edges

    def decide_to_generate(self, state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        # print("---ASSESS GRADED DOCUMENTS---")
        # state["question"]
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            # print(
            #     "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            # )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            # print("---DECISION: GENERATE---")
            return "generate"
        
    def grade_generation_v_documents_and_question(self, state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        # print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = self.hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score.binary_score

        # Check hallucination
        if grade == "yes":
            # print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            # print("---GRADE GENERATION vs QUESTION---")
            score = self.answer_grader.invoke({"question": question, "generation": generation})
            grade = score.binary_score
            if grade == "yes":
                # print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                # print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not_useful"
        else:
            # pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not_supported"
