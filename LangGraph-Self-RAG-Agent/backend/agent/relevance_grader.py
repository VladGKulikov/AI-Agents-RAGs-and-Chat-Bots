from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Data model
class GradeRelevance(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Question is grounded in the data, 'yes' or 'no'"
    )

class RelevanceGrader:
    def __init__(self, relevance_grader_llm='gpt-4o-mini'):
        self.llm = ChatOpenAI(model=relevance_grader_llm, temperature=0)
        self.structured_llm_grader = self.llm.with_structured_output(GradeRelevance)
        self.system_prompt = """
        You are a grader assessing whether a question is 1) meaningful and 2) relevant \n
        to our data topic: 'Transformer and Large Language Models(LLMs).' \n
        Provide a binary score of 'yes' or 'no.' \n
        'Yes' means that the question is 1) meaningful and 2) relevant to the topic.
        """

    def get_relevance_grader(self):
        relevance_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "Question is: \n\n {question}"),
            ]
        )

        relevance_grader = relevance_prompt | self.structured_llm_grader
        return relevance_grader