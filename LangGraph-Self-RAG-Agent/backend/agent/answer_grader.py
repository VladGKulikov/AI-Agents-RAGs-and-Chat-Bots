### Answer Grader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

class AnswerGrader:
    def __init__(self, answer_grader_llm='gpt-4o-mini'):
        self.llm = ChatOpenAI(model=answer_grader_llm, temperature=0)
        self.structured_llm_grader = self.llm.with_structured_output(GradeAnswer)
        self.system_prompt = """
        You are a grader assessing whether an answer addresses resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
        """

    def get_answer_grader(self):
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )

        answer_grader = answer_prompt | self.structured_llm_grader
        return answer_grader
