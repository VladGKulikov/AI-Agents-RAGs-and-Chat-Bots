### Retrieval Grader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class QuestionRewriter:
    def __init__(self, retrieval_grader_llm='gpt-4o-mini'):
        self.llm = ChatOpenAI(model=retrieval_grader_llm, temperature=0)
        self.system_prompt = """
        You a question re-writer that converts an input question to a better version that is optimized \n
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent meaning.
        """

    def get_question_rewriter(self):
         re_write_prompt = ChatPromptTemplate.from_messages(
             [
                 ("system", self.system_prompt),
                 (
                     "human",
                     "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                 ),
             ]
         )

         question_rewriter = re_write_prompt | self.llm | StrOutputParser()
         return question_rewriter
