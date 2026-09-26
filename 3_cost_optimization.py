from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()


class TokenBudget():
    def __init__(self, max_tokens_per_request: int=4000):
        self.max_per_request = max_tokens_per_request
        self.usage = {'total_input': 0, 'total_output': 0, 'requests': 0}

    def estimate_tokens(self, text: str) -> int:
        return int(len(text.split())) * 1.3

    def check_budget(self, text: str) -> tuple[bool, str]:
        tokens = self.estimate_tokens(text)
        return tokens < self.max_per_request, tokens

    def record_tokens(self, input_tokens: int, output_tokens: int):
        self.usage['total_input'] += input_tokens
        self.usage['total_output'] += output_tokens
        self.usage['requests'] += 1

    def get_stats(self) -> dict:
        return {
            **self.usage,
            'total_tokens': self.usage['total_input'] + self.usage['total_output'],
            'avg_per_request': (
                (self.usage['total_input'] + self.usage['total_output'])/ max(self.usage['requests'], 1)
            )
        }
        
from langsmith import traceable
class BudgetLLM:
    '''LLM with Token budgeting'''
    def __init__(self, max_tokens: int = 4000):
        self.llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
        self.budget = TokenBudget(max_tokens_per_request=max_tokens)

    @traceable(name="budgeted_invoke")
    def invoke(self, query: str):
        within_budget, tokens = self.budget.check_budget(query)

        if not within_budget:
            raise ValueError(
                f"Query exceeds token budget: {tokens} > {self.budget.max_per_request}"
            )

        response = self.llm.invoke(query)
        result = response.content

        #record_usage
        output_tokens = self.budget.estimate_tokens(result)
        self.budget.record_tokens(tokens, output_tokens)

        return result

    def get_stats(self) -> dict:
        return self.budget.get_stats()

def demo_token_budgeting():
    '''Demonstrate token budgeting'''

    llm = BudgetLLM(max_tokens=100)

    queries = [
        "What is AI?",  # Within budget
        "Explain " + "very " * 100 + "complex topic",  # Over budget
    ]

    for query in queries:
        try:
            result = llm.invoke(query)
            print(f"{query} --> {result[:80]}")
        except ValueError as e:
            print(f"{query} --> {e}")

    print(f"\nUsage: {llm.get_stats()}")

if __name__ == '__main__':
    demo_token_budgeting()