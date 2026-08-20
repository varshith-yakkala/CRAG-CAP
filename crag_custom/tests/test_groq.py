import pytest
from crag_custom.llm.groq_client import GroqClient
from crag_custom.llm.query_rewriter import GroqQuestionRewriter
from crag_custom.llm.generator import GroqGenerator

def test_groq_client_mock():
    client = GroqClient(debug_mode=True)
    res = client.generate([{"role": "user", "content": "Hello"}])
    assert "[MOCK GROQ GENERATION" in res

def test_groq_query_rewriter_mock():
    client = GroqClient(debug_mode=True)
    rewriter = GroqQuestionRewriter(client=client)
    res = rewriter.rewrite("What is George Rankin's occupation?")
    assert res is not None

def test_groq_generator_mock():
    client = GroqClient(debug_mode=True)
    generator = GroqGenerator(client=client)
    res = generator.generate(query="What is George Rankin's occupation?", context="George Rankin was a soldier.")
    assert "[MOCK GROQ GENERATION" in res
