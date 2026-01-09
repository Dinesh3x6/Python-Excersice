import os
import time
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from openai import OpenAI
from google import genai

load_dotenv()


# =========================
# Abstract Adapter
# =========================
class ModelAdapter(ABC):
    name: str

    @abstractmethod
    def generate(self, prompt: str) -> dict:
        pass


# =========================
# OpenAI Adapter
# =========================
class OpenAIAdapter(ModelAdapter):
    def __init__(self):
        self.name = "OpenAI GPT-4.1-mini"
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, prompt: str) -> dict:
        start = time.time()

        response = self.client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )

        latency = time.time() - start

        return {
            "provider": self.name,
            "text": response.output_text,
            "latency": round(latency, 2),
            "tokens": response.usage.total_tokens if response.usage else 0,
        }


# =========================
# Gemini Adapter
# =========================
class GeminiAdapter(ModelAdapter):
    def __init__(self):
        self.name = "Google Gemini 1.5 Flash"
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def generate(self, prompt: str) -> dict:
        start = time.time()

        response = self.client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )

        latency = time.time() - start

        tokens = (
            response.usage_metadata.total_token_count
            if response.usage_metadata
            else len(response.text.split())
        )

        return {
            "provider": self.name,
            "text": response.text,
            "latency": round(latency, 2),
            "tokens": tokens,
        }


# =========================
# Benchmark Runner
# =========================
class BenchmarkRunner:
    def __init__(self, adapters):
        self.adapters = adapters

    def run(self, prompt: str):
        results = []

        for adapter in self.adapters:
            try:
                result = adapter.generate(prompt)
                results.append(result)
            except Exception as e:
                results.append({
                    "provider": adapter.name,
                    "error": str(e)
                })

        return results


# =========================
# Main
# =========================
def main():
    prompt = input("Enter your prompt: ")

    adapters = [
        OpenAIAdapter(),
        GeminiAdapter()
    ]

    runner = BenchmarkRunner(adapters)
    results = runner.run(prompt)

    print("\n--- MODEL-AGNOSTIC AI BENCHMARK ---\n")

    for r in results:
        print("Provider:", r["provider"])

        if "error" in r:
            print("ERROR:", r["error"])
        else:
            print("Latency:", r["latency"], "seconds")
            print("Token Count:", r["tokens"])
            print("Response:")
            print(r["text"])

        print("-" * 60)


if __name__ == "__main__":
    main()
