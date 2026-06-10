"""Simple test to verify HIP vLLM can run inference."""
from vllm import LLM, SamplingParams

MODEL_PATH = "/data/Competitions/PRA26/Qwen3.5-9B"

def main():
    print("Loading model...")
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        max_model_len=512,
        gpu_memory_utilization=0.8,
    )

    sampling_params = SamplingParams(temperature=0.7, max_tokens=64)

    prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]

    for i, prompt in enumerate(prompts):
        print(f"\n--- Request {i+1} ---")
        print(f"Prompt: {prompt}")
        outputs = llm.generate([prompt], sampling_params)
        for output in outputs:
            generated = output.outputs[0].text
            print(f"Output: {generated}")

    print("\n=== Test passed! vLLM HIP is working correctly. ===")

if __name__ == "__main__":
    main()
