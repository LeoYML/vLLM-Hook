import multiprocessing as mp
import os

mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import SamplingParams
from vllm_hook_plugins import HookLLM


def comma_count(text: str) -> int:
    return text.count(",")


if __name__ == "__main__":
    cache_dir = "./cache/"
    model = "microsoft/Phi-3-mini-4k-instruct"

    llm = HookLLM(
        model=model,
        worker_name="steer_hook_act",
        config_file="model_configs/activation_steer/Phi-3-mini-4k-instruct-no-comma.json",
        download_dir=cache_dir,
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        trust_remote_code=True,
        dtype="auto",
        enforce_eager=True,
        enable_prefix_caching=True,
        enable_hook=True,
        tensor_parallel_size=1,
    )

    prompts = [
        (
            "I am planning a trip to Japan and I would like thee to write an itinerary "
            "for my journey in a Shakespearean style. You are not allowed to use any "
            "commas in your response."
        ),
        (
            "Write a short product profile for a travel mug that keeps coffee hot for "
            "a full workday. Do not use any commas in your response."
        ),
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=220,
        stop_token_ids=[llm.tokenizer.eos_token_id, 32007],
    )

    examples = [
        llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in prompts
    ]

    steered_outputs = llm.generate(examples, sampling_params)
    llm.llm_engine.reset_prefix_cache()
    baseline_outputs = llm.generate(examples, sampling_params, use_hook=False)
    llm.llm_engine.reset_prefix_cache()

    for prompt, steered, baseline in zip(prompts, steered_outputs, baseline_outputs):
        steered_text = steered.outputs[0].text
        baseline_text = baseline.outputs[0].text

        print("=" * 80)
        print("[Prompt]")
        print(prompt)

        print("\n[With no-comma activation steering]")
        print(steered_text)
        print(f"Commas: {comma_count(steered_text)}")

        print("\n[Without activation steering]")
        print(baseline_text)
        print(f"Commas: {comma_count(baseline_text)}")
