import json
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Optional

mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import SamplingParams
from vllm_hook_plugins import HookLLM


MODEL = "microsoft/Phi-3-mini-4k-instruct"
CONFIG_PATH = "model_configs/activation_steer/Phi-3-mini-4k-instruct-no-comma.json"

PROMPTS = {
    "travel": (
        "Write a compact five sentence itinerary for Kyoto Osaka and Nara in a "
        "Shakespearean style. Mention temples trains tea and evening walks. "
        "Do not use any commas in your response."
    ),
}


@dataclass(frozen=True)
class DemoCase:
    label: str
    prompt_key: str
    use_hook: bool
    method: Optional[str] = None
    coefficient: Optional[float] = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 180
    seed: Optional[int] = None


@dataclass(frozen=True)
class DemoResult:
    case: DemoCase
    text: str


def comma_count(text: str) -> int:
    return text.count(",")


def coefficient_label(case: DemoCase) -> str:
    if not case.use_hook:
        return "-"
    if case.method == "adjust_rs":
        return "n/a"
    if case.coefficient is None:
        return "json"
    return f"{case.coefficient:g}"


def load_steering_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)["steering"]


def apply_chat_template(llm: HookLLM, prompt: str) -> str:
    return llm.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )


def steering_override(default_config: dict, case: DemoCase) -> Optional[dict]:
    if not case.use_hook:
        return None

    steer = dict(default_config)
    if case.method is not None:
        steer["method"] = case.method
    if case.coefficient is not None:
        steer["coefficient"] = case.coefficient
    return steer


def sampling_params(llm: HookLLM, default_config: dict, case: DemoCase) -> SamplingParams:
    stop_token_ids = [
        token_id
        for token_id in (llm.tokenizer.eos_token_id, 32007)
        if token_id is not None
    ]
    kwargs = {
        "temperature": case.temperature,
        "top_p": case.top_p,
        "max_tokens": case.max_tokens,
        "stop_token_ids": stop_token_ids,
    }
    if case.seed is not None:
        kwargs["seed"] = case.seed

    steer = steering_override(default_config, case)
    if steer is not None:
        kwargs["extra_args"] = {"steer": steer}

    return SamplingParams(**kwargs)


def run_case(llm: HookLLM, default_config: dict, case: DemoCase) -> DemoResult:
    prompt = apply_chat_template(llm, PROMPTS[case.prompt_key])
    params = sampling_params(llm, default_config, case)
    output = llm.generate(prompt, params, use_hook=case.use_hook)[0]
    text = output.outputs[0].text.strip()

    # Prefix cache keys do not include steering params, so clear between cases.
    llm.llm_engine.reset_prefix_cache()
    return DemoResult(case=case, text=text)


def print_result(result: DemoResult) -> None:
    case = result.case
    hook = "on" if case.use_hook else "off"
    method = case.method or "none"

    print("=" * 88)
    print(
        f"[{case.label}] prompt={case.prompt_key} hook={hook} "
        f"method={method} coefficient={coefficient_label(case)} "
        f"temperature={case.temperature:g} top_p={case.top_p:g}"
    )
    print(f"Commas: {comma_count(result.text)}")
    print()
    print(result.text)


def print_summary(results: list[DemoResult]) -> None:
    print("=" * 88)
    print("Summary")
    print(
        f"{'Case':<28} {'Prompt':<8} {'Method':<10} {'Coeff':>6} "
        f"{'Temp':>5} {'Top-p':>5} {'Commas':>7}"
    )
    for result in results:
        case = result.case
        method = case.method or "none"
        print(
            f"{case.label:<28} {case.prompt_key:<8} {method:<10} "
            f"{coefficient_label(case):>6} "
            f"{case.temperature:>5g} {case.top_p:>5g} {comma_count(result.text):>7}"
        )


if __name__ == "__main__":
    cache_dir = "./cache/"
    default_config = load_steering_config(CONFIG_PATH)

    llm = HookLLM(
        model=MODEL,
        worker_name="steer_hook_act",
        config_file=CONFIG_PATH,
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

    cases = [
        DemoCase("baseline greedy", "travel", use_hook=False),
        DemoCase(
            "weak add_vector",
            "travel",
            use_hook=True,
            method="add_vector",
            coefficient=10,
        ),
        DemoCase(
            "default add_vector",
            "travel",
            use_hook=True,
            method="add_vector",
            coefficient=default_config["coefficient"],
        ),
        DemoCase(
            "tuned add_vector",
            "travel",
            use_hook=True,
            method="add_vector",
            coefficient=55,
        ),
        DemoCase(
            "oversteered add_vector",
            "travel",
            use_hook=True,
            method="add_vector",
            coefficient=75,
        ),
        DemoCase("adjust_rs", "travel", use_hook=True, method="adjust_rs"),
        DemoCase(
            "baseline sampled",
            "travel",
            use_hook=False,
            temperature=0.7,
            top_p=0.9,
            seed=7,
        ),
        DemoCase(
            "steered sampled",
            "travel",
            use_hook=True,
            method="add_vector",
            coefficient=55,
            temperature=0.7,
            top_p=0.9,
            seed=7,
        ),
    ]

    print(f"Running {len(cases)} cases with one HookLLM instance for {MODEL}.")

    results = []
    try:
        for case in cases:
            result = run_case(llm, default_config, case)
            results.append(result)
            print_result(result)

        print_summary(results)
    finally:
        llm.close()
