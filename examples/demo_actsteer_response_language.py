import argparse
import json
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

VLLM_HOOK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VLLM_HOOK_ROOT / "vllm_hook_plugins"))

from vllm import SamplingParams
from vllm_hook_plugins import HookLLM


try:
    import langdetect
except Exception:
    langdetect = None


DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_LANGUAGE = "ko"
DEFAULT_PROMPT_KEY = "trip_planning"
DEFAULT_COEFFICIENTS = "40,60,75,80,90,100"
DEFAULT_EXTRA_PROMPTS = "school_schedules,exercise_habits,trip_lesson"
MODEL_CONFIG_STEMS = {
    "microsoft/Phi-3-mini-4k-instruct": "Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B-Instruct",
}
PROMPT_KEYS = (
    "exercise_habits",
    "focus",
    "moon",
    "plain_focus",
    "school_schedules",
    "trip_lesson",
    "trip_planning",
)
LANGUAGE_NAMES = {
    "hi": "Hindi",
    "ko": "Korean",
}

SCRIPT_RANGES = {
    "hi": [(0x0900, 0x097F)],
    "ko": [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)],
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
    detected_language: Optional[str]
    target_script_ratio: float
    ascii_alpha_ratio: float
    char_repetition_ratio: float
    token_unique_ratio: float
    target_language_ok: bool


def parse_coefficients(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_prompt_keys(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def language_name(language: str) -> str:
    return LANGUAGE_NAMES.get(language, language)


def model_config_stem(model: str) -> str:
    return MODEL_CONFIG_STEMS.get(model, model.rsplit("/", maxsplit=1)[-1])


def default_config_path(language: str, model: str) -> str:
    stem = model_config_stem(model)
    return f"model_configs/activation_steer/{stem}-response-language-{language}.json"


def build_prompts(language: str) -> dict[str, str]:
    name = language_name(language)
    return {
        "trip_planning": "Explain trip planning in simple terms.",
        "school_schedules": "Suggest first steps for improving school schedules.",
        "exercise_habits": "Write encouraging advice about exercise habits.",
        "trip_lesson": "Create a beginner-friendly lesson about trip planning.",
        "plain_focus": (
            "Give three practical tips for staying focused during a busy week."
        ),
        "moon": (
            "Write a concise museum label about the Moon landing. "
            f"Use only {name} in your response, no other language is allowed."
        ),
        "focus": (
            "Give three practical tips for staying focused during a busy week. "
            f"Your entire response should be in {name}, no other language is allowed."
        ),
    }


def detect_language(text: str) -> Optional[str]:
    if langdetect is None or not text.strip():
        return None
    try:
        return langdetect.detect(text)
    except Exception:
        return None


def script_char_count(text: str, language: str) -> int:
    ranges = SCRIPT_RANGES.get(language, [])
    count = 0
    for char in text:
        codepoint = ord(char)
        if any(start <= codepoint <= end for start, end in ranges):
            count += 1
    return count


def target_script_ratio(text: str, language: str) -> float:
    non_space = sum(1 for char in text if not char.isspace())
    if non_space == 0:
        return 0.0
    return script_char_count(text, language) / non_space


def ascii_alpha_count(text: str) -> int:
    return sum(1 for char in text if ("a" <= char.lower() <= "z"))


def ascii_alpha_ratio(text: str) -> float:
    non_space = sum(1 for char in text if not char.isspace())
    if non_space == 0:
        return 0.0
    return ascii_alpha_count(text) / non_space


def char_repetition_ratio(text: str) -> float:
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return 0.0
    repeated = 0
    previous = None
    run = 0
    for char in chars:
        if char == previous:
            run += 1
        else:
            run = 1
            previous = char
        if run >= 4:
            repeated += 1
    return repeated / len(chars)


def response_tokens(text: str) -> list[str]:
    punctuation = ".,;:!?()[]{}<>\"'`*_-=+/\\|#~"
    return [
        token.strip(punctuation)
        for token in text.split()
        if token.strip(punctuation)
    ]


def token_unique_ratio(text: str) -> float:
    tokens = response_tokens(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def response_language_ok(text: str, language: str) -> bool:
    return (
        target_script_ratio(text, language) >= 0.75
        and ascii_alpha_ratio(text) <= 0.03
        and char_repetition_ratio(text) <= 0.35
        and token_unique_ratio(text) >= 0.30
    )


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


def assert_vector_exists(config: dict) -> None:
    vector_path = Path(config["vector_path"])
    if not vector_path.is_absolute():
        vector_path = VLLM_HOOK_ROOT / vector_path
    if not vector_path.exists():
        raise FileNotFoundError(
            f"Missing steering vector {vector_path}. Run the matching AutoSteer "
            "response_language pipeline first, then copy the selected vector into "
            "vLLM-Hook/steering_vectors."
        )


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


def run_case(
    llm: HookLLM,
    default_config: dict,
    prompts: dict[str, str],
    language: str,
    case: DemoCase,
) -> DemoResult:
    prompt = apply_chat_template(llm, prompts[case.prompt_key])
    params = sampling_params(llm, default_config, case)
    output = llm.generate(prompt, params, use_hook=case.use_hook)[0]
    text = output.outputs[0].text.strip()

    # Prefix cache keys do not include steering params, so clear between cases.
    llm.llm_engine.reset_prefix_cache()

    detected = detect_language(text)
    script_ratio = target_script_ratio(text, language)
    english_ratio = ascii_alpha_ratio(text)
    repetition_ratio = char_repetition_ratio(text)
    unique_ratio = token_unique_ratio(text)
    return DemoResult(
        case=case,
        text=text,
        detected_language=detected,
        target_script_ratio=script_ratio,
        ascii_alpha_ratio=english_ratio,
        char_repetition_ratio=repetition_ratio,
        token_unique_ratio=unique_ratio,
        target_language_ok=response_language_ok(text, language),
    )


def print_result(result: DemoResult, language: str) -> None:
    case = result.case
    hook = "on" if case.use_hook else "off"
    method = case.method or "none"

    print("=" * 88)
    print(
        f"[{case.label}] prompt={case.prompt_key} hook={hook} "
        f"method={method} coefficient={coefficient_label(case)} "
        f"temperature={case.temperature:g} top_p={case.top_p:g}"
    )
    print(
        f"Target={language_name(language)} detected={result.detected_language} "
        f"ok={result.target_language_ok} "
        f"target_script_ratio={result.target_script_ratio:.3f} "
        f"ascii_alpha_ratio={result.ascii_alpha_ratio:.3f} "
        f"char_repetition_ratio={result.char_repetition_ratio:.3f} "
        f"token_unique_ratio={result.token_unique_ratio:.3f}"
    )
    print()
    print(result.text)


def print_summary(results: list[DemoResult]) -> None:
    print("=" * 88)
    print("Summary")
    print(
        f"{'Case':<28} {'Prompt':<8} {'Method':<10} {'Coeff':>6} "
        f"{'Detected':>9} {'OK':>4} {'Script':>7} {'ASCII':>7} "
        f"{'Repeat':>7} {'Unique':>7}"
    )
    for result in results:
        case = result.case
        method = case.method or "none"
        detected = result.detected_language or "?"
        print(
            f"{case.label:<28} {case.prompt_key:<8} {method:<10} "
            f"{coefficient_label(case):>6} {detected:>9} "
            f"{str(result.target_language_ok):>4} "
            f"{result.target_script_ratio:>7.3f} "
            f"{result.ascii_alpha_ratio:>7.3f} "
            f"{result.char_repetition_ratio:>7.3f} "
            f"{result.token_unique_ratio:>7.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--config", default=None)
    parser.add_argument("--prompt-key", choices=sorted(PROMPT_KEYS), default=DEFAULT_PROMPT_KEY)
    parser.add_argument("--coefficients", default=DEFAULT_COEFFICIENTS)
    parser.add_argument("--extra-prompt-keys", default=DEFAULT_EXTRA_PROMPTS)
    parser.add_argument("--include-adjust-rs", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=180)
    args = parser.parse_args()

    cache_dir = "./cache/"
    config_path = args.config or default_config_path(args.language, args.model)
    prompts = build_prompts(args.language)
    default_config = load_steering_config(config_path)
    assert_vector_exists(default_config)

    llm = HookLLM(
        model=args.model,
        worker_name="steer_hook_act",
        config_file=config_path,
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

    cases = [DemoCase("baseline greedy", args.prompt_key, use_hook=False, max_tokens=args.max_tokens)]
    for coeff in parse_coefficients(args.coefficients):
        cases.append(
            DemoCase(
                f"add_vector coeff {coeff:g}",
                args.prompt_key,
                use_hook=True,
                method="add_vector",
                coefficient=coeff,
                max_tokens=args.max_tokens,
            )
        )
    tuned_coeff = default_config.get("coefficient")
    for prompt_key in parse_prompt_keys(args.extra_prompt_keys):
        if prompt_key == args.prompt_key:
            continue
        cases.extend(
            [
                DemoCase(
                    f"baseline {prompt_key}",
                    prompt_key,
                    use_hook=False,
                    max_tokens=args.max_tokens,
                ),
                DemoCase(
                    f"tuned {prompt_key}",
                    prompt_key,
                    use_hook=True,
                    method=default_config.get("method", "add_vector"),
                    coefficient=tuned_coeff,
                    max_tokens=args.max_tokens,
                ),
            ]
        )
    if args.include_adjust_rs:
        cases.append(
            DemoCase(
                "adjust_rs",
                args.prompt_key,
                use_hook=True,
                method="adjust_rs",
                max_tokens=args.max_tokens,
            )
        )

    print(f"Running {len(cases)} cases with one HookLLM instance for {args.model}.")
    print(f"Target instruction: response_language={language_name(args.language)} ({args.language})")

    results = []
    try:
        for case in cases:
            result = run_case(llm, default_config, prompts, args.language, case)
            results.append(result)
            print_result(result, args.language)

        print_summary(results)
    finally:
        llm.close()
