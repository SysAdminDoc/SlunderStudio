"""
Slunder Studio v0.0.2 — Lyrics Engine
Dual-backend LLM wrapper: llama-cpp-python (primary) or transformers (fallback).
Supports streaming token output, model loading via Model Manager, and generation pipeline.

Backend selection:
  - llama-cpp-python: Fast, low VRAM, loads GGUF files. Needs C++ compiler to install.
  - transformers: Pure Python, no compiler needed. Uses standard HuggingFace models.
    Fallback when llama-cpp-python can't install (e.g. Python 3.14, no Visual Studio).
"""
import os
import threading
import time
from typing import Optional, Callable, Generator
from pathlib import Path

from core.settings import Settings
from core.model_manager import ModelManager, cleanup_gpu


# -- Model File Resolution -----------------------------------------------------

GGUF_FILENAMES = {
    "llama-3.1-8b-q4": [
        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "meta-llama-3.1-8b-instruct-q4_k_m.gguf",
    ],
    "llama-3.2-3b-q4": [
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "llama-3.2-3b-instruct-q4_k_m.gguf",
    ],
    "qwen-2.5-14b-q4": [
        "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        "qwen2.5-14b-instruct-q4_k_m.gguf",
    ],
}

# Ungated HuggingFace models for transformers fallback backend
# Matched by capability/size to the GGUF models they replace
_TRANSFORMERS_MODELS = {
    "llama-3.1-8b-q4": "Qwen/Qwen2.5-7B-Instruct",
    "llama-3.2-3b-q4": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-2.5-14b-q4": "Qwen/Qwen2.5-14B-Instruct",
}
_TRANSFORMERS_DEFAULT = "Qwen/Qwen2.5-3B-Instruct"


def _find_gguf_file(model_id: str) -> Optional[str]:
    """Find the GGUF file path for a model. Searches cache directory."""
    mgr = ModelManager()
    cache_dir = mgr.get_cache_dir(model_id)

    candidates = GGUF_FILENAMES.get(model_id, [])
    for name in candidates:
        path = cache_dir / name
        if path.exists():
            return str(path)

    if cache_dir.exists():
        for f in cache_dir.rglob("*.gguf"):
            return str(f)

    info = mgr.get_model_info(model_id)
    if info:
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        model_dir = hf_cache / f"models--{info.source.replace('/', '--')}"
        if model_dir.exists():
            for f in model_dir.rglob("*.gguf"):
                return str(f)

    return None


def _llama_cpp_available() -> bool:
    """Check if llama-cpp-python is importable without attempting install."""
    try:
        import llama_cpp
        return True
    except ImportError:
        return False


# -- LLM Wrapper (dual-backend) ------------------------------------------------

class LyricsLLM:
    """
    Dual-backend LLM for lyrics generation.
    Tries llama-cpp-python first (GGUF, fast, low VRAM).
    Falls back to transformers (pure Python, no compiler needed).
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._backend: Optional[str] = None  # "llama_cpp" or "transformers"
        self._model_id: Optional[str] = None
        self._model_path: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_id(self) -> Optional[str]:
        return self._model_id

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    def load(self, model_id: str = None, model_path: str = None, n_ctx: int = 4096):
        """
        Load a model for inference. Tries llama-cpp-python first,
        falls back to transformers if unavailable.
        """
        self.unload()

        # Phase 1: Try llama-cpp-python (best performance)
        if _llama_cpp_available():
            try:
                self._load_llama_cpp(model_id, model_path, n_ctx)
                return
            except Exception as e:
                print(f"[Lyrics] llama-cpp-python load failed: {e}")

        # Phase 2: Try to install llama-cpp-python
        if not _llama_cpp_available():
            try:
                from core.deps import ensure
                ensure("llama_cpp")
                self._load_llama_cpp(model_id, model_path, n_ctx)
                return
            except ImportError:
                print("[Lyrics] llama-cpp-python unavailable, using transformers backend")
            except Exception as e:
                print(f"[Lyrics] llama-cpp-python load failed: {e}")

        # Phase 3: Transformers fallback
        self._load_transformers(model_id)

    def _load_llama_cpp(self, model_id: str = None, model_path: str = None,
                         n_ctx: int = 4096):
        """Load via llama-cpp-python (GGUF files)."""
        from llama_cpp import Llama

        if model_id and not model_path:
            model_path = _find_gguf_file(model_id)
            if not model_path:
                raise FileNotFoundError(
                    f"GGUF file not found for model '{model_id}'. "
                    f"Please download it from the Model Hub first."
                )

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            verbose=False,
            n_threads=os.cpu_count() or 4,
        )
        self._backend = "llama_cpp"
        self._model_id = model_id or Path(model_path).stem
        self._model_path = model_path

    def _load_transformers(self, model_id: str = None):
        """Load via transformers (standard HuggingFace models, no compiler needed)."""
        from core.deps import ensure
        ensure("transformers", "torch")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Map GGUF model IDs to equivalent HuggingFace models
        hf_model = _TRANSFORMERS_MODELS.get(model_id, _TRANSFORMERS_DEFAULT)
        print(f"[Lyrics] Loading transformers model: {hf_model}")

        # Detect best dtype
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(
            hf_model, trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            hf_model,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )

        self._backend = "transformers"
        self._model_id = model_id or hf_model.split("/")[-1]
        self._model_path = hf_model

    def unload(self):
        """Unload the model and free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._backend = None
        self._model_id = None
        cleanup_gpu()

    def cleanup(self):
        self.unload()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        top_p: float = 0.92,
        top_k: int = 50,
        repeat_penalty: float = 1.1,
        max_tokens: int = 2048,
        cancel_event: Optional[threading.Event] = None,
        progress_cb: Optional[Callable[[int], None]] = None,
        token_cb: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> str:
        """Generate lyrics from prompts. Routes to active backend."""
        if not self.is_loaded:
            raise RuntimeError("No model loaded. Call load() first.")

        if self._backend == "llama_cpp":
            return self._generate_llama_cpp(
                system_prompt, user_prompt, temperature, top_p, top_k,
                repeat_penalty, max_tokens, cancel_event, progress_cb, token_cb,
            )
        else:
            return self._generate_transformers(
                system_prompt, user_prompt, temperature, top_p, top_k,
                repeat_penalty, max_tokens, cancel_event, progress_cb, token_cb,
            )

    def _generate_llama_cpp(
        self, system_prompt, user_prompt, temperature, top_p, top_k,
        repeat_penalty, max_tokens, cancel_event, progress_cb, token_cb,
    ) -> str:
        """Generate via llama-cpp-python streaming chat completion."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        output_tokens = []
        token_count = 0

        stream = self._model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            if cancel_event and cancel_event.is_set():
                break

            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                output_tokens.append(content)
                token_count += 1

                if token_cb:
                    token_cb(content)

                if progress_cb:
                    estimated_total = max(200, max_tokens // 4)
                    pct = min(95, int(token_count / estimated_total * 100))
                    progress_cb(pct)

        if progress_cb:
            progress_cb(100)

        return "".join(output_tokens)

    def _generate_transformers(
        self, system_prompt, user_prompt, temperature, top_p, top_k,
        repeat_penalty, max_tokens, cancel_event, progress_cb, token_cb,
    ) -> str:
        """Generate via transformers with streaming."""
        import torch
        from threading import Thread

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Apply chat template
        if hasattr(self._tokenizer, "apply_chat_template"):
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            text = f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant: "

        inputs = self._tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._model.device)
        input_len = input_ids.shape[1]

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": max(temperature, 0.01),
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repeat_penalty,
            "do_sample": True,
            "pad_token_id": self._tokenizer.eos_token_id,
        }

        # Try streaming via TextIteratorStreamer
        try:
            from transformers import TextIteratorStreamer
            streamer = TextIteratorStreamer(
                self._tokenizer, skip_prompt=True, skip_special_tokens=True,
            )
            gen_kwargs["streamer"] = streamer

            thread = Thread(
                target=lambda: self._model.generate(input_ids, **gen_kwargs),
                daemon=True,
            )
            thread.start()

            output_tokens = []
            token_count = 0
            for text_chunk in streamer:
                if cancel_event and cancel_event.is_set():
                    break
                if text_chunk:
                    output_tokens.append(text_chunk)
                    token_count += 1
                    if token_cb:
                        token_cb(text_chunk)
                    if progress_cb:
                        estimated_total = max(200, max_tokens // 4)
                        pct = min(95, int(token_count / estimated_total * 100))
                        progress_cb(pct)

            thread.join(timeout=5)

            if progress_cb:
                progress_cb(100)
            return "".join(output_tokens)

        except ImportError:
            # TextIteratorStreamer not available, fall back to non-streaming
            pass

        # Non-streaming fallback
        if progress_cb:
            progress_cb(10)

        with torch.no_grad():
            output = self._model.generate(input_ids, **gen_kwargs)

        new_tokens = output[0][input_len:]
        result = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        if token_cb:
            token_cb(result)
        if progress_cb:
            progress_cb(100)

        return result

    def generate_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 2,
        **kwargs,
    ) -> str:
        """Generate with automatic retry on failure."""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return self.generate(system_prompt, user_prompt, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(0.5)
        raise last_error


# -- High-Level Generation Functions -------------------------------------------

def generate_lyrics(
    prompt: str,
    genre_id: str = "pop",
    mood: str = "",
    language: str = "en",
    structure_override: str = "",
    model_id: str = None,
    progress_cb: Callable = None,
    step_cb: Callable = None,
    log_cb: Callable = None,
    cancel_event: threading.Event = None,
    token_cb: Callable = None,
    **kwargs,
) -> dict:
    """High-level lyrics generation function for use with InferenceWorker."""
    from engines.lyrics_templates import build_generation_prompt

    settings = Settings()
    if model_id is None:
        model_id = settings.get("lyrics.model_id", "llama-3.1-8b-q4")

    temperature = kwargs.pop("temperature", settings.get("lyrics.temperature", 0.8))
    top_p = kwargs.pop("top_p", settings.get("lyrics.top_p", 0.92))
    top_k = kwargs.pop("top_k", settings.get("lyrics.top_k", 50))
    repeat_penalty = kwargs.pop("repeat_penalty", settings.get("lyrics.repeat_penalty", 1.1))
    max_tokens = kwargs.pop("max_tokens", settings.get("lyrics.max_tokens", 2048))

    if step_cb:
        step_cb("Loading lyrics model...")
    if log_cb:
        log_cb(f"Loading model: {model_id}")

    mgr = ModelManager()
    llm = LyricsLLM()

    def _loader():
        llm.load(model_id=model_id)
        return llm

    mgr.load_model(model_id, loader_fn=_loader)

    if cancel_event and cancel_event.is_set():
        return {"lyrics": "", "cancelled": True}

    if step_cb:
        backend_name = llm.backend or "unknown"
        step_cb(f"Generating lyrics ({backend_name})...")

    system_prompt, user_prompt = build_generation_prompt(
        user_prompt=prompt,
        genre_id=genre_id,
        mood=mood,
        language=language,
        structure_override=structure_override,
    )

    if log_cb:
        log_cb(f"Genre: {genre_id}, Mood: {mood}, Backend: {llm.backend}")

    lyrics = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        max_tokens=max_tokens,
        cancel_event=cancel_event,
        progress_cb=progress_cb,
        token_cb=token_cb,
    )

    return {
        "lyrics": lyrics.strip(),
        "genre": genre_id,
        "mood": mood,
        "language": language,
        "model_id": model_id,
        "backend": llm.backend,
        "generation_params": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "max_tokens": max_tokens,
            "structure_override": structure_override,
        },
    }


def generate_lyrics_quick(
    description: str,
    progress_cb: Callable = None,
    step_cb: Callable = None,
    log_cb: Callable = None,
    cancel_event: threading.Event = None,
    token_cb: Callable = None,
    **kwargs,
) -> dict:
    """Quick mode generation -- auto-detect genre and structure from description."""
    from engines.lyrics_templates import build_quick_prompt

    settings = Settings()
    model_id = kwargs.pop("model_id", settings.get("lyrics.model_id", "llama-3.1-8b-q4"))
    temperature = kwargs.pop("temperature", settings.get("lyrics.temperature", 0.8))
    top_p = kwargs.pop("top_p", settings.get("lyrics.top_p", 0.92))
    max_tokens = kwargs.pop("max_tokens", settings.get("lyrics.max_tokens", 2048))

    if step_cb:
        step_cb("Loading lyrics model...")

    mgr = ModelManager()
    llm = LyricsLLM()

    def _loader():
        llm.load(model_id=model_id)
        return llm

    mgr.load_model(model_id, loader_fn=_loader)

    if cancel_event and cancel_event.is_set():
        return {"lyrics": "", "cancelled": True}

    if step_cb:
        step_cb("Writing lyrics...")

    system_prompt, user_prompt = build_quick_prompt(description)

    lyrics = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        cancel_event=cancel_event,
        progress_cb=progress_cb,
        token_cb=token_cb,
    )

    return {
        "lyrics": lyrics.strip(),
        "genre": "auto",
        "mood": "",
        "language": "en",
        "model_id": model_id,
        "backend": llm.backend,
        "generation_params": {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "mode": "quick",
        },
    }


def regenerate_section(
    full_lyrics: str,
    section_tag: str,
    genre_id: str = "pop",
    mood: str = "",
    progress_cb: Callable = None,
    step_cb: Callable = None,
    log_cb: Callable = None,
    cancel_event: threading.Event = None,
    token_cb: Callable = None,
    **kwargs,
) -> dict:
    """Regenerate a specific section of existing lyrics."""
    from engines.lyrics_templates import GENRE_TEMPLATES, BASE_SYSTEM_PROMPT

    settings = Settings()
    model_id = kwargs.pop("model_id", settings.get("lyrics.model_id", "llama-3.1-8b-q4"))
    temperature = kwargs.pop("temperature", settings.get("lyrics.temperature", 0.8) + 0.1)

    mgr = ModelManager()
    current = mgr.current_model
    if not isinstance(current, LyricsLLM) or not current.is_loaded:
        llm = LyricsLLM()

        def _loader():
            llm.load(model_id=model_id)
            return llm

        current = mgr.load_model(model_id, loader_fn=_loader)

    if step_cb:
        step_cb(f"Rewriting {section_tag}...")

    template = GENRE_TEMPLATES.get(genre_id, GENRE_TEMPLATES["pop"])

    system = BASE_SYSTEM_PROMPT + f"""

GENRE: {template.name}
STYLE: {template.vocabulary_style}

You are rewriting ONLY the [{section_tag}] section of existing lyrics.
Keep the same theme, mood, and style. Output ONLY the new section content (no tags, no other sections).
Make it fresh and different from the original while fitting the song."""

    user_msg = f"""Here are the full lyrics:

{full_lyrics}

Rewrite ONLY the [{section_tag}] section. Output just the new lines for that section, nothing else."""

    new_section = current.generate(
        system_prompt=system,
        user_prompt=user_msg,
        temperature=temperature,
        max_tokens=512,
        cancel_event=cancel_event,
        progress_cb=progress_cb,
        token_cb=token_cb,
    )

    return {
        "section_tag": section_tag,
        "new_content": new_section.strip(),
        "genre": genre_id,
    }


# -- Model Loader for ModelManager Registry ------------------------------------

def load_model(cache_dir: str = None, model_id: str = None, **kwargs) -> LyricsLLM:
    """Loader function called by ModelManager._dynamic_load()."""
    if model_id is None:
        settings = Settings()
        model_id = settings.get("lyrics.model_id", "llama-3.1-8b-q4")
    llm = LyricsLLM()
    llm.load(model_id=model_id)
    return llm
