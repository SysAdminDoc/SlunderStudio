"""
Slunder Studio v0.0.2 — Lyrics Engine
llama-cpp-python wrapper for local LLM inference with streaming token output,
model loading via Model Manager, and generation pipeline.
"""
import os
import threading
import time
from typing import Optional, Callable, Generator
from pathlib import Path

from core.settings import Settings
from core.model_manager import ModelManager, cleanup_gpu


# ── Model File Resolution ──────────────────────────────────────────────────────

# Map model IDs to expected GGUF filenames (most common quantization)
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


def _find_gguf_file(model_id: str) -> Optional[str]:
    """Find the GGUF file path for a model. Searches cache directory."""
    mgr = ModelManager()
    cache_dir = mgr.get_cache_dir(model_id)

    # Check known filenames
    candidates = GGUF_FILENAMES.get(model_id, [])
    for name in candidates:
        path = cache_dir / name
        if path.exists():
            return str(path)

    # Search for any .gguf file in cache
    if cache_dir.exists():
        for f in cache_dir.rglob("*.gguf"):
            return str(f)

    # Check HuggingFace default cache
    info = mgr.get_model_info(model_id)
    if info:
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        model_dir = hf_cache / f"models--{info.source.replace('/', '--')}"
        if model_dir.exists():
            for f in model_dir.rglob("*.gguf"):
                return str(f)

    return None


# ── LLM Wrapper ────────────────────────────────────────────────────────────────

class LyricsLLM:
    """
    Wrapper around llama-cpp-python for lyrics generation.
    Supports streaming, cancellation, and parameter control.
    """

    def __init__(self):
        self._model = None
        self._model_id: Optional[str] = None
        self._model_path: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_id(self) -> Optional[str]:
        return self._model_id

    def load(self, model_id: str = None, model_path: str = None, n_ctx: int = 4096):
        """
        Load a GGUF model for inference.
        Provide either model_id (looked up from registry) or model_path (direct path).
        """
        from core.deps import ensure
        ensure("llama_cpp")
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

        # Unload previous model
        self.unload()

        self._model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,  # Full GPU offload
            verbose=False,
            n_threads=os.cpu_count() or 4,
        )
        self._model_id = model_id or Path(model_path).stem
        self._model_path = model_path

    def unload(self):
        """Unload the model and free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_id = None
            cleanup_gpu()

    def cleanup(self):
        """Alias for unload, used by ModelManager."""
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
        """
        Generate lyrics from prompts. Returns complete text.

        Args:
            system_prompt: System instructions for the LLM
            user_prompt: User's creative request
            temperature: Randomness (0.1-2.0)
            top_p: Nucleus sampling threshold
            top_k: Top-K sampling
            repeat_penalty: Penalize repetition
            max_tokens: Maximum output tokens
            cancel_event: Threading event to check for cancellation
            progress_cb: Callback for progress percentage (0-100)
            token_cb: Callback for each generated token (for streaming UI)
        """
        if not self.is_loaded:
            raise RuntimeError("No model loaded. Call load() first.")

        # Build messages in chat format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Stream generation
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
            # Check cancellation
            if cancel_event and cancel_event.is_set():
                break

            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                output_tokens.append(content)
                token_count += 1

                # Stream to UI
                if token_cb:
                    token_cb(content)

                # Progress estimate (rough — we don't know total tokens ahead of time)
                if progress_cb:
                    # Estimate based on typical lyrics length (~200-400 tokens)
                    estimated_total = max(200, max_tokens // 4)
                    pct = min(95, int(token_count / estimated_total * 100))
                    progress_cb(pct)

        if progress_cb:
            progress_cb(100)

        return "".join(output_tokens)

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


# ── High-Level Generation Functions ────────────────────────────────────────────
# These are designed to be called from InferenceWorker threads.

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
    """
    High-level lyrics generation function for use with InferenceWorker.

    Returns dict with:
        - lyrics: str (generated text)
        - genre: str
        - mood: str
        - model_id: str
        - generation_params: dict
    """
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

    # Load model via ModelManager
    mgr = ModelManager()
    llm = LyricsLLM()

    def _loader():
        llm.load(model_id=model_id)
        return llm

    mgr.load_model(model_id, loader_fn=_loader)

    if cancel_event and cancel_event.is_set():
        return {"lyrics": "", "cancelled": True}

    if step_cb:
        step_cb("Generating lyrics...")

    # Build prompts
    system_prompt, user_prompt = build_generation_prompt(
        user_prompt=prompt,
        genre_id=genre_id,
        mood=mood,
        language=language,
        structure_override=structure_override,
    )

    if log_cb:
        log_cb(f"Genre: {genre_id}, Mood: {mood}, Temp: {temperature}")

    # Generate
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
    """
    Quick mode generation — auto-detect genre and structure from a simple description.
    """
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
    """
    Regenerate a specific section of existing lyrics.
    The LLM receives the full lyrics context and rewrites only the target section.
    """
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


# ── Model Loader for ModelManager Registry ─────────────────────────────────────

def load_model(cache_dir: str = None, model_id: str = None, **kwargs) -> LyricsLLM:
    """
    Loader function called by ModelManager._dynamic_load().
    Loads the specified lyrics model, falling back to settings default.
    """
    if model_id is None:
        settings = Settings()
        model_id = settings.get("lyrics.model_id", "llama-3.1-8b-q4")
    llm = LyricsLLM()
    llm.load(model_id=model_id)
    return llm
