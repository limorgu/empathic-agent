# gpt.py — resilient OpenAI vision helper for EmpathyAgent (macOS-friendly)

import os
import base64
import time
from typing import List
from openai import OpenAI
from openai import APIConnectionError, RateLimitError

# ---- small helpers ---------------------------------------------------------

def _clean_api_key() -> str:
    """Return a sanitized API key and fail fast if malformed."""
    raw = os.getenv("OPENAI_API_KEY", "") or ""
    key = raw.strip().strip('"').strip("'")
    if not key or (not key.startswith("sk-")) or (not key.isascii()):
        raise ValueError(
            "OPENAI_API_KEY is missing/malformed (non-ASCII or wrong quotes). "
            "Re-export it with straight ASCII characters."
        )
    return key


def _evenly_sample(seq: List[str], k: int) -> List[str]:
    if len(seq) <= k:
        return seq
    step = max(1, len(seq) // k)
    return seq[::step][:k]


# ---- main client classes ---------------------------------------------------

class GPT:
    """
    Vision client with payload controls:
      - max 12 frames (evenly spaced) by default
      - optional JPEG reencode+downscale to shrink payload
      - retries with fewer frames on connection/timeout
    """

    def __init__(self, model_name: str):
        # validate model choice
        allowed = {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-vision-preview"}
        if model_name not in allowed:
            raise ValueError(f"Wrong model name: {model_name}. Must be one of {sorted(allowed)}")
        self.model_name = model_name

        # build OpenAI client (timeout + retries)
        base_url = os.environ.get("OPENAI_API_BASE") or None
        self.client = OpenAI(
            api_key=_clean_api_key(),
            base_url=base_url,
            timeout=60,        # seconds
            max_retries=3,     # client-level retries
        )

    # ---- frame handling ----------------------------------------------------

    def base64_encode(self, input_path: str, max_frames: int = 12,
                      target_width: int = 512, jpeg_quality: int = 60) -> List[str]:
        """
        Read up to `max_frames` images from `input_path`, downscale and JPEG-reencode
        (if Pillow available) to keep request bodies small; return base64 strings.
        """
        files = sorted(os.listdir(input_path)) if os.path.isdir(input_path) else []
        if not files:
            print("0 frames read.")
            return []

        files = _evenly_sample(files, max_frames)

        frames: List[str] = []
        try:
            from PIL import Image  # optional – used if available
            from io import BytesIO
            use_pil = True
        except Exception:
            use_pil = False

        for fn in files:
            fp = os.path.join(input_path, fn)
            with open(fp, "rb") as f:
                raw = f.read()

            if use_pil:
                try:
                    im = Image.open(BytesIO(raw)).convert("RGB")
                    if im.width > target_width:
                        new_h = int(im.height * (target_width / im.width))
                        im = im.resize((target_width, new_h))
                    buf = BytesIO()
                    im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
                    raw = buf.getvalue()
                except Exception:
                    # If Pillow fails, keep original bytes
                    pass

            frames.append(base64.b64encode(raw).decode("utf-8"))

        print(f"{len(frames)} frames read.")
        return frames

    # ---- generation --------------------------------------------------------

    def generate(self, script_path: str, text_prompt: str) -> str:
        """
        Build a multimodal prompt (text + sampled frames) and call the chat API.
        Retries with fewer frames on connection/timeouts.
        """
        # Try a few frame budgets, backing off if needed
        budgets = [12, 8, 6, 4]
        last_err = None

        for max_frames in budgets:
            try:
                base64_frames = self.base64_encode(script_path, max_frames=max_frames)

                # Build content parts: first the text, then images (if any)
                content_parts = [{"type": "text", "text": text_prompt}]
                for b64 in base64_frames:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}
                    })

                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=0,
                    max_tokens=300,
                    messages=[{"role": "user", "content": content_parts}],
                )

                msg = completion.choices[0].message.content
                print(msg)
                time.sleep(0.1)  # gentle pacing to avoid rate-limit errors
                return msg

            except (APIConnectionError, TimeoutError) as e:
                last_err = e
                if max_frames > 4:
                    print(f"Connection issue; retrying with fewer frames …")
                    continue
                raise
            except RateLimitError as e:
                # bubble up rate limits (quota exhaustion, etc.)
                raise
            except Exception as e:
                last_err = e
                text = str(e).lower()
                if any(t in text for t in ("connection", "timeout", "read timed out")) and max_frames > 4:
                    print("Transient error; retrying with fewer frames …")
                    continue
                raise

        # If we get here, all attempts failed
        raise last_err or RuntimeError("Unknown error during generate().")


class GPT_text:
    """Plain text client (no images)."""

    def __init__(self, model_name: str):
        allowed = {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-vision-preview", "gpt-3.5-turbo"}
        if model_name not in allowed:
            raise ValueError(f"Wrong model name: {model_name}. Must be one of {sorted(allowed)}")
        self.model_name = model_name

        base_url = os.environ.get("OPENAI_API_BASE") or None
        self.client = OpenAI(
            api_key=_clean_api_key(),
            base_url=base_url,
            timeout=60,
            max_retries=3,
        )

    def generate(self, text_prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            max_tokens=300,
            messages=[{"role": "user", "content": text_prompt}],
        )
        msg = completion.choices[0].message.content
        print(msg)
        time.sleep(0.1)  # gentle pacing
        return msg


# Optional quick test: run this file directly to check text path works.
if __name__ == "__main__":
    demo = os.environ.get("EA_DEMO", "0") == "1"
    if demo:
        t = GPT_text("gpt-4o-mini")
        print(t.generate("Say 'hello' in one short sentence."))
