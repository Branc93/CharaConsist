import os
# Avoid transformers importing torchvision (prevents torchvision::nms issues)
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
# Prefer disabling xformers auto-import if not correctly installed
os.environ.setdefault("XFORMERS_DISABLED", "1")
import threading
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
import gradio as gr

from models.pipeline_characonsist import CharaConsistPipeline
from models.attention_processor_characonsist import (
    reset_attn_processor,
    set_text_len,
    reset_id_bank,
)


# -----------------------------
# Global pipeline (lazy-loaded)
# -----------------------------
_PIPELINE_LOCK = threading.Lock()
_PIPELINE = None
_PIPELINE_MODEL_PATH = None


def _enable_optimizations(pipe: CharaConsistPipeline) -> None:
    # xFormers (optional)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    # VAE slicing
    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass

    # CPU offload
    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        pass


def _load_pipeline(model_path: str, hf_token: Optional[str]) -> CharaConsistPipeline:
    pipe = CharaConsistPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        use_safetensors=True,
        token=hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
    )
    _enable_optimizations(pipe)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _ensure_flux_weights(auto_download: bool, custom_model_path: Optional[str], hf_token: Optional[str]) -> str:
    """Ensure FLUX.1-dev weights are available locally and return a path or repo id for loading.

    If auto_download is True or custom_model_path is empty, download to ./flux_model when missing.
    """
    if not auto_download and custom_model_path:
        return custom_model_path

    local_dir = os.path.abspath("./flux_model")
    if os.path.isdir(local_dir) and any(os.scandir(local_dir)):
        return local_dir

    # Download using DiffusionPipeline, then save for reuse
    from diffusers import DiffusionPipeline

    dp = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.float16,
        use_safetensors=True,
        token=hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
    )
    dp.save_pretrained(local_dir, safe_serialization=True)
    return local_dir


def _ensure_pipeline(model_path: Optional[str], auto_download: bool, hf_token: Optional[str]) -> CharaConsistPipeline:
    global _PIPELINE, _PIPELINE_MODEL_PATH
    source_path = _ensure_flux_weights(auto_download, model_path, hf_token)
    with _PIPELINE_LOCK:
        if _PIPELINE is None or _PIPELINE_MODEL_PATH != source_path:
            _PIPELINE = _load_pipeline(source_path, hf_token)
            _PIPELINE_MODEL_PATH = source_path
        return _PIPELINE


# -----------------------------
# Utilities
# -----------------------------
def _overlay_mask_on_image(image: Image.Image, mask: np.ndarray, color: Tuple[int, int, int]) -> Image.Image:
    img_array = np.array(image).astype(np.float32) * 0.5
    mask_zero = np.zeros_like(img_array)

    mask_resized = Image.fromarray(mask.astype(np.uint8))
    mask_resized = mask_resized.resize(image.size, Image.NEAREST)
    mask_resized = np.array(mask_resized)
    mask_resized = mask_resized[:, :, None]
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, -1)
    mask_resized_color = mask_resized * color_arr
    img_array = img_array + mask_resized_color * 0.5
    mask_zero = mask_zero + mask_resized_color
    out_img = np.concatenate([img_array, mask_zero], axis=1)
    out_img[out_img > 255] = 255
    out_img = out_img.astype(np.uint8)
    return Image.fromarray(out_img)


def _get_text_tokens_length(pipe: CharaConsistPipeline, text: str) -> int:
    # Use the pipeline's tokenizer_2, matching inference.py logic
    text_mask = pipe.tokenizer_2(
        text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    ).attention_mask
    return int(text_mask.sum().item() - 1)


def _modify_prompt_and_get_length(bg: str, fg: str, act: str, pipe: CharaConsistPipeline) -> Tuple[str, int, int]:
    bg = (bg or "") + " "
    fg = (fg or "") + " "
    prompt = bg + fg + (act or "")
    return prompt, _get_text_tokens_length(pipe, bg), _get_text_tokens_length(pipe, prompt)


def _run_sequence(
    pipe: CharaConsistPipeline,
    prompts: List[str],
    bg_lens: List[int],
    real_lens: List[int],
    *,
    height: int,
    width: int,
    seed: int,
    use_interpolate: bool,
    share_bg: bool,
    num_inference_steps: int,
    guidance_scale: float,
    return_masks_overlay: bool,
) -> List[Image.Image]:
    # Reset attention processors to current latent spatial size
    reset_attn_processor(pipe, size=(height // 16, width // 16))

    generator = torch.Generator("cpu").manual_seed(int(seed))

    common_kwargs = dict(
        height=height,
        width=width,
        use_interpolate=use_interpolate,
        share_bg=share_bg,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    images: List[Image.Image] = []

    # ID image
    id_prompt = prompts[0]
    set_text_len(pipe, bg_lens[0], real_lens[0])
    id_images, id_spatial_kwargs = pipe(id_prompt, is_id=True, **common_kwargs)
    id_fg_mask = id_spatial_kwargs.get("curr_fg_mask")
    id_img = id_images[0]
    if return_masks_overlay and id_fg_mask is not None:
        images.append(_overlay_mask_on_image(id_img, id_fg_mask[0].cpu().numpy(), (255, 0, 0)))
    else:
        images.append(id_img)

    # Frames
    spatial_kwargs = dict(id_fg_mask=id_fg_mask, id_bg_mask=~id_fg_mask) if id_fg_mask is not None else {}
    for idx, prompt in enumerate(prompts[1:]):
        set_text_len(pipe, bg_lens[1:][idx], real_lens[1:][idx])
        # optional pre-run
        _, spatial_kwargs = pipe(prompt, is_pre_run=True, spatial_kwargs=spatial_kwargs, **common_kwargs)
        frame_images, spatial_kwargs = pipe(prompt, spatial_kwargs=spatial_kwargs, **common_kwargs)
        frame_img = frame_images[0]
        if return_masks_overlay and spatial_kwargs.get("curr_fg_mask") is not None:
            images.append(
                _overlay_mask_on_image(frame_img, spatial_kwargs["curr_fg_mask"][0].cpu().numpy(), (255, 0, 0))
            )
        else:
            images.append(frame_img)

    reset_id_bank(pipe)
    return images


# -----------------------------
# UI Handlers
# -----------------------------
def _parse_lines(text: str) -> List[str]:
    if not text:
        return []
    # Split on newlines or semicolons
    parts = [p.strip() for p in text.replace(";", "\n").splitlines()]
    return [p for p in parts if p]


def ui_gen_bg_fg(
    model_path: str,
    auto_download: bool,
    hf_token: str,
    foreground: str,
    background: str,
    actions_text: str,
    height: int,
    width: int,
    seed: int,
    num_steps: int,
    guidance: float,
    use_interpolate: bool,
    save_masks_overlay: bool,
):
    pipe = _ensure_pipeline(model_path, auto_download, hf_token)
    actions = _parse_lines(actions_text)
    if len(actions) == 0:
        actions = [""]

    # Build prompts: first is ID, then one for each action
    prompts: List[str] = []
    bg_lens: List[int] = []
    real_lens: List[int] = []
    for idx, act in enumerate([actions[0]] + actions[1:]):
        prompt, bg_len, real_len = _modify_prompt_and_get_length(background, foreground, act, pipe)
        # Skip duplicating the first action (already ID) in frames
        if idx == 0:
            prompts.append(prompt)
            bg_lens.append(bg_len)
            real_lens.append(real_len)
        else:
            prompts.append(prompt)
            bg_lens.append(bg_len)
            real_lens.append(real_len)

    return _run_sequence(
        pipe,
        prompts,
        bg_lens,
        real_lens,
        height=height,
        width=width,
        seed=seed,
        use_interpolate=use_interpolate,
        share_bg=True,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        return_masks_overlay=save_masks_overlay,
    )


def ui_gen_fg_only(
    model_path: str,
    auto_download: bool,
    hf_token: str,
    foreground: str,
    backgrounds_text: str,
    actions_text: str,
    height: int,
    width: int,
    seed: int,
    num_steps: int,
    guidance: float,
    use_interpolate: bool,
    save_masks_overlay: bool,
):
    pipe = _ensure_pipeline(model_path, auto_download, hf_token)
    backgrounds = _parse_lines(backgrounds_text)
    actions = _parse_lines(actions_text)
    if len(backgrounds) == 0:
        backgrounds = [""]
    if len(actions) == 0:
        actions = [""]

    # Use first pair as ID, then generate for the cartesian or paired list
    id_bg = backgrounds[0]
    id_act = actions[0]

    prompts: List[str] = []
    bg_lens: List[int] = []
    real_lens: List[int] = []

    id_prompt, id_bg_len, id_real_len = _modify_prompt_and_get_length(id_bg, foreground, id_act, pipe)
    prompts.append(id_prompt)
    bg_lens.append(id_bg_len)
    real_lens.append(id_real_len)

    # Frames: pair up to the longest list
    max_len = max(len(backgrounds), len(actions))
    for i in range(max_len):
        bg = backgrounds[i % len(backgrounds)]
        act = actions[i % len(actions)]
        if i == 0:  # avoid duplicating the ID pair
            continue
        prompt, bg_len, real_len = _modify_prompt_and_get_length(bg, foreground, act, pipe)
        prompts.append(prompt)
        bg_lens.append(bg_len)
        real_lens.append(real_len)

    return _run_sequence(
        pipe,
        prompts,
        bg_lens,
        real_lens,
        height=height,
        width=width,
        seed=seed,
        use_interpolate=use_interpolate,
        share_bg=False,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        return_masks_overlay=save_masks_overlay,
    )


def ui_gen_mix(
    model_path: str,
    auto_download: bool,
    hf_token: str,
    foreground: str,
    fixed_background: str,
    variable_backgrounds_text: str,
    actions_text: str,
    height: int,
    width: int,
    seed: int,
    num_steps: int,
    guidance: float,
    use_interpolate: bool,
    save_masks_overlay: bool,
):
    pipe = _ensure_pipeline(model_path, auto_download, hf_token)
    var_bgs = _parse_lines(variable_backgrounds_text)
    actions = _parse_lines(actions_text)
    if len(var_bgs) == 0:
        var_bgs = [""]
    if len(actions) == 0:
        actions = [""]

    # ID uses fixed background only
    prompts: List[str] = []
    bg_lens: List[int] = []
    real_lens: List[int] = []

    id_prompt, id_bg_len, id_real_len = _modify_prompt_and_get_length(fixed_background, foreground, actions[0], pipe)
    prompts.append(id_prompt)
    bg_lens.append(id_bg_len)
    real_lens.append(id_real_len)

    # Frames: fixed bg + each variable bg snippet
    max_len = max(len(var_bgs), len(actions))
    for i in range(max_len):
        bg = (fixed_background + " " + var_bgs[i % len(var_bgs)]).strip()
        act = actions[i % len(actions)]
        if i == 0:  # avoid duplicating ID
            continue
        prompt, bg_len, real_len = _modify_prompt_and_get_length(bg, foreground, act, pipe)
        prompts.append(prompt)
        bg_lens.append(bg_len)
        real_lens.append(real_len)

    # Keep background largely consistent, but allow variation via prompt composition
    return _run_sequence(
        pipe,
        prompts,
        bg_lens,
        real_lens,
        height=height,
        width=width,
        seed=seed,
        use_interpolate=use_interpolate,
        share_bg=True,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        return_masks_overlay=save_masks_overlay,
    )


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="CharaConsist - Gradio UI") as demo:
    gr.Markdown("**CharaConsist**: FLUX.1-based consistent character generation")

    with gr.Tab("Generate Background + Foreground"):
        with gr.Row():
            model_path_bgfg = gr.Textbox(label="Model path (leave empty to auto-download)", placeholder="/path/to/FLUX.1-dev", value="")
        with gr.Row():
            auto_dl_bgfg = gr.Checkbox(value=True, label="Auto-download FLUX.1-dev to ./flux_model if missing")
            token_bgfg = gr.Textbox(label="HF token (if gated)", type="password")
        with gr.Row():
            fg_bgfg = gr.Textbox(label="Foreground (character)")
            bg_bgfg = gr.Textbox(label="Background (fixed)")
        actions_bgfg = gr.Textbox(label="Actions (one per line)")
        with gr.Row():
            height_bgfg = gr.Number(value=1024, precision=0, label="Height")
            width_bgfg = gr.Number(value=1024, precision=0, label="Width")
            seed_bgfg = gr.Number(value=2025, precision=0, label="Seed")
        with gr.Row():
            steps_bgfg = gr.Slider(10, 60, value=50, step=1, label="Steps")
            guidance_bgfg = gr.Slider(0.0, 10.0, value=3.5, step=0.1, label="Guidance")
        with gr.Row():
            interp_bgfg = gr.Checkbox(value=True, label="Use interpolate")
            masks_bgfg = gr.Checkbox(value=False, label="Return masks overlay")
        out_bgfg = gr.Gallery(label="Results", columns=2, height="auto")
        run_bgfg = gr.Button("Generate")

        run_bgfg.click(
            ui_gen_bg_fg,
            inputs=[
                model_path_bgfg,
                auto_dl_bgfg,
                token_bgfg,
                fg_bgfg,
                bg_bgfg,
                actions_bgfg,
                height_bgfg,
                width_bgfg,
                seed_bgfg,
                steps_bgfg,
                guidance_bgfg,
                interp_bgfg,
                masks_bgfg,
            ],
            outputs=[out_bgfg],
        )

    with gr.Tab("Generate Foreground Only"):
        with gr.Row():
            model_path_fg = gr.Textbox(label="Model path (leave empty to auto-download)", placeholder="/path/to/FLUX.1-dev", value="")
        with gr.Row():
            auto_dl_fg = gr.Checkbox(value=True, label="Auto-download FLUX.1-dev to ./flux_model if missing")
            token_fg = gr.Textbox(label="HF token (if gated)", type="password")
        with gr.Row():
            fg_fg = gr.Textbox(label="Foreground (character)")
        backgrounds_fg = gr.Textbox(label="Backgrounds (one per line)")
        actions_fg = gr.Textbox(label="Actions (one per line)")
        with gr.Row():
            height_fg = gr.Number(value=1024, precision=0, label="Height")
            width_fg = gr.Number(value=1024, precision=0, label="Width")
            seed_fg = gr.Number(value=2025, precision=0, label="Seed")
        with gr.Row():
            steps_fg = gr.Slider(10, 60, value=50, step=1, label="Steps")
            guidance_fg = gr.Slider(0.0, 10.0, value=3.5, step=0.1, label="Guidance")
        with gr.Row():
            interp_fg = gr.Checkbox(value=True, label="Use interpolate")
            masks_fg = gr.Checkbox(value=False, label="Return masks overlay")
        out_fg = gr.Gallery(label="Results", columns=2, height="auto")
        run_fg = gr.Button("Generate")

        run_fg.click(
            ui_gen_fg_only,
            inputs=[
                model_path_fg,
                auto_dl_fg,
                token_fg,
                fg_fg,
                backgrounds_fg,
                actions_fg,
                height_fg,
                width_fg,
                seed_fg,
                steps_fg,
                guidance_fg,
                interp_fg,
                masks_fg,
            ],
            outputs=[out_fg],
        )

    with gr.Tab("Generate Mix"):
        with gr.Row():
            model_path_mix = gr.Textbox(label="Model path (leave empty to auto-download)", placeholder="/path/to/FLUX.1-dev", value="")
        with gr.Row():
            auto_dl_mix = gr.Checkbox(value=True, label="Auto-download FLUX.1-dev to ./flux_model if missing")
            token_mix = gr.Textbox(label="HF token (if gated)", type="password")
        with gr.Row():
            fg_mix = gr.Textbox(label="Foreground (character)")
            fixed_bg_mix = gr.Textbox(label="Fixed background")
        var_bgs_mix = gr.Textbox(label="Variable backgrounds (one per line)")
        actions_mix = gr.Textbox(label="Actions (one per line)")
        with gr.Row():
            height_mix = gr.Number(value=1024, precision=0, label="Height")
            width_mix = gr.Number(value=1024, precision=0, label="Width")
            seed_mix = gr.Number(value=2025, precision=0, label="Seed")
        with gr.Row():
            steps_mix = gr.Slider(10, 60, value=50, step=1, label="Steps")
            guidance_mix = gr.Slider(0.0, 10.0, value=3.5, step=0.1, label="Guidance")
        with gr.Row():
            interp_mix = gr.Checkbox(value=True, label="Use interpolate")
            masks_mix = gr.Checkbox(value=False, label="Return masks overlay")
        out_mix = gr.Gallery(label="Results", columns=2, height="auto")
        run_mix = gr.Button("Generate")

        run_mix.click(
            ui_gen_mix,
            inputs=[
                model_path_mix,
                auto_dl_mix,
                token_mix,
                fg_mix,
                fixed_bg_mix,
                var_bgs_mix,
                actions_mix,
                height_mix,
                width_mix,
                seed_mix,
                steps_mix,
                guidance_mix,
                interp_mix,
                masks_mix,
            ],
            outputs=[out_mix],
        )


if __name__ == "__main__":
    # Allow binding to all interfaces by default
    demo.launch()


