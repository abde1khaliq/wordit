import gradio as gr
from transformers import AutoModel, AutoTokenizer
import torch
import os
import sys
import tempfile
import shutil
from PIL import Image, ImageDraw, ImageFont, ImageOps
import fitz
import re
import numpy as np
import base64
from io import StringIO, BytesIO

MODEL_NAME = 'deepseek-ai/DeepSeek-OCR-2'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, _attn_implementation='flash_attention_2',
                                  torch_dtype=torch.bfloat16, trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda()

BASE_SIZE = 1024
IMAGE_SIZE = 768
CROP_MODE = True

TASK_PROMPTS = {
    "📋 Markdown": {"prompt": "<image>\n<|grounding|>Convert the document to markdown.", "has_grounding": True},
    "📝 Free OCR": {"prompt": "<image>\nFree OCR.", "has_grounding": False},
    "📍 Locate": {"prompt": "<image>\nLocate <|ref|>text<|/ref|> in the image.", "has_grounding": True},
    "🔍 Describe": {"prompt": "<image>\nDescribe this image in detail.", "has_grounding": False},
    "✏️ Custom": {"prompt": "", "has_grounding": False}
}


def extract_grounding_references(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    return re.findall(pattern, text, re.DOTALL)


def draw_bounding_boxes(image, refs, extract_images=False):
    img_w, img_h = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    crops = []

    color_map = {}
    np.random.seed(42)

    for ref in refs:
        label = ref[1]
        if label not in color_map:
            color_map[label] = (np.random.randint(50, 255), np.random.randint(
                50, 255), np.random.randint(50, 255))

        color = color_map[label]
        coords = eval(ref[2])
        color_a = color + (60,)

        for box in coords:
            x1, y1, x2, y2 = int(
                box[0]/999*img_w), int(box[1]/999*img_h), int(box[2]/999*img_w), int(box[3]/999*img_h)

            if extract_images and label == 'image':
                crops.append(image.crop((x1, y1, x2, y2)))

            width = 5 if label == 'title' else 3
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            draw2.rectangle([x1, y1, x2, y2], fill=color_a)

            text_bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            ty = max(0, y1 - 20)
            draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 4], fill=color)
            draw.text((x1 + 2, ty + 2), label, font=font, fill=(255, 255, 255))

    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw, crops


def clean_output(text, include_images=False):
    if not text:
        return ""
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    img_num = 0

    for match in matches:
        if '<|ref|>image<|/ref|>' in match[0]:
            if include_images:
                text = text.replace(
                    match[0], f'\n\n**[Figure {img_num + 1}]**\n\n', 1)
                img_num += 1
            else:
                text = text.replace(match[0], '', 1)
        else:
            text = re.sub(
                rf'(?m)^[^\n]*{re.escape(match[0])}[^\n]*\n?', '', text)

    text = text.replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

    return text.strip()


def embed_images(markdown, crops):
    if not crops:
        return markdown
    for i, img in enumerate(crops):
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        markdown = markdown.replace(
            f'**[Figure {i + 1}]**', f'\n\n![Figure {i + 1}](data:image/png;base64,{b64})\n\n', 1)
    return markdown


def process_image(image, task, custom_prompt):
    if image is None:
        return "Error: Upload an image", "", "", None, []
    if task in ["✏️ Custom", "📍 Locate"] and not custom_prompt.strip():
        return "Please enter a prompt", "", "", None, []

    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image = ImageOps.exif_transpose(image)

    if task == "✏️ Custom":
        prompt = f"<image>\n{custom_prompt.strip()}"
        has_grounding = '<|grounding|>' in custom_prompt
    elif task == "📍 Locate":
        prompt = f"<image>\nLocate <|ref|>{custom_prompt.strip()}<|/ref|> in the image."
        has_grounding = True
    else:
        prompt = TASK_PROMPTS[task]["prompt"]
        has_grounding = TASK_PROMPTS[task]["has_grounding"]

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    image.save(tmp.name, 'JPEG', quality=95)
    tmp.close()
    out_dir = tempfile.mkdtemp()

    stdout = sys.stdout
    sys.stdout = StringIO()

    model.infer(
        tokenizer=tokenizer,
        prompt=prompt,
        image_file=tmp.name,
        output_path=out_dir,
        base_size=BASE_SIZE,
        image_size=IMAGE_SIZE,
        crop_mode=CROP_MODE,
        save_results=False
    )

    debug_filters = ['PATCHES', '====', 'BASE:',
                     'directly resize', 'NO PATCHES', 'torch.Size', '%|']
    result = '\n'.join([l for l in sys.stdout.getvalue().split('\n')
                        if l.strip() and not any(s in l for s in debug_filters)]).strip()
    sys.stdout = stdout

    os.unlink(tmp.name)
    shutil.rmtree(out_dir, ignore_errors=True)

    if not result:
        return "No text detected", "", "", None, []

    cleaned = clean_output(result, False)
    markdown = clean_output(result, True)

    img_out = None
    crops = []

    if has_grounding and '<|ref|>' in result:
        refs = extract_grounding_references(result)
        if refs:
            img_out, crops = draw_bounding_boxes(image, refs, True)

    markdown = embed_images(markdown, crops)

    return cleaned, markdown, result, img_out, crops


def process_pdf(path, task, custom_prompt, page_num):
    doc = fitz.open(path)
    total_pages = len(doc)
    if page_num < 1 or page_num > total_pages:
        doc.close()
        return f"Invalid page number. PDF has {total_pages} pages.", "", "", None, []
    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72), alpha=False)
    img = Image.open(BytesIO(pix.tobytes("png")))
    doc.close()

    return process_image(img, task, custom_prompt)


def process_file(path, task, custom_prompt, page_num):
    if not path:
        return "Error: Upload a file", "", "", None, []
    if path.lower().endswith('.pdf'):
        return process_pdf(path, task, custom_prompt, page_num)
    else:
        return process_image(Image.open(path), task, custom_prompt)


def toggle_prompt(task):
    if task == "✏️ Custom":
        return gr.update(visible=True, label="Custom Prompt", placeholder="Add <|grounding|> for bounding boxes")
    elif task == "📍 Locate":
        return gr.update(visible=True, label="Text to Locate", placeholder="Enter text to locate")
    return gr.update(visible=False)


def select_boxes(task):
    if task == "📍 Locate":
        return gr.update(selected="tab_boxes")
    return gr.update()


def get_pdf_page_count(file_path):
    if not file_path or not file_path.lower().endswith('.pdf'):
        return 1
    doc = fitz.open(file_path)
    count = len(doc)
    doc.close()
    return count


def load_image(file_path, page_num=1):
    if not file_path:
        return None
    if file_path.lower().endswith('.pdf'):
        doc = fitz.open(file_path)
        page_idx = max(0, min(int(page_num) - 1, len(doc) - 1))
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72), alpha=False)
        img = Image.open(BytesIO(pix.tobytes("png")))
        doc.close()
        return img
    else:
        return Image.open(file_path)


def update_page_selector(file_path):
    if not file_path:
        return gr.update(visible=False)
    if file_path.lower().endswith('.pdf'):
        page_count = get_pdf_page_count(file_path)
        return gr.update(visible=True, maximum=page_count, value=1, minimum=1,
                         label=f"Select Page (1-{page_count})")
    return gr.update(visible=False)


with gr.Blocks(title="DeepSeek-OCR-2") as demo:
    gr.Markdown("""
    # 🚀 DeepSeek-OCR-2 Demo
    **Convert documents to markdown, extract text, parse figures, and locate specific content with bounding boxes.** 
    **Model uses DeepEncoder v2 and achieves 91.09% on OmniDocBench (+3.73% over v1).** 
    
    **Hope this tool was helpful! If so, a quick like ❤️ would mean a lot :)**
    """)

    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(label="Upload Image or PDF", file_types=[
                              "image", ".pdf"], type="filepath")
            input_img = gr.Image(label="Input Image", type="pil", height=300)
            page_selector = gr.Number(
                label="Select Page", value=1, minimum=1, step=1, visible=False)
            task = gr.Dropdown(list(TASK_PROMPTS.keys()),
                               value="📋 Markdown", label="Task")
            prompt = gr.Textbox(label="Prompt", lines=2, visible=False)
            btn = gr.Button("Extract", variant="primary", size="lg")

        with gr.Column(scale=2):
            with gr.Tabs() as tabs:
                with gr.Tab("Text", id="tab_text"):
                    text_out = gr.Textbox(
                        lines=20, buttons=["copy"], show_label=False)
                with gr.Tab("Markdown Preview", id="tab_markdown"):
                    md_out = gr.Markdown("")
                with gr.Tab("Boxes", id="tab_boxes"):
                    img_out = gr.Image(
                        type="pil", height=500, show_label=False)
                with gr.Tab("Cropped Images", id="tab_crops"):
                    gallery = gr.Gallery(
                        show_label=False, columns=3, height=400)
                with gr.Tab("Raw Text", id="tab_raw"):
                    raw_out = gr.Textbox(lines=20, buttons=[
                                         "copy"], show_label=False)

    gr.Examples(
        examples=[
            ["examples/ocr.jpg", "📋 Markdown", ""],
            ["examples/reachy-mini.jpg", "📍 Locate", "Robot"]
        ],
        inputs=[input_img, task, prompt],
        cache_examples=False
    )

    with gr.Accordion("ℹ️ Info", open=False):
        gr.Markdown("""
        ### Configuration
        1024 base + 768 patches with dynamic cropping (2-6 patches). 144 tokens per patch + 256 base tokens.
        
        ### Tasks
        - **Markdown**: Convert document to structured markdown with layout detection (grounding ✅)
        - **Free OCR**: Simple text extraction without layout
        - **Locate**: Find and highlight specific text/elements in image (grounding ✅)
        - **Describe**: General image description
        - **Custom**: Your own prompt
        
        ### Special Tokens
        - `<image>` - Placeholder where visual tokens are inserted
        - `<|grounding|>` - Enables layout detection with bounding boxes
        - `<|ref|>text<|/ref|>` - Reference text to locate in the image
    
        """)

    file_in.change(load_image, [file_in, page_selector], [input_img])
    file_in.change(update_page_selector, [file_in], [page_selector])
    page_selector.change(load_image, [file_in, page_selector], [input_img])
    task.change(toggle_prompt, [task], [prompt])
    task.change(select_boxes, [task], [tabs])

    def run(image, file_path, task, custom_prompt, page_num):
        if file_path:
            return process_file(file_path, task, custom_prompt, int(page_num))
        if image is not None:
            return process_image(image, task, custom_prompt)
        return "Error: Upload a file or image", "", "", None, []

    submit_event = btn.click(run, [input_img, file_in, task, prompt, page_selector],
                             [text_out, md_out, raw_out, img_out, gallery])
    submit_event.then(select_boxes, [task], [tabs])

if __name__ == "__main__":
    demo.queue(max_size=20).launch(theme=gr.themes.Soft())
