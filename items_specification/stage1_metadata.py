import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY", "sk-0fc24c229d174d1b99624a49544b1c7a"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

SYSTEM_PROMPT = """You are an expert document analyst specialising in Indonesian construction project documents.
You will receive images of the first few pages of a scanned construction PDF.

Your task is to return a single JSON object with these keys:

{
  "nama_proyek":    "<project name, or null>",
  "nama_site":      "<site name, or null>",
  "lokasi":         "<location / address, or null>",
  "nama_bangunan":  "<building name, or null>",
  "jenis_pekerjaan": ["<work type 1>", "..."],
  "context_prompt": "<see below>"
}

For context_prompt, write a concise description (2-5 sentences) of what construction item /
material specification tables look like in THIS specific document: note the column names
(e.g. No., Uraian, Nama Barang, Spesifikasi, Merek/Brand, Satuan, Volume, etc.), the language
used (Indonesian / English / mixed), typical formatting, and any other patterns a downstream
model should know to reliably extract rows from those tables.

Respond with valid JSON only — no markdown fences, no extra commentary."""


def extract_project_metadata(page_images: list[dict]) -> dict:
    """
    Send the first few page images to qwen-vl-max to extract project metadata
    and generate a tailored context_prompt for Stage 2.

    Args:
        page_images: List of dicts with keys "page_num" and "base64_image".
                     Typically the first 3 pages (or fewer).

    Returns:
        Parsed metadata dict including the "context_prompt" key.
    """
    # Build the multimodal content list: interleave page labels with images
    content = []
    for item in page_images:
        content.append({
            "type": "text",
            "text": f"--- Page {item['page_num']} ---",
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{item['base64_image']}",
            },
        })

    content.append({
        "type": "text",
        "text": (
            "Based on all pages shown above, extract the project metadata and "
            "write the context_prompt as described in the system instructions. "
            "Return valid JSON only."
        ),
    })

    response = client.chat.completions.create(
        model="qwen-vl-max",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if the model ignored the instruction
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    metadata = json.loads(raw)
    return metadata
