import argparse
import base64
import mimetypes

from lib.llm_client import (
    MODEL,
    get_client,
)


def describe_image_prompt() -> str:
    return """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
"""


def describe_image(query: str, image: bytes, mime: str) -> tuple[str, int | None]:
    data_url = f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": describe_image_prompt()},
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": query},
                ],
            }
        ],
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("LLM returned an empty response")

    usage = response.usage.total_tokens if response.usage is not None else None

    return content, usage


def main() -> int:
    parser = argparse.ArgumentParser(description="Describe Image CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--query", type=str, required=True, help="Question to ask about the image")

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, mode="rb") as f:
        image_bytes = f.read()

    try:
        content, usage = describe_image(args.query, image_bytes, mime)
    except Exception as e:
        print(e)
        return 1

    print(f"Rewritten query: {content.strip()}")
    if usage is not None:
        print(f"Total tokens:    {usage}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
