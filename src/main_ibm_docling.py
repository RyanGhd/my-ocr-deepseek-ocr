from docling.document_converter import DocumentConverter
from pathlib import Path

converter = DocumentConverter()
image_path = Path(__file__).resolve().parent / "images/converted/13.png"
result = converter.convert(str(image_path))

doc = result.document

# for item, level in doc.iter_recurse():
#     if hasattr(item, 'bbox'):  # Has bounding box
#         bbox = item.bbox
#         print(f"Type: {type(item).__name__}")
#         print(f"Text: {item.get_text()[:100] if hasattr(item, 'get_text') else 'N/A'}")
#         print(f"Bbox: x0={bbox.l}, y0={bbox.t}, x1={bbox.r}, y1={bbox.b}")
#         print("---")

print("----- Exported Markdown -----")      
# Get markdown output
markdown = result.document.export_to_markdown()
print(markdown)
