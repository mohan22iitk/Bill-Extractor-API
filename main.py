import os
from dotenv import load_dotenv
load_dotenv()
import requests
import uvicorn
import json
from typing import List, Literal, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pdf2image import convert_from_bytes
import google.generativeai as genai
from io import BytesIO

# --- Configuration ---
# You need a Google Cloud API Key for Gemini
# export GOOGLE_API_KEY="your_api_key_here"
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

app = FastAPI()

# --- Pydantic Models (Strict Schema Enforcement) ---

class BillItem(BaseModel):
    item_name: str = Field(..., description="Name of the item exactly as mentioned")
    item_amount: float = Field(..., description="Net Amount of the item post discounts")
    item_rate: float = Field(..., description="Unit rate of the item")
    item_quantity: float = Field(..., description="Quantity of the item")

class PageLineItems(BaseModel):
    page_no: str
    page_type: Literal["Bill Detail", "Final Bill", "Pharmacy"]
    bill_items: List[BillItem]

class ExtractionData(BaseModel):
    pagewise_line_items: List[PageLineItems]
    total_item_count: int
    reconciled_amount: float

class TokenUsage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int

class APIResponse(BaseModel):
    is_success: bool
    token_usage: TokenUsage
    data: ExtractionData

class APIRequest(BaseModel):
    document: str

# --- Helper Functions ---

# def download_file(url: str) -> bytes:
#     response = requests.get(url)
#     if response.status_code != 200:
#         raise HTTPException(status_code=400, detail="Failed to download document")
#     return response.content
def download_file(url: str) -> bytes:
    # --- DEV MODE: Check if it's a local file first ---
    # This allows you to test with files on your hard drive
    if os.path.exists(url):
        print(f"Loading local file: {url}")
        with open(url, "rb") as f:
            return f.read()

    # --- PROD MODE: Download from URL ---
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            # Print the actual error from the web for debugging
            print(f"Download failed: {response.status_code} - {response.text}")
            raise HTTPException(status_code=400, detail="Failed to download document")
        return response.content
    except Exception as e:
        print(f"Network error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

def process_page_with_llm(image, page_num) -> tuple[PageLineItems, TokenUsage]:
    """
    Sends a single page image to Gemini Flash to extract structured data.
    """
    model = genai.GenerativeModel('gemini-2.5-pro',
        generation_config={"response_mime_type": "application/json"}
    )

    prompt = f"""
    You are an expert invoice data extractor. Analyze this image (Page {page_num}).
    
    1. Identify the Page Type: 'Bill Detail', 'Final Bill', or 'Pharmacy'.
    2. Extract LINE ITEMS only. 
       - CRITICAL: Do NOT extract Sub-totals, Tax summaries, or Grand Totals as items. 
       - If a row says "Total" or "Carry Forward", IGNORE IT.
       - Extract Item Name, Amount, Rate, and Quantity. 
       - If Quantity is missing, infer it as 1.0 or 0.0 based on context.
    
    Output JSON strictly matching this schema:
    {{
        "page_no": "{page_num}",
        "page_type": "...",
        "bill_items": [
            {{ "item_name": "...", "item_amount": 0.0, "item_rate": 0.0, "item_quantity": 0.0 }}
        ]
    }}
    """
    
    # Generate response
    response = model.generate_content([prompt, image])
    
    # Parse Usage Metadata
    usage = response.usage_metadata
    tokens = TokenUsage(
        total_tokens=usage.total_token_count,
        input_tokens=usage.prompt_token_count,
        output_tokens=usage.candidates_token_count
    )

    try:
        # Clean JSON if model adds markdown blocks
        text_response = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text_response)
        
        # Validate with Pydantic
        validated_data = PageLineItems(**data)
        return validated_data, tokens
    except Exception as e:
        print(f"Error parsing page {page_num}: {e}")
        # Return empty structure on failure to keep pipeline moving
        return PageLineItems(page_no=str(page_num), page_type="Bill Detail", bill_items=[]), tokens

# --- Main Endpoint ---

@app.post("/extract-bill-data", response_model=APIResponse)
async def extract_bill_data(request: APIRequest):
    try:
        # 1. Download Document
        file_bytes = download_file(request.document)
        
        # 2. Convert to Images (Handle PDF vs Image)
        images = []
        if request.document.lower().endswith(".pdf") or file_bytes[:4] == b'%PDF':
            # Convert PDF to images (requires poppler installed)
            images = convert_from_bytes(file_bytes)
        else:
            # Load as single image (using PIL for consistency if needed, 
            # but Gemini accepts bytes for many formats. Here we assume PDF logic mainly)
            # For simplicity in this snippet, let's assume the input is mostly PDFs as per samples.
            # If image, we would wrap it in a list.
            from PIL import Image
            images = [Image.open(BytesIO(file_bytes))]

        # 3. Process Pages Parallel or Sequential
        all_page_items = []
        cumulative_tokens = {"total": 0, "input": 0, "output": 0}

        for i, img in enumerate(images):
            page_data, tokens = process_page_with_llm(img, i + 1)
            all_page_items.append(page_data)
            
            # Aggregate tokens
            cumulative_tokens["total"] += tokens.total_tokens
            cumulative_tokens["input"] += tokens.input_tokens
            cumulative_tokens["output"] += tokens.output_tokens

        # 4. Calculate Final Counts and Amounts
        total_items = 0
        total_amount = 0.0

        for page in all_page_items:
            # Count items
            total_items += len(page.bill_items)
            
            # Sum amounts
            for item in page.bill_items:
                total_amount += item.item_amount

        # 5. Construct Response
        return APIResponse(
            is_success=True,
            token_usage=TokenUsage(
                total_tokens=cumulative_tokens["total"],
                input_tokens=cumulative_tokens["input"],
                output_tokens=cumulative_tokens["output"]
            ),
            data=ExtractionData(
                pagewise_line_items=all_page_items,
                total_item_count=total_items,
                reconciled_amount=round(total_amount, 2) # <--- Add the sum here
            )
        )

    except Exception as e:
        # Log error in real production
        print(f"Server Error: {str(e)}")
        # Return a failure response compliant with schema structure (or raise 500)
        # For strict schema compliance, we might return is_success=False
        return APIResponse(
            is_success=False,
            token_usage=TokenUsage(total_tokens=0, input_tokens=0, output_tokens=0),
            data=ExtractionData(pagewise_line_items=[], total_item_count=0, reconciled_amount=0.0)
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)