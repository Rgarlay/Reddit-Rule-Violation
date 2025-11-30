from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from uvicorn import run as app_run
import pandas as pd
import sys
from reddit_data.utils.main_utils.transformation_utils import CleaningEmbed
from reddit_data.utils.main_utils.utils import load_pickle_file
import torch
from reddit_data.exception.exception import CustomException

model = load_pickle_file("final_obj/model.pkl")

app = FastAPI()

# Point to templates folder
templates = Jinja2Templates(directory="templates")
# Home route - shows the form
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# When user clicks "Run", the form submits here
@app.post("/predict")
async def predict(
    user_text: str = Form(...), option: str = Form(...)):
    print("User text:", user_text)
    print("Selected option:", option)
    clean_and_embed = CleaningEmbed()
    df = pd.DataFrame({'body':[user_text],'rule':[option]})

    text = df['body'].tolist()
    rule_text = df['rule'].tolist()

    # Preprocess
    cleaned_text = clean_and_embed.clean_text(text)
    embedded_text = clean_and_embed.embed_text(cleaned_text, padding='max_length')

    # Embed
    embedded_rule = clean_and_embed.embed_text(text_list=rule_text)

    # Convert to torch tensors
    body_emb = torch.tensor(embedded_text, dtype=torch.float32)
    rule_emb = torch.tensor(embedded_rule, dtype=torch.float32)

    # Set model to eval
    model.eval()

    with torch.no_grad():
        output = model(body_emb, rule_emb)
        pred = torch.argmax(output, dim=1).item()

    return {"prediction": int(pred)}

if __name__ == "__main__":
    try:
        app_run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        raise CustomException(e,sys)