from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from uvicorn import run as app_run
import pandas as pd
import sys
from reddit_data.utils.main_utils.utils import clean_text, tokenization_of_text
import torch
from reddit_data.exception.exception import CustomException
from reddit_data.utils.ml_utils.main_ml_utils import ViolationClassifier
import sentencepiece as spm

app = FastAPI()



body_tokenizer_obj = spm.SentencePieceProcessor()
body_tokenizer_obj.load(r'final_obj\body_tokenizer.model')

rule_tokenizer_obj = spm.SentencePieceProcessor()
rule_tokenizer_obj.load(r'final_obj\rule_tokenizer.model')

pad_id = body_tokenizer_obj.piece_to_id("<pad>")


# Point to templates folder
templates = Jinja2Templates(directory="templates")
# Home route - shows the form
@app.get("/home", response_class=HTMLResponse)
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
    df = pd.DataFrame({'body':[user_text],'rule':[option]})

    body_text = df['body'].tolist()
    rule_text = df['rule'].tolist()

    # Preprocess
    cleaned_body_text = clean_text(body_text)
    tokenized_body_text = tokenization_of_text(tokenizer=body_tokenizer_obj, file_name=cleaned_body_text, pad_id=pad_id)


    # Embed
    tokenized_rule_text = tokenization_of_text(tokenizer=rule_tokenizer_obj, file_name=rule_text, pad_id=pad_id)

    # Convert to torch tensors
    body_emb = torch.tensor(tokenized_body_text, dtype=torch.long)
    rule_emb = torch.tensor(tokenized_rule_text, dtype=torch.long)

    # Set model to eval
    model = ViolationClassifier(pad_id=pad_id)
    state_dict = torch.load("final_obj/model_weights.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.to("cpu")
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