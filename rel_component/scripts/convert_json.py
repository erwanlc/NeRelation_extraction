import json
import typer
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import spacy

nlp = spacy.load("en_core_web_trf")

def get_relation(token, label_mapping):
    return {
            "child": label_mapping[token["to_id"]], 
            "head": label_mapping[token["from_id"]], 
            "relationLabel": token["labels"][0].replace("org:", "")
            }

def get_token(token, doc):
    start = token["value"]["start"]
    end = token["value"]["end"]
    token_i = doc.char_span(start, end)
    token_i = [t.i for t in token_i]
    return {
        "text": token["value"]["text"], 
        "start": start, "end": end, 
        "token_start": token_i[0], 
        "token_end": token_i[-1], 
        "entityLabel": token["value"]["labels"][0]
        }

def generate_json(data, output):
    json_data = json.loads(data.to_json(orient="records"))
    for record in tqdm(json_data):
        input_text = record["document"]
        doc = nlp(input_text)
        new_tokens_list = []
        new_relations_list = []
        label_mapping = {}
        for token in record["tokens"]:
            if "value" in token:
                new_token = get_token(token, doc)
                label_mapping[token["id"]] = new_token["token_start"]
                new_tokens_list.append(new_token)
            elif "from_id" in token:
                new_relations_list.append(get_relation(token, label_mapping))
            else:
                pass
        record["tokens"] = new_tokens_list
        record["relations"] = new_relations_list
    with open(output, "w") as outfile:
        json.dump(json_data, outfile)

def main(input_path: Path, train_path: Path, dev_path: Path):
    df = pd.read_json(input_path)
    df["document"] = pd.json_normalize(df.data)
    df_annotations = pd.json_normalize(df["annotations"].explode())
    df["tokens"] = df_annotations["result"]
    df = df[["document", "tokens"]]

    df_dev = df.sample(frac=0.2)
    df_train = df[~df.index.isin(df_dev.index)]

    generate_json(df_train, train_path)
    generate_json(df_dev, dev_path)

if __name__ == "__main__":
    typer.run(main)