"""Convert textcat annotation from JSONL to spaCy v3 .spacy format."""
from pathlib import Path

import pandas as pd
import spacy
import typer
from sklearn.model_selection import train_test_split
from spacy.tokens import DocBin


def convert(lang: str, input_path: Path, train_path: Path, test_path: Path) -> None:
    """Conversion du fichier des commentaires annotés en format binaire Spacy."""
    print(f"Lecture de {input_path}")
    nlp = spacy.blank(lang)

    responses = pd.read_csv(input_path, header=0, sep=";")

    # Split input data
    train, test = train_test_split(responses, stratify=responses["opinion"])

    # Save to DocBins
    db = DocBin()
    for _index, line in train.iterrows():
        doc = nlp.make_doc(line["checked_text"])
        doc.cats = {line["opinion"]: 1.0}
        # print(doc[1:20], doc.cats)
        db.add(doc)
    print(f"Ecriture dans {train_path} de {len(train)} commentaires")
    db.to_disk(train_path)

    db = DocBin()
    for _index, line in test.iterrows():
        doc = nlp.make_doc(line["checked_text"])
        doc.cats = {line["opinion"]: 1.0}
        # print(doc[1:20], doc.cats)
        db.add(doc)
    print(f"Ecriture dans {test_path} de {len(test)} commentaires")
    db.to_disk(test_path)

    return None


if __name__ == "__main__":
    typer.run(convert)
