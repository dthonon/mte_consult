"""Command-line interface."""
import csv
import logging
from functools import partial

# import unicodedata
from pathlib import Path
from typing import Any

import click
import hunspell  # type: ignore
import pandas as pd
import spacy
from textacy import preprocessing


# Spell checking word counter (global)
nb_words = 0

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


@click.version_option()
@click.group()
@click.option(
    "--data_directory",
    default=".",
    help="Répertoire contenant les répertoires de données raw, preprocessed ...",
)
@click.option(
    "--start_comment", default=1, help="Numéro de premier commentaire à analyser"
)
@click.option(
    "--end_comment",
    default=0,
    help="Numéro de dernier commentaire à analyser, 0 : pas de limite",
)
@click.argument("consultation")
@click.pass_context
def main(
    ctx: click.Context,
    consultation: str,
    data_directory: str,
    start_comment: int,
    end_comment: int,
) -> None:
    """Mte_Consult."""
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)

    ctx.obj["CONSULTATION"] = consultation
    ctx.obj["DATA_DIRECTORY"] = data_directory
    ctx.obj["START_COMMENT"] = start_comment if start_comment > 1 else None
    ctx.obj["END_COMMENT"] = end_comment if end_comment > 0 else None


@main.command()
@click.pass_context
def preprocess(ctx: click.Context) -> None:
    """Prétraitement du fichier brut contenant les commentaires."""
    consultation = ctx.obj["CONSULTATION"]
    data_dir = ctx.obj["DATA_DIRECTORY"]
    logging.info(f"Prétraitement de {consultation} dans {data_dir}")
    csv_file = Path(data_dir + "/raw/" + consultation + ".csv")
    logging.debug("Lecture %s", csv_file)
    responses = pd.read_csv(csv_file, header=0, quoting=csv.QUOTE_ALL, nrows=1000000)
    logging.info("Nombre de commentaires bruts : %d", len(responses))

    # Découpe du sujet en éléments
    responses[["titre", "date", "heure"]] = responses.sujet.str.extract(
        "(.*), le (.*) à (.*)", expand=True
    )
    responses = responses[["titre", "date", "heure", "texte", "sujet"]].sort_values(
        by="date", ignore_index=True
    )

    # Suppression des ligne dupliquées
    responses = responses.drop_duplicates(subset=["titre", "texte"])
    logging.info("Commentaires restants après déduplication : %d", len(responses))

    # Fusion en une colonne pour traitement du texte
    responses["raw_text"] = responses["titre"] + ". " + responses["texte"]
    responses["raw_text"].fillna(value="?", inplace=True)
    responses = responses.drop(columns=["texte"])

    # Nettoyage du texte brut
    responses.raw_text = responses.raw_text.str.replace("[_%=/°]", " ", regex=True)
    responses.raw_text = responses.raw_text.str.replace("+", " plus ")
    responses.raw_text = responses.raw_text.str.replace("*", " fois ")
    responses.raw_text = responses.raw_text.str.replace("qq", "quelque")
    responses.raw_text = responses.raw_text.str.replace(r"\d\dh\d\d", "", regex=True)
    preproc = preprocessing.pipeline.make_pipeline(
        preprocessing.normalize.bullet_points,
        preprocessing.normalize.hyphenated_words,
        preprocessing.replace.urls,
        partial(preprocessing.replace.numbers, repl=" NOMBRE "),
        partial(preprocessing.replace.emojis, repl=" "),
        partial(preprocessing.replace.emails, repl=" "),
        partial(preprocessing.replace.currency_symbols, repl=" Euros "),
        preprocessing.remove.html_tags,
        preprocessing.normalize.whitespace,
    )
    responses.raw_text = responses["raw_text"].apply(preproc)

    # Ecriture du fichier résultant
    csv_file = Path(data_dir + "/preprocessed/" + consultation + ".csv")
    logging.debug("Ecriture dans %s", csv_file)
    responses.to_csv(csv_file, header=True, quoting=csv.QUOTE_ALL)


def _spell_correction(doc: spacy.tokenizer, spell: Any) -> str:
    """Spell correction of misspelled words."""
    global nb_words
    nb_words += 1
    if (nb_words % 100) == 0:
        logging.info(f"Spell checking word number {nb_words}")
    text = ""
    for d in doc:
        word = d.text
        # Spell check meaningfull words only
        if d.is_space:
            pass  # Nothing to check
        elif d.is_stop or d.is_punct or spell.spell(word):
            text += d.text_with_ws
        else:
            sp = spell.suggest(word)
            if len(sp) > 0:
                rep = sp[0]
                print(word + " => " + rep)
                text += rep + d.whitespace_
            else:
                logging.warning(f"Unable to correct {word}")
                text += d.text_with_ws
    return text


def _fr_nlp() -> spacy.language.Language:
    # Prepare NLP processing
    logging.info("Préparation du traitement NLP")
    # spacy.prefer_gpu()
    _nlp = spacy.load("fr_core_news_sm", disable=("tagger", "parser", "ner"))
    logging.info(f"NLP pipeline: {_nlp.pipe_names}")
    # Adjust stopwords for this specific topic
    _nlp.Defaults.stop_words |= {"y", "france", "esod"}
    _nlp.Defaults.stop_words -= {"pour"}
    return _nlp


@main.command()
@click.pass_context
def check(ctx: click.Context) -> None:
    """Correction orthographique des commentaires."""
    consultation = ctx.obj["CONSULTATION"]
    data_dir = ctx.obj["DATA_DIRECTORY"]
    start_comment = ctx.obj["START_COMMENT"]
    end_comment = ctx.obj["END_COMMENT"]
    logging.info(f"Correction orthographique des commentaires de {consultation}")
    csv_file = Path(data_dir + "/preprocessed/" + consultation + ".csv")
    logging.debug(f"Lecture {csv_file} depuis {start_comment} jusqu'à {end_comment}")
    responses = pd.read_csv(
        csv_file, header=0, quoting=csv.QUOTE_ALL, nrows=end_comment
    )
    logging.info("Nombre de commentaires à traiter : %d", len(responses))

    # Correction orthographique
    spell = hunspell.HunSpell(
        "/usr/share/hunspell/fr_FR.dic", "/usr/share/hunspell/fr_FR.aff"
    )
    added_words = Path(data_dir + "/external/" + "mtes.txt")
    spell.add_dic(added_words)
    # spell.remove("abatage")
    tokenizer = _fr_nlp().tokenizer
    responses["checked_text"] = responses["raw_text"].apply(
        lambda d: _spell_correction(tokenizer(d), spell)
    )

    # Read classification data, if it exists
    csv_file = Path(data_dir + "/external/" + consultation + "_cat.csv")
    if csv_file.exists():
        logging.info(f"Chargement des classifications {csv_file}")
        classif = pd.read_csv(csv_file, header=0, quoting=csv.QUOTE_ALL)
        logging.info(f"Lu {len(classif)} données de classification")
        classif = classif.drop(columns=["checked_text"])
        responses = responses.merge(classif, on="sujet", how="left")
    else:
        logging.info(f"Pas de classification trouvée dans {csv_file}")
        classif = None

    # Save processed data to csv file
    csv_file = Path(data_dir + "/interim/" + consultation + ".csv")
    logging.info(f"Storing {len(responses)} rows of processed data to {csv_file}")
    responses.to_csv(csv_file, index=False, quoting=csv.QUOTE_ALL)


if __name__ == "__main__":
    main(obj={})  # pragma: no cover
