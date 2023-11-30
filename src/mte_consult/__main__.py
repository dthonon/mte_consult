"""Command-line interface."""
import csv
import logging
import random
import re
import time
from functools import partial

# import unicodedata
from pathlib import Path
from typing import Any
from typing import List
from typing import Set
from typing import Tuple

import click
import hunspell  # type: ignore
import pandas as pd
import requests
import spacy
from bs4 import BeautifulSoup
from spacy.tokenizer import Tokenizer
from spacy.tokens import DocBin
from textacy import preprocessing


# Spell checking word counter (global)
nb_words = 0

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@click.version_option()
@click.group()
@click.option(
    "--data_directory",
    default=".",
    help="Répertoire contenant les répertoires de données raw, preprocessed ...",
)
@click.option(
    "--domain",
    default=".",
    help="Base de l'URL du site",
)
@click.option(
    "--start_comment", default=1, help="Numéro de premier commentaire à télécharger"
)
@click.option(
    "--end_comment",
    default=0,
    help="Numéro de dernier commentaire à télécharger, 0 : pas de limite",
)
@click.option(
    "--nb_pages",
    default=0,
    help="Nombre de pages à télécharger, 0 : pas de limite",
)
@click.argument("consultation")
@click.pass_context
def main(
    ctx: click.Context,
    domain: str,
    consultation: str,
    start_comment: int,
    end_comment: int,
    nb_pages: int,
    data_directory: str,
) -> None:
    """Mte_Consult."""
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)

    ctx.obj["DOMAIN"] = domain
    ctx.obj["CONSULTATION"] = consultation
    ctx.obj["DATA_DIRECTORY"] = data_directory
    ctx.obj["START_COMMENT"] = start_comment if start_comment > 1 else 0
    ctx.obj["END_COMMENT"] = end_comment if end_comment > 0 else 100000
    ctx.obj["NB_PAGES"] = nb_pages if nb_pages > 0 else 100000


@main.command()
@click.pass_context
def retrieve(ctx: click.Context) -> None:
    """Récupération des commentaires de la consultation."""
    domain = ctx.obj["DOMAIN"]
    consultation = ctx.obj["CONSULTATION"]
    data_dir = ctx.obj["DATA_DIRECTORY"]
    start_comment = ctx.obj["START_COMMENT"]
    end_comment = ctx.obj["END_COMMENT"]
    nb_pages = ctx.obj["NB_PAGES"]

    logging.info(f"Téléchargement de {consultation}")
    csv_file = Path(data_dir + "/raw/" + consultation + ".csv")
    if csv_file.is_file():
        logging.debug(f"Lecture des téléchargements depuis {csv_file}")
        responses = pd.read_csv(csv_file, header=0, sep=";")
    else:
        logging.debug(f"Pas de téléchargement précédents")
        responses = pd.DataFrame()
    logging.info(f"Nombre de commentaires lus : {len(responses)}")

    # Récupération des pages de commentaire
    url = "https://www." + domain + "/" + consultation + ".html"
    nb_com_re = re.compile(r"(Consultation.* )(\d+) contributions")
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    with requests.Session() as s:
        starting = start_comment
        pages = [i for i in range(starting, end_comment, 20)]
        random.shuffle(pages)
        for npage in pages[:nb_pages]:
            payload = {"lang": "fr", "debut_forums": str(npage)}
            logging.info(f"Téléchargement depuis {url}, params : {payload}")
            try:
                page = s.get(url, params=payload, timeout=10, headers=headers)
                if page.status_code != requests.codes.ok:
                    break
                contenu = BeautifulSoup(page.content, "html.parser")
                nb_com = re.match(
                    nb_com_re,
                    contenu.select_one("div.dateart").text.strip().replace("\n", ""),
                )
                if npage == starting:
                    max_com = int(nb_com.group(2))
                commentaires = contenu.select("div.ligne-com")
                logging.info(f"Commentaires dans la page : {len(commentaires)}")
                pre_drop = len(responses)
                for com in commentaires:
                    c = pd.DataFrame(
                        {
                            "sujet": com.select_one("div.titresujet").text.strip(),
                            "texte": com.select_one("div.textesujet")
                            .text.strip()
                            .replace("\n", " "),
                        },
                        index=[0],
                    )
                    responses = pd.concat([c, responses.loc[:]]).reset_index(drop=True)
                # Suppression des ligne dupliquées et sauvegarde
                responses = responses.drop_duplicates(subset=["sujet"])
                logging.info(
                    f"Nb de nouveaux commentaires : {len(responses) - pre_drop}, total : {len(responses)}"
                )
                logging.debug(f"Ecriture dans {csv_file}")
                responses.to_csv(csv_file, header=True, sep=";", index=False)
                time.sleep(2 * 60)
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
                requests.exceptions.Timeout,
                TimeoutError,
            ):
                logging.warning("Page en timeout")
                time.sleep(10)


def _spell_correction(doc: Tokenizer, spell: Any) -> str:
    """Spell correction of misspelled words."""
    global nb_words
    nb_words += 1
    if (nb_words % 100) == 0:
        logging.info(f"Vérification orthographique de {nb_words} mots")
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
def preprocess(ctx: click.Context) -> None:
    """Prétraitement du fichier brut contenant les commentaires."""
    consultation = ctx.obj["CONSULTATION"]
    data_dir = ctx.obj["DATA_DIRECTORY"]

    logging.info(f"Prétraitement de {consultation} dans {data_dir}")
    csv_file = Path(data_dir + "/raw/" + consultation + ".csv")
    logging.debug("Lecture %s", csv_file)
    responses = pd.read_csv(csv_file, header=0, sep=";")
    logging.info(f"Nombre de commentaires bruts : {len(responses)}")

    # Découpe du sujet en éléments
    responses[["titre", "date", "heure"]] = responses.sujet.str.extract(
        "(.*), le (.*) à (.*)", expand=True
    )
    responses = responses[["titre", "date", "heure", "texte", "sujet"]].sort_values(
        by="date", ignore_index=True
    )

    # Suppression des ligne dupliquées
    responses = responses.drop_duplicates(subset=["texte"])
    logging.info(
        f"Commentaires restants après déduplication du texte: {len(responses)}"
    )

    # Fusion en une colonne pour traitement du texte
    responses["raw_text"] = responses["titre"] + ". " + responses["texte"]
    responses["raw_text"].fillna(value="?", inplace=True)
    responses = responses.drop(columns=["texte"])

    # Nettoyage du texte brut
    logging.info("Nettoyage du texte brut")
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

    # Correction orthographique des commentaires
    logging.info(f"Correction orthographique des commentaires de {consultation}")
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

    # Ecriture du fichier résultant
    csv_file = Path(data_dir + "/preprocessed/" + consultation + ".csv")
    logging.debug(f"Ecriture dans {csv_file}")
    responses.to_csv(csv_file, header=True, sep=";", index=False)


# Create a function to create a spacy dataset
def _make_docs(data: List[Tuple[str, str]], target_file: str, cats: Set[str]):
    docs = DocBin()
    # Use nlp.pipe to efficiently process a large number of text inputs,
    # the as_tuple arguments enables giving a list of tuples as input and
    # reuse it in the loop, here for the labels
    for doc, label in nlp.pipe(data, as_tuples=True):
        # Encode the labels (assign 1 the subreddit)
        for cat in cats:
            doc.cats[cat] = 1 if cat == label else 0
        docs.add(doc)
    docs.to_disk(target_file)
    return docs


@main.command()
@click.pass_context
def prepare(ctx: click.Context) -> None:
    """Correction orthographique des commentaires."""
    consultation = ctx.obj["CONSULTATION"]
    data_dir = ctx.obj["DATA_DIRECTORY"]

    csv_file = Path(data_dir + "/preprocessed/" + consultation + ".csv")
    logging.debug(f"Lecture de {csv_file}")
    responses = pd.read_csv(csv_file, header=0, sep=";")
    logging.info(f"Nombre de commentaires prétraités : {len(responses)}")

    # Read classification data, if it exists
    csv_file = Path(data_dir + "/external/" + consultation + "_cat.csv")
    if csv_file.exists():
        logging.info(f"Chargement des classifications {csv_file}")
        classif = pd.read_csv(csv_file, header=0, sep=";")
        logging.info(f"Lu {len(classif)} données de classification")
        classif.drop(columns=["checked_text"], inplace=True)
        responses = pd.merge(
            responses, classif, on="sujet", how="left", suffixes=(None, "_c")
        )
    else:
        logging.info(f"Pas de classification trouvée dans {csv_file}")
        classif = None

    # Sauvegarde des données préparées dans un fichier csv
    csv_file = Path(data_dir + "/interim/" + consultation + ".csv")
    logging.info(f"Storing {len(responses)} rows of processed data to {csv_file}")
    responses.to_csv(csv_file, index=False, header=True, sep=";")

    cats = responses.opinion.unique().tolist()
    print(cats)


@main.command()
@click.pass_context
def classify(ctx: click.Context) -> None:
    """Classification des commentaires."""
    logging.info("Classification des commentaires")
    consultation = ctx.obj["CONSULTATION"]
    data_dir = ctx.obj["DATA_DIRECTORY"]


if __name__ == "__main__":
    main(obj={})  # pragma: no cover
