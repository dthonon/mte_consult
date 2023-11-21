"""Command-line interface."""
import csv
import logging
import re
from functools import partial
import time

# import unicodedata
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
import click
import hunspell  # type: ignore
import pandas as pd
import requests
import spacy
import textacy
from spacy.tokenizer import Tokenizer
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
    "--domain",
    default=".",
    help="Base de l'URL du site",
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
    domain: str,
    consultation: str,
    start_comment: int,
    end_comment: int,
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


@main.command()
@click.pass_context
def retrieve(ctx: click.Context) -> None:
    """Récupération des commentaires de la consultation."""
    domain = ctx.obj["DOMAIN"]
    consultation = ctx.obj["CONSULTATION"]
    data_dir = ctx.obj["DATA_DIRECTORY"]
    start_comment = ctx.obj["START_COMMENT"]
    end_comment = ctx.obj["END_COMMENT"]

    # Récupération des pages de commentaire
    url = "https://www." + domain + "/" + consultation + ".html"
    nb_com_re = re.compile(r"(Consultation.* )(\d+) contributions")
    max_com = 0
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    with requests.Session() as s:
        with open(Path(data_dir + "/raw/" + consultation + ".csv"), "a") as csvfile:
            # Note : ecriture en mode append pour essayer de trouver un maximum de contributions
            # vu que chaque boucle n'en récupère qu'une partie (pb de cache serveur ?)
            logging.info(f"Ecriture des commentaires dans {csvfile.name}")
            comwriter = csv.writer(csvfile, delimiter=",")
            comwriter.writerow(["sujet", "texte"])
            for npage in range(start_comment + 1, end_comment, 20):
                if npage == start_comment + 1:
                    payload = {"lang": "fr"}
                else:
                    payload = {"lang": "fr", "debut_forums": str(npage)}
                logging.info(f"Téléchargement depuis {url}, params : {payload}")
                page = s.get(url, params=payload, timeout=10, headers=headers)
                if page.status_code != requests.codes.ok:
                    break
                contenu = BeautifulSoup(page.content, "html.parser")
                nb_com = re.match(
                    nb_com_re,
                    contenu.select_one("div.dateart").text.strip().replace("\n", ""),
                )
                if npage == start_comment + 1:
                    max_com = int(nb_com.group(2))
                commentaires = contenu.select("div.ligne-com")
                logging.info(f"Commentaires dans la page : {len(commentaires)}")
                for com in commentaires:
                    c = [
                        com.select_one("div.titresujet").text.strip(),
                        com.select_one("div.textesujet")
                        .text.strip()
                        .replace("\n", " "),
                    ]
                    comwriter.writerow(c)
                if npage > max_com:
                    break
                time.sleep(5)


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
    start_comment = ctx.obj["START_COMMENT"]
    end_comment = ctx.obj["END_COMMENT"]

    logging.info(f"Prétraitement de {consultation} dans {data_dir}")
    csv_file = Path(data_dir + "/raw/" + consultation + ".csv")
    logging.debug("Lecture %s", csv_file)
    responses = pd.read_csv(csv_file, header=0, quoting=csv.QUOTE_ALL, nrows=1000000)
    logging.info(f"Nombre de commentaires bruts : {len(responses)}")

    # Découpe du sujet en éléments
    responses[["titre", "date", "heure"]] = responses.sujet.str.extract(
        "(.*), le (.*) à (.*)", expand=True
    )
    responses = responses[["titre", "date", "heure", "texte", "sujet"]].sort_values(
        by="date", ignore_index=True
    )

    # Suppression des ligne dupliquées
    responses = responses.drop_duplicates(subset=["titre", "texte"])
    logging.info(f"Commentaires restants après déduplication : {len(responses)}")

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
    responses.to_csv(csv_file, header=True, quoting=csv.QUOTE_ALL)


@main.command()
@click.pass_context
def prepare(ctx: click.Context) -> None:
    """Correction orthographique des commentaires."""
    consultation = ctx.obj["CONSULTATION"]
    data_dir = ctx.obj["DATA_DIRECTORY"]

    csv_file = Path(data_dir + "/preprocessed/" + consultation + ".csv")
    logging.debug(f"Lecture de {csv_file}")
    responses = pd.read_csv(csv_file, header=0, quoting=csv.QUOTE_ALL, nrows=1000000)
    logging.info(f"Nombre de commentaires prétraités : {len(responses)}")

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

    # Sauvegarde des données préparées dans un fichier csv
    csv_file = Path(data_dir + "/interim/" + consultation + ".csv")
    logging.info(f"Storing {len(responses)} rows of processed data to {csv_file}")
    responses.to_csv(csv_file, index=False, quoting=csv.QUOTE_ALL)

    # Prepare final corpus from spell-checked text, for analysis
    # corpus = textacy.Corpus(_fr_nlp())
    # for row in responses.itertuples():
    #     if textacy.lang_utils.identify_lang(row.raw_text) == "fr":
    #         corpus.add_record(
    #             (
    #                 self._spell_correction(
    #                     spell,
    #                     textacy.make_spacy_doc(row.raw_text, self._fr_nlp),
    #                     logger,
    #                 ).lower(),
    #                 {
    #                     "name": row.nom,
    #                     "date": row.date,
    #                     "time": row.heure,
    #                     "opinion": row.opinion,
    #                     "uid": row.uid,
    #                 },
    #             )
    #         )
    # logger.info(_("Response spell checked corpus %s"), corpus)
    # for d in range(50):
    #     print(corpus[d]._.preview)
    #     print("meta:", corpus[d]._.meta)
    # # Save data
    # corpus_file = Path.home() / (
    #     "ana_consult/data/interim/" + config.consultation_name + "_doc.pkl"
    # )
    # logger.info(_("Storing NLP document to %s"), corpus_file)
    # corpus.save(corpus_file)


@main.command()
@click.pass_context
def classify(ctx: click.Context) -> None:
    """Classification des commentaires."""
    logging.info("Classification des commentaires")
    consultation = ctx.obj["CONSULTATION"]
    data_dir = ctx.obj["DATA_DIRECTORY"]


if __name__ == "__main__":
    main(obj={})  # pragma: no cover
