"""Command-line interface."""

import locale
import logging
import random
import re
import time
from collections import Counter
from datetime import datetime
from datetime import timedelta
from functools import partial


# import unicodedata
from pathlib import Path
from typing import Any

import click
import hunspell  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import spacy
from bs4 import BeautifulSoup
from imblearn.under_sampling import RandomUnderSampler
from lingua import Language
from lingua import LanguageDetectorBuilder
from sklearn import metrics  # type: ignore
from sklearn.cluster import DBSCAN  # type: ignore
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.naive_bayes import ComplementNB  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.svm import LinearSVC  # type: ignore
from spacy.tokenizer import Tokenizer  # type: ignore
from textacy import preprocessing
from unidecode import unidecode


# Constantes
NB_COMMENTS = 20  # Nombre de commentaires par page
NB_COMMENTS_MAX = 1000000  # Nombre maximum de commentaires à télécharger

# Spell checking word counter (global)
nb_comm = 0

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@click.version_option()
@click.group()
@click.option(
    "--data_directory",
    default="data",
    help="Répertoire contenant les répertoires de données raw, preprocessed ...",
)
@click.option(
    "--domain",
    default="consultations-publiques.developpement-durable.gouv.fr",
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
        logging.info(f"Lecture des téléchargements depuis {csv_file}")
        responses = pd.read_csv(csv_file, header=0, sep=";", encoding="utf-8-sig")
    else:
        logging.info("Pas de téléchargement précédents")
        responses = pd.DataFrame()
    logging.info(f"Nombre de commentaires déjà téléchargés : {len(responses)}")

    # Récupération des pages de commentaire
    url = "https://www." + domain + "/" + consultation + ".html"
    nb_com_re = re.compile(r"(Consultation.* )(\d+) contributions")
    forum_re = re.compile(r".*>(\d+)</a>")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/119.0.0.0 Safari/537.36"
        ),
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    # Création de la liste des pages à télécharger
    pages = [i for i in range(start_comment, end_comment, NB_COMMENTS)]
    pages = pages[: nb_pages + 1]
    max_com = NB_COMMENTS_MAX

    while len(pages) > 0:
        # La liste est mélangée pour éviter de boucler sur une page en erreur transitoire
        random.shuffle(pages)
        npage = pages[0]
        if npage < max_com:
            payload = {"lang": "fr", "debut_forums": str(npage)}
            logging.info(f"Téléchargement depuis {url}, params : {payload}")
            try:
                page = requests.get(url, params=payload, timeout=10, headers=headers)
                if page.status_code != requests.codes.ok:
                    logging.warning(f"Page en erreur HTTP : {page.status_code}")
                    time.sleep(10)
                    continue
                contenu = BeautifulSoup(page.content, "html.parser")

                dateart_tag = contenu.select_one("div.dateart")
                dateart_text = (
                    dateart_tag.text.strip().splitlines()
                    if dateart_tag is not None
                    else []
                )
                nb_com = re.match(
                    nb_com_re,
                    " ".join(dateart_text),
                )
                if (
                    nb_com is not None
                    and int(nb_com.group(2)) > 0
                    and max_com != int(nb_com.group(2))
                ):
                    max_com = int(nb_com.group(2))
                    logging.info(f"Nombre total de commentaires {max_com}")

                forum = re.match(
                    forum_re,
                    str(
                        contenu.find(
                            "a",
                            class_="fr-pagination__link",
                            attrs={"aria-current": "page"},
                        )
                    ),
                )

                if forum is not None:
                    rec = (int(forum.group(1)) - 1) * NB_COMMENTS
                    # La page reçue est retirée de la liste
                    if rec in pages:
                        pages.remove(rec)
                    logging.info(
                        f"Page demandée : {npage}, page reçue : {rec}, pages restantes {len(pages)}"
                    )
                else:
                    logging.warning(
                        "Impossible de trouver le numéro de page dans le contenu HTML."
                    )
                    # Optionally, you can skip or handle this case differently
                    pages.remove(npage)

                commentaires = contenu.select("div.ligne-com")
                pre_drop = len(responses)
                for com in commentaires:
                    titre_tag = com.select_one("div.titresujet")
                    texte_tag = com.select_one("div.textesujet")
                    titre = (
                        " ".join(titre_tag.text.strip().splitlines())
                        if titre_tag is not None
                        else ""
                    )
                    texte = (
                        " ".join(texte_tag.text.strip().splitlines())
                        if texte_tag is not None
                        else ""
                    )
                    c = pd.DataFrame(
                        {
                            "titre": titre,
                            "texte": texte,
                        },
                        index=[0],
                    )
                    responses = pd.concat([c, responses.loc[:]]).reset_index(drop=True)
                # Suppression des ligne dupliquées et sauvegarde
                responses = responses.drop_duplicates()
                logging.info(
                    f"Nb de nouveaux commentaires : {len(responses) - pre_drop}/{len(commentaires)}, total : {len(responses)}/{max_com}"
                )
                responses.to_csv(
                    csv_file, header=True, sep=";", index=False, encoding="utf-8-sig"
                )
                time.sleep(random.uniform(1, 3))  # Pause entre les requêtes
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
                TimeoutError,
            ) as exc:
                logging.warning(f"Page en erreur : {type(exc)}")
                time.sleep(10)


def _spell_correction(doc: Tokenizer, spell: Any, corrected: pd.DataFrame) -> str:
    """Spell correction of misspelled words, hunspell version."""
    global nb_comm
    nb_comm += 1
    if (nb_comm % 100) == 0:
        logging.info(f"Vérification orthographique de {nb_comm} commentaires")
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
                corrected.loc[len(corrected)] = {
                    "raw_text": word,
                    "checked_text": rep,
                }
                # print(word + " => " + rep)
                text += rep + d.whitespace_
            else:
                logging.warning(f"Unable to correct {word}")
                text += d.text_with_ws
    return text


def _fr_nlp() -> spacy.language.Language:
    # Prepare NLP processing
    logging.info("Préparation du traitement NLP")
    # spacy.prefer_gpu()
    _nlp = spacy.load(
        "fr_core_news_sm",
        disable=(
            "ner",
            "textcat",
            "attribute_ruler",
            "tok2vec",
        ),
    )
    logging.info(f"NLP pipeline: {_nlp.pipe_names}")
    # Adjust stopwords for this specific topic
    _nlp.Defaults.stop_words |= {"y", "france"}
    _nlp.Defaults.stop_words -= {"pour"}
    return _nlp


def _lemmatize(doc: Tokenizer) -> str:
    # Take the `token.lemma_` of each non-stop word
    return " ".join(
        [token.lemma_ for token in doc if not (token.is_stop or token.is_punct)]
    )


@main.command()
@click.pass_context
def preprocess(ctx: click.Context) -> None:
    """Prétraitement du fichier brut contenant les commentaires."""
    consultation = ctx.obj["CONSULTATION"]
    data_dir = ctx.obj["DATA_DIRECTORY"]

    languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN]
    detector = (
        LanguageDetectorBuilder.from_languages(*languages)
        # .with_minimum_relative_distance(0.99)
        .build()
    )

    logging.info(f"Prétraitement de {consultation} dans {data_dir}")
    csv_file = Path(data_dir + "/raw/" + consultation + ".csv")
    responses = pd.read_csv(
        csv_file,
        header=0,
        sep=";",
        names=["sujet", "texte"],
        encoding="utf-8-sig",
        # nrows=200,
    )
    nb_comments = len(responses)
    logging.info(f"Nombre de commentaires bruts : {len(responses)}")

    # Supression des commentaires déjà prétraités, le cas échéant
    preproc_file = Path(data_dir + "/preprocessed/" + consultation + ".csv")
    preproc_responses = None
    if preproc_file.is_file():
        logging.info(f"Lecture des commentaires prétraités depuis {preproc_file}")
        preproc_responses = pd.read_csv(
            preproc_file,
            header=0,
            sep=";",
            encoding="utf-8-sig",
            parse_dates=["dateheure"],
        )
        responses = responses[~responses.sujet.isin(preproc_responses.sujet)]
        logging.info(
            f"Commentaires à prétraiter : {len(responses)} (déjà prétraités : {nb_comments - len(responses)})"
        )

    # Découpe du sujet en éléments
    responses[["titre", "dateheure"]] = responses.sujet.str.extract(
        "(.*), le (.* à .*)", expand=True
    )
    # Suppression des ligne dupliquées
    responses = responses.drop_duplicates(subset=["titre", "texte"])
    logging.info(
        f"Commentaires restants après déduplication du titre et du texte: {len(responses)}"
    )

    # Fusion en une colonne pour traitement du texte
    responses["raw_text"] = responses["titre"] + " | " + responses["texte"]
    responses["raw_text"] = responses["raw_text"].fillna(value="?")

    # Suppression du texte étranger
    lang = responses.texte.apply(lambda d: detector.detect_language_of(str(d)))
    for t in responses.texte[lang != Language.FRENCH]:
        logging.info(f"Langue {detector.detect_language_of(t)} : {t}")
        confidence_values = detector.compute_language_confidence_values(t)
        for confidence in confidence_values:
            logging.info(f"{confidence.language.name}: {confidence.value:.2f}")
    responses.drop(responses[lang != Language.FRENCH].index, inplace=True)
    logging.info(f"Commentaires français restant : {len(responses)}")

    locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")
    responses["dateheure"] = pd.to_datetime(
        responses["dateheure"].apply(lambda d: d.replace("1er", "1")),
        format="%d %B %Y à %Hh%M",
        errors="coerce",
    )

    # Nettoyage du texte brut
    logging.info("Nettoyage du texte brut")
    responses["raw_text"] = responses["raw_text"].str.lower()
    responses["raw_text"] = responses["raw_text"].apply(
        lambda d: d.encode("latin-1", "ignore").decode("latin-1")
    )

    preproc = preprocessing.pipeline.make_pipeline(
        preprocessing.normalize.bullet_points,
        preprocessing.normalize.hyphenated_words,
        preprocessing.replace.urls,
        partial(preprocessing.replace.emails, repl=" courriel "),
        partial(preprocessing.replace.numbers, repl=" nombre "),
        partial(preprocessing.replace.emojis, repl=" "),
        partial(preprocessing.replace.emails, repl=" "),
        partial(preprocessing.replace.currency_symbols, repl=" euros "),
        preprocessing.remove.html_tags,
        preprocessing.normalize.whitespace,
    )
    responses.raw_text = responses["raw_text"].apply(preproc)
    responses.raw_text = responses["raw_text"].str.replace(r"\r", " ", regex=True)
    responses.raw_text = responses["raw_text"].str.replace("+", " plus ")
    responses.raw_text = responses["raw_text"].str.replace("*", " fois ")
    responses.raw_text = responses["raw_text"].str.replace("grd", "grand")
    responses.raw_text = responses["raw_text"].str.replace("qq", "quelque")
    responses.raw_text = responses["raw_text"].str.replace("qlqs", "quelques")
    responses.raw_text = responses["raw_text"].str.replace(".euses", "")
    responses.raw_text = responses["raw_text"].str.replace(".", ". ")
    responses.raw_text = responses["raw_text"].str.replace("(", " (")
    responses.raw_text = responses["raw_text"].str.replace(")", ") ")
    responses.raw_text = responses["raw_text"].str.replace(r"[_%=/°]", " ", regex=True)
    responses.raw_text = responses["raw_text"].str.replace(
        r"\d?\dh\d\d", "HEURE", regex=True
    )

    # Correction orthographique des commentaires
    logging.info(f"Correction orthographique des commentaires de {consultation}")
    nlp = _fr_nlp()
    tokenizer = nlp.tokenizer
    corrected = pd.DataFrame(columns=["raw_text", "checked_text"])
    added_words = Path(data_dir + "/external/" + "mtes.txt")
    # Correction orthographique avec hunspell
    spell = hunspell.HunSpell(
        "/usr/share/hunspell/fr_FR.dic", "/usr/share/hunspell/fr_FR.aff"
    )
    if added_words.is_file():
        logging.info(f"Ajout des mots du fichier {added_words}")
        spell.add_dic(added_words)
    responses["checked_text"] = responses["raw_text"].apply(
        lambda d: _spell_correction(tokenizer(str(d)), spell, corrected)
    )

    logging.info(f"Nombre de mots corrigés : {len(corrected)}")
    corrected = corrected.drop_duplicates(
        subset=["raw_text", "checked_text"]
    ).sort_values(by="raw_text")

    # Ecriture du fichier des corrections orthographiques
    csv_file = Path(data_dir + "/preprocessed/" + consultation + "_corrected.csv")
    corrected.to_csv(
        csv_file, header=True, sep=";", index=False, encoding="utf-8-sig", mode="a"
    )

    # Lemmatisation des commentaires
    logging.info("Lemmatisation des commentaires")
    responses["lemma"] = responses["checked_text"].apply(
        lambda d: _lemmatize(nlp(str(d)))
    )

    # Concatenation des commentaires prétraités
    if preproc_responses is not None and len(preproc_responses) > 0:
        logging.info(
            f"Concaténation des commentaires prétraités : préexistants {len(responses)} et nouveaux {len(preproc_responses)}"
        )
        responses = pd.concat([preproc_responses, responses], ignore_index=True)

    # Ecriture du fichier résultant
    csv_file = Path(data_dir + "/preprocessed/" + consultation + ".csv")
    logging.info(
        f"Ecriture dans {csv_file} de {len(responses)} commentaires prétraités"
    )
    responses.sort_values(by="dateheure", ignore_index=True).to_csv(
        csv_file, header=True, sep=";", index=False, encoding="utf-8-sig"
    )

    # Ajout de la colonne opinion
    if "opinion" not in responses.columns:
        logging.info("Ajout de la colonne opinion")
        responses["opinion"] = ""

    # Découpage selon l'avis exprimé
    # Séparation des commentaires favorables
    avis_favorable = responses.checked_text.apply(
        lambda d: "avis favorable" in str(d).lower()[: len("avis favorable")]
        or "avis positif" in str(d).lower()[: len("avis positif")]
        or "favorable" in str(d).lower()[: len("favorable")]
        or "tres favorable" in unidecode(str(d).lower()[: len("tres favorable")])
        or "avis tres favorable"
        in unidecode(str(d).lower()[: len("avis tres favorable")])
        or "avis nombre favorable"
        in unidecode(str(d).lower()[: len("avis nombre favorable")])
        or "avis fortement favorable"
        in unidecode(str(d).lower()[: len("avis fortement favorable")])
        or "avis tres favorable"
        in unidecode(str(d).lower()[: len("avis tres favorable")])
    )
    logging.info(
        f"Nombre de commentaires avec avis favorable : {len(responses[avis_favorable])}"
    )
    responses.loc[avis_favorable, "opinion"] = "Favorable"
    # Sauve les commentaires favorables dans un fichier
    csv_file = Path(data_dir + "/preprocessed/" + consultation + "_avis_favorable.csv")
    logging.info(f"Sauvegarde des commentaires favorables dans {csv_file}")
    responses[avis_favorable].to_csv(
        csv_file, header=True, sep=";", index=False, encoding="utf-8-sig"
    )
    # Suppression des commentaires favorables
    responses = responses[~avis_favorable].reset_index(drop=True)
    logging.info(
        f"Nombre de commentaires restants : {len(responses)} (avis défavorable ou inconnus)"
    )

    # Séparation des commentaires défavorables
    avis_defavorable = responses.checked_text.apply(
        lambda d: "non favorable" in str(d).lower()[: len("non favorable")]
        or "avis négatif" in str(d).lower()[: len("avis négatif")]
        or "defavorable" in unidecode(str(d).lower()[: len("defavorable")])
        or "nombre defavorable"
        in unidecode(str(d).lower()[: len("nombre defavorable")])
        or "tres defavorable" in unidecode(str(d).lower()[: len("tres defavorable")])
        or "totalement defavorable"
        in unidecode(str(d).lower()[: len("totalement defavorable")])
        or "extremement defavorable"
        in unidecode(str(d).lower()[: len("extremement defavorable")])
        or "completement defavorable"
        in unidecode(str(d).lower()[: len("completement defavorable")])
        or "contre" in str(d).lower()[: len("contre")]
        or "avis defavorable" in unidecode(str(d).lower()[: len("avis defavorable")])
        or "avis tres defavorable"
        in unidecode(str(d).lower()[: len("avis tres defavorable")])
        or "avis tres tres defavorable"
        in unidecode(str(d).lower()[: len("avis tres tres defavorable")])
        or "avis nombre defavorable"
        in unidecode(str(d).lower()[: len("avis nombre defavorable")])
        or "avis completement defavorable"
        in unidecode(str(d).lower()[: len("avis completement defavorable")])
        or "avis absolument defavorable"
        in unidecode(str(d).lower()[: len("avis absolument defavorable")])
        or "avis extremement defavorable"
        in unidecode(str(d).lower()[: len("avis extremement defavorable")])
        or "avis totalement defavorable"
        in unidecode(str(d).lower()[: len("avis totalement defavorable")])
    )
    responses.loc[avis_defavorable, "opinion"] = "Défavorable"

    # Sauve les commentaires défavorables dans un fichier
    logging.info(
        f"Nombre de commentaires avec avis défavorable : {len(responses[avis_defavorable])}"
    )
    csv_file = Path(
        data_dir + "/preprocessed/" + consultation + "_avis_defavorable.csv"
    )
    logging.info(f"Sauvegarde des commentaires défavorables dans {csv_file}")
    responses[avis_defavorable].to_csv(
        csv_file, header=True, sep=";", index=False, encoding="utf-8-sig"
    )
    # Suppression des commentaires défavorables
    responses = responses[~avis_defavorable].reset_index(drop=True)
    logging.info(f"Nombre de commentaires restants : {len(responses)} (avis inconnus)")

    # Il ne reste que les commentaires sans avis
    logging.info(f"Nombre de commentaires sans avis : {len(responses)}")
    responses.opinion = "Inconnu"
    # Sauve les commentaires sans avis dans un fichier
    csv_file = Path(data_dir + "/preprocessed/" + consultation + "_avis_inconnu.csv")
    logging.info(f"Sauvegarde des commentaires sans avis dans {csv_file}")
    # Ecriture du fichier résultant
    responses.to_csv(csv_file, header=True, sep=";", index=False, encoding="utf-8-sig")


@main.command()
@click.pass_context
def cluster(ctx: click.Context) -> None:
    """Clustering des commentaires."""
    consultation = ctx.obj["CONSULTATION"]
    data_dir = ctx.obj["DATA_DIRECTORY"]

    pd.set_option("display.max_colwidth", None)

    csv_file = Path(data_dir + "/processed/" + consultation + ".csv")
    responses = pd.read_csv(csv_file, header=0, sep=";", encoding="utf-8-sig")
    logging.info(f"Nombre de commentaires prétraités : {len(responses)}")

    logging.info("Vectorisation des textes")
    stop = ["arrêté", "avis", "loup"]
    tfidf_vectorizer = TfidfVectorizer(
        max_df=1.0, min_df=0.1, stop_words=stop, use_idf=True, ngram_range=(1, 3)
    )
    # Fit vectoriser to NLP processed column
    tfidf_matrix = tfidf_vectorizer.fit_transform(responses.lemma)
    logging.info(f"TF-IDF (n_samples, n_features): {tfidf_matrix.shape}")

    # K-means clustering
    logging.info("K-means clustering")
    true_k = 2
    model = KMeans(n_clusters=true_k, init="k-means++", max_iter=100, n_init=1)
    logging.info("Résumé de clusterisation:")
    labels = ["Favorable", "Défavorable"]
    true_labels = [0 if d == labels[0] else 1 for d in responses.Opinion_estimée]
    pred_labels = list(model.fit_predict(tfidf_matrix))
    responses["pred_label"] = pred_labels
    print(responses[:10])
    print(true_labels[:30])
    print(pred_labels[:30])
    logging.info(f"Homogénéité: {metrics.homogeneity_score(true_labels, pred_labels)}")
    logging.info(f"Rand index: {metrics.adjusted_rand_score(true_labels, pred_labels)}")
    logging.info(
        f"Confusion matrix :\n{metrics.confusion_matrix(true_labels, pred_labels)}"
    )
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf_vectorizer.get_feature_names_out()
    cl_size = Counter(model.labels_)
    for i in range(true_k):
        logging.info(
            f"Cluster {labels[i]}, proportion: {cl_size[i] / len(responses) * 100}%, top terms:"
        )

        top_t = ", ".join([terms[t] for t in order_centroids[i, :20]])
        logging.info(top_t)
        print(responses.checked_text[responses["pred_label"] == i][:10])

    # DBSCAN clustering
    logging.info("DBSCAN clustering")
    model = DBSCAN(eps=2.1, min_samples=10, metric="l1")
    model.fit(tfidf_matrix)
    logging.info("Cluster summary:")
    core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
    core_samples_mask[model.core_sample_indices_] = True
    labels = model.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    logging.info("Estimated number of clusters: %d", n_clusters_)
    logging.info("Estimated number of noise points: %d", n_noise_)
    for i in range(n_clusters_):
        logging.info(
            "Cluster %d, proportion: %d%%", i, cl_size[i] / len(responses) * 100
        )


@main.command()
@click.pass_context
def classify(ctx: click.Context) -> None:
    """Classification des commentaires."""
    logging.info("Classification des commentaires")
    consultation = ctx.obj["CONSULTATION"]
    data_dir = ctx.obj["DATA_DIRECTORY"]

    # Lecture des commentaires annotés
    csv_file = Path(data_dir + "/preprocessed/" + consultation + "_avis_favorable.csv")
    if csv_file.is_file():
        logging.info(f"Lecture des commentaires favorables depuis {csv_file}")
        favorable = pd.read_csv(csv_file, header=0, sep=";", encoding="utf-8-sig")
    else:
        logging.warning(f"Pas de fichier avis_favorable {csv_file}")
    csv_file = Path(
        data_dir + "/preprocessed/" + consultation + "_avis_defavorable.csv"
    )
    if csv_file.is_file():
        logging.info(f"Lecture des commentaires défavorables depuis {csv_file}")
        defavorable = pd.read_csv(csv_file, header=0, sep=";", encoding="utf-8-sig")
    else:
        logging.warning(f"Pas de fichier avis_defavorable {csv_file}")

    csv_file = Path(data_dir + "/preprocessed/" + consultation + "_annotated.csv")
    if csv_file.is_file():
        logging.info(f"Lecture des commentaires annotés depuis {csv_file}")
        annotated = pd.read_csv(csv_file, header=0, sep=";", encoding="utf-8-sig")
        annotated = annotated.dropna(subset=["opinion"])
    else:
        logging.warning(f"Pas de fichier annoted {csv_file}")
        annotated = pd.DataFrame(columns=favorable.columns)
    # Concaténation des commentaires annotés
    annotated = pd.concat([favorable, defavorable, annotated], ignore_index=True)

    # Lecture des commentaires sans avis
    csv_file = Path(data_dir + "/preprocessed/" + consultation + "_avis_inconnu.csv")
    inconnu = pd.read_csv(csv_file, header=0, sep=";", encoding="utf-8-sig")

    # Concaténation des commentaires annotés
    responses = pd.concat([favorable, defavorable, inconnu], ignore_index=True)
    logging.info(f"Lecture de {len(responses)} commentaires prétraités")

    nb_comments = len(responses)
    nb_annotés = len(annotated)
    nb_favorable = len(annotated[annotated.opinion == "Favorable"])
    nb_defavorable = len(annotated[annotated.opinion == "Défavorable"])
    nb_inconnnu = len(inconnu)
    logging.info(
        f"Nombre de commentaires totaux : {nb_comments}, annotés : {nb_annotés}"
    )
    logging.info(
        f"Nombre de commentaires favorables : {nb_favorable} ({nb_favorable / nb_comments * 100:.2f}% du total, {nb_favorable / nb_annotés * 100:.2f}% des annotés)"
    )
    logging.info(
        f"Nombre de commentaires défavorables : {nb_defavorable} ({nb_defavorable / nb_comments * 100:.2f}% du total, {nb_defavorable / nb_annotés * 100:.2f}% des annotés)"
    )
    logging.info(
        f"Nombre de commentaires inconnus : {nb_inconnnu} ({nb_inconnnu / nb_comments * 100:.2f}%)"
    )
    responses.lemma = responses.lemma.fillna(value="Neutre")

    # Sous-échantillonnage des commentaires majoritaires
    logging.info("Sous-échantillonnage des commentaires majoritaires")
    rus = RandomUnderSampler(random_state=42, sampling_strategy=0.5)
    x_res, y_res = rus.fit_resample(annotated[["titre", "lemma"]], annotated.opinion)
    x_res = x_res.lemma

    logging.info(f"Dimensions sous-échantillonnées : {x_res.shape}")
    logging.info(f"Forme initiale : {Counter(annotated.opinion)}")
    logging.info(f"Forme sous-échantillonnée : {Counter(y_res)}")

    # Séparation des données en train et test
    logging.info("Séparation des données en entrainement et test")
    x_train, x_test, y_train, y_test = train_test_split(
        x_res, y_res, test_size=0.2, random_state=42
    )
    logging.info(
        f"Dimensions initiale, Entrainement : {x_train.shape}, Test : {x_test.shape}"
    )

    # Création du modèle de classification
    stop = ["arrete", "avis", "decret", "loup"]
    pipeline_rf = Pipeline(
        [
            (
                "vect",
                TfidfVectorizer(
                    stop_words=stop,
                    decode_error="ignore",
                    lowercase=False,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    max_df=0.55,
                    min_df=30,
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    criterion="gini",
                    max_depth=None,
                    min_samples_leaf=1,
                    min_samples_split=5,
                    n_estimators=300,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline_svc = Pipeline(
        [
            (
                "vect",
                TfidfVectorizer(
                    stop_words=stop,
                    decode_error="ignore",
                    lowercase=False,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    max_df=0.55,
                    min_df=30,
                ),
            ),
            (
                "classifier",
                # ComplementNB(),
                LinearSVC(random_state=42),
            ),
        ]
    )
    pipeline_nb = Pipeline(
        [
            (
                "vect",
                TfidfVectorizer(
                    stop_words=stop,
                    decode_error="ignore",
                    lowercase=False,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    max_df=0.55,
                    min_df=30,
                ),
            ),
            (
                "classifier",
                ComplementNB(),
            ),
        ]
    )

    # # Define pipelines  and their parameter grids
    # pipelines_params = {
    #     "SVC_Pipeline": {
    #         "pipeline": pipeline_svc,
    #         "params": {
    #             "classifier__C": [0.1, 1, 10],
    #         },
    #     },
    #     "RandomForest_Pipeline": {
    #         "pipeline": pipeline_rf,
    #         "params": {
    #             "classifier__n_estimators": [200, 300, 400],
    #         },
    #     },
    #     "NaiveBayes_Pipeline": {
    #         "pipeline": pipeline_nb,
    #         "params": {
    #             "classifier__alpha": np.logspace(2, 6, 5),
    #         },
    #     },
    # }

    # results = {}
    # print("--- Starting Pipeline Tuning (with inner parallelism) ---")
    # for pipeline_name, config in pipelines_params.items():
    #     print(f"Tuning {pipeline_name}...")
    #     grid_search = GridSearchCV(
    #         config["pipeline"],
    #         config["params"],
    #         cv=5,
    #         scoring="f1_weighted",
    #         n_jobs=-1,  # Use all available cores for THIS GridSearchCV run
    #     )
    #     grid_search.fit(x_train, y_train)
    #     results[pipeline_name] = {
    #         "best_score": grid_search.best_score_,
    #         "best_params": grid_search.best_params_,
    #         "best_estimator": grid_search.best_estimator_,
    #     }
    #     print(f"Finished {pipeline_name}. Best score: {grid_search.best_score_:.4f}")

    # # Display results
    # print("\n--- Pipeline Tuning Results ---")
    # for pipeline_name, res in results.items():
    #     print(f"\nPipeline: {pipeline_name}")
    #     print(f" Best Score: {res['best_score']:.4f}")
    #     print(f" Best Params: {res['best_params']}")

    # Entraînement du modèle sur le jeu sous-échantillonné
    pipeline = pipeline_rf
    logging.info(f"Modèle : {pipeline}")
    logging.info(f"Entraînement sur le jeu sous-échantillonné : {x_train.shape}")

    pipeline.fit(x_train, y_train)

    # Prédiction sur le jeu de test
    logging.info(f"Prédiction sur le jeu de test sous-échantillonné : {x_test.shape}")
    y_pred = pipeline.predict(x_test)
    logging.info(f"Accuracy du modèle : {metrics.accuracy_score(y_test, y_pred)}")
    logging.info(
        f"Rapport de classification :\n{metrics.classification_report(y_test, y_pred, labels=['Favorable', 'Défavorable'], digits=4)}"
    )
    logging.info(
        f"Confusion matrix :\n{metrics.confusion_matrix(y_test, y_pred, labels=['Favorable', 'Défavorable'])}"
    )
    logging.info(
        f"Confusion matrix :\n{metrics.confusion_matrix(y_test, y_pred, labels=['Favorable', 'Défavorable'], normalize='all')}"
    )

    # Prédiction sur tous les commentaires
    logging.info("Prédiction sur tous les commentaires")
    responses["Opinion_estimée"] = pipeline.predict(responses.lemma)
    nb_favorable = len(responses[responses.Opinion_estimée == "Favorable"])
    nb_defavorable = len(responses[responses.Opinion_estimée == "Défavorable"])
    logging.info(
        f"Nombre de commentaires favorables : {nb_favorable} ({nb_favorable / len(responses) * 100:.2f}%)"
    )
    logging.info(
        f"Nombre de commentaires défavorables : {nb_defavorable} ({nb_defavorable / len(responses) * 100:.2f}%)"
    )
    logging.info(
        f"Confusion matrix :\n{metrics.confusion_matrix(responses.opinion, responses.Opinion_estimée)}"
    )
    # Forcage des avis connus
    responses.loc[responses.opinion == "Favorable", "Opinion_estimée"] = "Favorable"
    responses.loc[responses.opinion == "Défavorable", "Opinion_estimée"] = "Défavorable"
    nb_favorable = len(responses[responses.Opinion_estimée == "Favorable"])
    nb_defavorable = len(responses[responses.Opinion_estimée == "Défavorable"])
    logging.info(
        f"Nombre de commentaires favorables : {nb_favorable} ({nb_favorable / len(responses) * 100:.2f}%)"
    )
    logging.info(
        f"Nombre de commentaires défavorables : {nb_defavorable} ({nb_defavorable / len(responses) * 100:.2f}%)"
    )
    # Ecriture du fichier résultant
    csv_file = Path(data_dir + "/processed/" + consultation + ".csv")
    logging.info(f"Ecriture dans {csv_file}")
    responses.to_csv(csv_file, header=True, sep=";", index=False, encoding="utf-8-sig")


@main.command()
@click.pass_context
def report(ctx: click.Context) -> None:
    """Rapport sur les tendances des commentaires."""
    logging.info("Rapport sur les tendances des commentaires")
    consultation = ctx.obj["CONSULTATION"]
    data_dir = ctx.obj["DATA_DIRECTORY"]

    # Lecture des commentaires annotés
    csv_file = Path(data_dir + "/processed/" + consultation + ".csv")
    if csv_file.is_file():
        logging.info(f"Lecture des commentaires classifiés depuis {csv_file}")
        commentaires = pd.read_csv(
            csv_file,
            header=0,
            sep=";",
            encoding="utf-8-sig",
            parse_dates=["dateheure"],
        )
    else:
        logging.warning(f"Pas de fichier {csv_file}")

    commentaires["date"] = commentaires.dateheure.dt.date
    opinions = (
        commentaires.groupby(["date", "Opinion_estimée"]).size().unstack().fillna(0)
    )
    print(opinions)
    cum_opinions = opinions.cumsum()
    cum_opinions["Total"] = cum_opinions.sum(axis=1)
    cum_opinions["Pct Favorable"] = (
        cum_opinions["Favorable"] / cum_opinions["Total"] * 100
    )
    print(cum_opinions)
    logging.info("Génération de l'évolution des commentaires dans le temps")
    plt.figure(figsize=(1024 / 100, 768 / 100), dpi=100)
    sns.histplot(
        data=commentaires,
        x="date",
        hue="Opinion_estimée",
        palette={"Favorable": "red", "Défavorable": "green"},
        multiple="dodge",
        shrink=0.8,
        bins=21,
    )
    plt.xlim(
        datetime.strptime("26/11/2025", "%d/%m/%Y"),
        datetime.strptime("20/12/2025", "%d/%m/%Y"),
    )
    plt.ylim(0, 2500)
    plt.text(
        x=datetime.strptime("27/11/2025", "%d/%m/%Y"),
        y=2200,
        s=" FNSEA/JA",
        fontsize=12,
        color="red",
    )
    plt.axvline(
        x=datetime.strptime("27/11/2025", "%d/%m/%Y"),
        color="red",
        linestyle="--",
    )

    plt.text(
        x=datetime.strptime("28/11/2025", "%d/%m/%Y"),
        y=2400,
        s=" ASPAS",
        fontsize=12,
        color="green",
    )
    plt.axvline(
        x=datetime.strptime("28/11/2025", "%d/%m/%Y"),
        color="green",
        linestyle="--",
    )
    plt.text(
        x=datetime.strptime("01/12/2025", "%d/%m/%Y"),
        y=2400,
        s=" FERUS",
        fontsize=12,
        color="green",
    )
    plt.axvline(
        x=datetime.strptime("01/12/2025", "%d/%m/%Y"),
        color="green",
        linestyle="--",
    )
    plt.text(
        x=datetime.strptime("05/12/2025", "%d/%m/%Y"),
        y=2400,
        s=" FNE",
        fontsize=12,
        color="green",
    )
    plt.axvline(
        x=datetime.strptime("05/12/2025", "%d/%m/%Y"),
        color="green",
        linestyle="--",
    )
    plt.text(
        x=datetime.strptime("07/12/2025", "%d/%m/%Y"),
        y=2200,
        s=" Fédé chasse",
        fontsize=12,
        color="red",
    )
    plt.axvline(
        x=datetime.strptime("07/12/2025", "%d/%m/%Y"),
        color="red",
        linestyle="--",
    )
    plt.text(
        x=datetime.strptime("09/12/2025", "%d/%m/%Y"),
        y=2400,
        s=" LPO",
        fontsize=12,
        color="green",
    )
    plt.axvline(
        x=datetime.strptime("09/12/2025", "%d/%m/%Y"),
        color="green",
        linestyle="--",
    )

    plt.text(
        x=datetime.strptime("09/12/2025", "%d/%m/%Y"),
        y=2300,
        s=" H&B",
        fontsize=12,
        color="green",
    )
    plt.axvline(
        x=datetime.strptime("09/12/2025", "%d/%m/%Y"),
        color="green",
        linestyle="--",
    )
    plt.title("Répartition des commentaires dans le temps")
    plt.xlabel("Date")
    plt.ylabel("Nombre de commentaires")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"plots/{consultation}_comments_over_time.png")

    logging.info("Génération du pourcentage de commentaires favorables dans le temps")
    plt.figure(figsize=(1024 / 100, 768 / 100), dpi=100)
    sns.lineplot(
        data=cum_opinions,
        x=cum_opinions.index,
        y="Pct Favorable",
        marker="o",
        color="blue",
    )
    plt.title("Pourcentage de commentaires favorables dans le temps")
    plt.xlabel("Date")
    plt.ylabel("Pourcentage de commentaires favorables (%)")
    plt.xticks(rotation=90)
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{consultation}_favorable_percentage_over_time.png")


if __name__ == "__main__":
    main(obj={})  # pragma: no cover
