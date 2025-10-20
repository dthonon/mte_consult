"""Command-line interface."""

import locale
import logging
import random
import re
import time
from collections import Counter
from functools import partial

# import unicodedata
from pathlib import Path
from typing import Any

import click
import hunspell  # type: ignore
import numpy as np
import pandas as pd
import requests
import spacy
from bs4 import BeautifulSoup
from imblearn.under_sampling import RandomUnderSampler
from lingua import Language, LanguageDetectorBuilder
from sklearn import metrics  # type: ignore
from sklearn.cluster import (
    DBSCAN,  # type: ignore
    KMeans,
)
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.svm import LinearSVC  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from spacy.tokenizer import Tokenizer  # type: ignore
from textacy import preprocessing

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
        logging.debug(f"Lecture des téléchargements depuis {csv_file}")
        responses = pd.read_csv(csv_file, header=0, sep=";")
    else:
        logging.debug("Pas de téléchargement précédents")
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
                logging.debug(f"Ecriture dans {csv_file}")
                responses.to_csv(csv_file, header=True, sep=";", index=False)
                time.sleep(10)
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


def _spell_correction2(doc: Tokenizer, spell: Any, corrected: pd.DataFrame) -> str:
    """Spell correction of misspelled words, pyspellchecker version."""
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
        elif d.is_stop or d.is_punct or word in spell:
            text += d.text_with_ws
        else:
            sp = spell.correction(word)
            if sp is not None:
                corrected.loc[len(corrected)] = {
                    "raw_text": word,
                    "checked_text": sp,
                }
                text += sp + d.whitespace_
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
    logging.debug("Lecture %s", csv_file)
    responses = pd.read_csv(csv_file, header=0, sep=";", names=["sujet", "texte"])
    logging.info(f"Nombre de commentaires bruts : {len(responses)}")

    # Découpe du sujet en éléments
    responses[["titre", "dateheure"]] = responses.sujet.str.extract(
        "(.*), le (.* à .*)", expand=True
    )
    responses = responses[["titre", "dateheure", "texte", "sujet"]]
    locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")
    responses["dateheure"] = pd.to_datetime(
        responses["dateheure"], format="%d %B %Y à %Hh%M", errors="coerce"
    )

    # Suppression des ligne dupliquées
    responses = responses.drop_duplicates(subset=["titre", "texte"])
    logging.info(
        f"Commentaires restants après déduplication du titre et du texte: {len(responses)}"
    )

    # Fusion en une colonne pour traitement du texte
    responses["raw_text"] = responses["titre"] + ". " + responses["texte"]
    responses["raw_text"] = responses["raw_text"].fillna(value="?")
    # responses = responses.drop(columns=["texte"])

    # Nettoyage du texte brut
    logging.info("Nettoyage du texte brut")
    responses["raw_text"] = responses["raw_text"].str.lower()
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
    responses.raw_text = responses.raw_text.str.replace(r"\r", " ", regex=True)
    responses.raw_text = responses.raw_text.str.replace("+", " plus ")
    responses.raw_text = responses.raw_text.str.replace("*", " fois ")
    responses.raw_text = responses.raw_text.str.replace("grd", "grand")
    responses.raw_text = responses.raw_text.str.replace("qq", "quelque")
    responses.raw_text = responses.raw_text.str.replace("qlqs", "quelques")
    responses.raw_text = responses.raw_text.str.replace(".euses", "")
    responses.raw_text = responses.raw_text.str.replace(".", ". ")
    responses.raw_text = responses.raw_text.str.replace("(", " (")
    responses.raw_text = responses.raw_text.str.replace(")", ") ")
    # responses.raw_text = responses.raw_text.str.replace(r"\s+", " ", regex=True)
    responses.raw_text = responses.raw_text.str.replace(r"[_%=/°]", " ", regex=True)
    responses.raw_text = responses.raw_text.str.replace(
        r"\d?\dh\d\d", "HEURE", regex=True
    )

    # Suppression du texte étranger
    lang = responses.raw_text.apply(lambda d: detector.detect_language_of(d))
    pd.set_option("display.max_colwidth", None)
    for t in responses.raw_text[lang != Language.FRENCH]:
        logging.info(f"Langue {detector.detect_language_of(t)} : {t}")
        confidence_values = detector.compute_language_confidence_values(t)
        for confidence in confidence_values:
            logging.debug(f"{confidence.language.name}: {confidence.value:.2f}")
    responses.drop(responses[lang != Language.FRENCH].index, inplace=True)
    logging.info(f"Commentaires français restant : {len(responses)}")

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
        lambda d: _spell_correction(tokenizer(d), spell, corrected)
    )
    # Correction orthographique avec pyspellchecker => très lent
    # spell = SpellChecker(language="fr")
    # if added_words.is_file():
    #     logging.info(f"Ajout des mots du fichier {added_words}")
    #     spell.word_frequency.load_text_file(added_words)
    # responses["checked_text"] = responses["raw_text"].apply(
    #     lambda d: _spell_correction2(tokenizer(d), spell, corrected)
    # )

    logging.info(f"Nombre de mots corrigés : {len(corrected)}")
    corrected = corrected.drop_duplicates(
        subset=["raw_text", "checked_text"]
    ).sort_values(by="raw_text")

    # Lemmatisation des commentaires
    logging.info("Lemmatisation des commentaires")
    responses["lemma"] = responses["checked_text"].apply(lambda d: _lemmatize(nlp(d)))

    # Ecriture du fichier des corrections orthographiques
    csv_file = Path(data_dir + "/preprocessed/" + consultation + "_corrected.csv")
    logging.debug(f"Ecriture dans {csv_file}")
    corrected.to_csv(csv_file, header=True, sep=";", index=False)

    # Ecriture du fichier résultant
    csv_file = Path(data_dir + "/preprocessed/" + consultation + ".csv")
    logging.debug(f"Ecriture dans {csv_file}")
    responses.sort_values(by="dateheure", ignore_index=True).to_csv(
        csv_file, header=True, sep=";", index=False
    )


@main.command()
@click.pass_context
def cluster(ctx: click.Context) -> None:
    """Clustering des commentaires."""
    consultation = ctx.obj["CONSULTATION"]
    data_dir = ctx.obj["DATA_DIRECTORY"]

    pd.set_option("display.max_colwidth", None)

    csv_file = Path(data_dir + "/processed/" + consultation + ".csv")
    logging.debug(f"Lecture de {csv_file}")
    responses = pd.read_csv(csv_file, header=0, sep=";")
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
def pretrain(ctx: click.Context) -> None:
    """Préparation des fichiers d'entrainement annotés."""
    consultation = ctx.obj["CONSULTATION"]
    data_dir = ctx.obj["DATA_DIRECTORY"]

    # Lecture des commentaires
    csv_file = Path(data_dir + "/preprocessed/" + consultation + ".csv")
    if csv_file:
        responses = pd.read_csv(csv_file, header=0, sep=";")
        logging.info(
            f"Lecture de {len(responses)} commentaires prétraités depuis {csv_file}"
        )
    else:
        logging.error(f"Fichier {csv_file} non trouvé")
        return

    # Ajout de la colonne opinion
    if "opinion" not in responses.columns:
        logging.info("Ajout de la colonne opinion")
        responses["opinion"] = ""

    # Découpage selon l'avis exprimé
    # Séparation des commentaires favorables
    avis_favorable = responses.checked_text.apply(
        lambda d: "avis favorable" in str(d).lower()[: len("avis favorable")]
        or "favorable" in str(d).lower()[: len("favorable")]
        or "avis très favorable" in str(d).lower()[: len("avis très favorable")]
    )
    logging.info(
        f"Nombre de commentaires avec avis favorable : {len(responses[avis_favorable])}"
    )
    responses.loc[avis_favorable, "opinion"] = "Favorable"
    # Sauve les commentaires favorables dans un fichier
    csv_file = Path(data_dir + "/preprocessed/" + consultation + "_avis_favorable.csv")
    logging.info(f"Sauvegarde des commentaires favorables dans {csv_file}")
    responses[avis_favorable].to_csv(csv_file, header=True, sep=";", index=False)
    # Suppression des commentaires favorables
    responses = responses[~avis_favorable].reset_index(drop=True)
    logging.info(
        f"Nombre de commentaires restants : {len(responses)} (avis défavorable ou inconnus)"
    )

    # Séparation des commentaires défavorables
    avis_defavorable = responses.checked_text.apply(
        lambda d: "non favorable" in str(d).lower()[: len("non favorable")]
        or "défavorable" in str(d).lower()[: len("défavorable")]
        or "très défavorable" in str(d).lower()[: len("très défavorable")]
        or "complètement défavorable"
        in str(d).lower()[: len("complètement défavorable")]
        or "avis défavorable" in str(d).lower()[: len("avis défavorable")]
        or "avis très défavorable" in str(d).lower()[: len("avis très défavorable")]
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
    responses[avis_defavorable].to_csv(csv_file, header=True, sep=";", index=False)
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
    responses.to_csv(csv_file, header=True, sep=";", index=False)


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
        favorable = pd.read_csv(csv_file, header=0, sep=";")
    else:
        logging.warning(f"Pas de fichier avis_favorable {csv_file}")
    csv_file = Path(
        data_dir + "/preprocessed/" + consultation + "_avis_defavorable.csv"
    )
    if csv_file.is_file():
        logging.info(f"Lecture des commentaires défavorables depuis {csv_file}")
        defavorable = pd.read_csv(csv_file, header=0, sep=";")
    else:
        logging.warning(f"Pas de fichier avis_defavorable {csv_file}")

    csv_file = Path(data_dir + "/preprocessed/" + consultation + "_annotated.csv")
    if csv_file.is_file():
        logging.info(f"Lecture des commentaires annotés depuis {csv_file}")
        annotated = pd.read_csv(csv_file, header=0, sep=";")
        annotated = annotated.dropna(subset=["opinion"])
    else:
        logging.warning(f"Pas de fichier annoted {csv_file}")
        annotated = pd.DataFrame(columns=favorable.columns)
    # Concaténation des commentaires annotés
    annotated = pd.concat([favorable, defavorable, annotated], ignore_index=True)

    # Lecture des commentaires sans avis
    csv_file = Path(data_dir + "/preprocessed/" + consultation + "_avis_inconnu.csv")
    inconnu = pd.read_csv(csv_file, header=0, sep=";")
    # Concaténation des commentaires annotés
    responses = pd.concat([favorable, defavorable, inconnu], ignore_index=True)
    logging.info(f"Lecture de {len(responses)} commentaires prétraités")

    nb_comments = len(annotated)
    nb_favorable = len(annotated[annotated.opinion == "Favorable"])
    nb_defavorable = len(annotated[annotated.opinion == "Défavorable"])
    nb_inconnnu = len(annotated[annotated.opinion == "Inconnu"])
    logging.info(f"Nombre de commentaires annotés : {nb_comments}")
    logging.info(
        f"Nombre de commentaires favorables : {nb_favorable} ({nb_favorable / nb_comments * 100:.2f}%)"
    )
    logging.info(
        f"Nombre de commentaires défavorables : {nb_defavorable} ({nb_defavorable / nb_comments * 100:.2f}%)"
    )
    logging.info(
        f"Nombre de commentaires inconnus : {nb_inconnnu} ({nb_inconnnu / nb_comments * 100:.2f}%)"
    )

    # Création du modèle de classification
    logging.info("Vectorisation des textes pré-traités")
    responses.lemma = responses.lemma.fillna(value="Neutre")
    stop = ["arrete", "avis"]
    tfidf_vectorizer = TfidfVectorizer(
        decode_error="ignore",  # Ignore decoding errors
        strip_accents="unicode",  # Normalize accents
        lowercase=False,
        max_df=0.98,  # Ignore terms that appear in more than x% of the documents
        min_df=0.1,  # Ignore terms that appear in less than x% of the documents
        stop_words=stop,
        use_idf=True,
        ngram_range=(1, 3),
    )
    # Fit vectoriser to NLP processed column
    tfidf_matrix = tfidf_vectorizer.fit_transform(responses.lemma)
    tf_rows, tf_cols = tfidf_matrix.shape
    logging.info(f"TF-IDF (n_samples, n_features): {tf_rows}, {tf_cols}")

    # Sous-échantillonnage des commentaires majoritaires
    logging.info("Sous-échantillonnage des commentaires majoritaires")
    rus = RandomUnderSampler(random_state=42)
    x_res, y_res = rus.fit_resample(annotated.lemma.to_frame(), annotated.opinion)
    x_res = x_res.lemma
    logging.info(f"Dimensions sous-échantillonnées : {len(x_res)}")
    logging.info(f"Forme initiale : {Counter(annotated.opinion)}")
    logging.info(f"Forme sous-échantillonnée : {Counter(y_res)}")
    # Séparation des données en train et test
    logging.info("Séparation des données en entrainement et test")
    x_train, x_test, y_train, y_test = train_test_split(
        x_res, y_res, test_size=0.2, random_state=42
    )
    logging.info(
        f"Dimensions initiale, Entrainement : {len(x_train)}, Test : {len(x_test)}"
    )

    # Entraînement du modèle
    logging.info(
        "Entraînement du modèle de classification RandomForestClassifier(log_loss)"
    )
    # classifier = LogisticRegression()
    classifier = RandomForestClassifier(
        criterion="log_loss", n_estimators=100, random_state=42
    )
    # classifier = LinearSVC
    # Create pipeline using TfidfVectorizer
    pipe = Pipeline(
        [
            ("vectorizer", tfidf_vectorizer),
            ("classifier", classifier),
        ]
    )
    # Entraînement du modèle sur le jeu sous-échantillonné
    logging.info(
        f"Entraînement du modèle sur le jeu sous-échantillonné : {len(x_train)} commentaires"
    )
    pipe.fit(x_train, y_train)
    # Prédiction sur le jeu de test
    logging.info(
        f"Prédiction sur le jeu de test sous-échantillonné : {len(x_test)} commentaires"
    )
    y_pred = pipe.predict(x_test)
    logging.info(f"Accuracy du modèle : {metrics.accuracy_score(y_train, y_pred)}")
    logging.info(
        f"Rapport de classification :\n{metrics.classification_report(y_train, y_pred, labels=['Favorable', 'Défavorable'], digits=4)}"
    )
    logging.info(
        f"Confusion matrix :\n{metrics.confusion_matrix(y_train, y_pred, labels=['Favorable', 'Défavorable'])}"
    )
    logging.info(
        f"Confusion matrix :\n{metrics.confusion_matrix(y_train, y_pred, labels=['Favorable', 'Défavorable'], normalize='all')}"
    )

    # Prédiction sur tous les commentaires
    logging.info("Prédiction sur tous les commentaires")
    responses["Opinion_estimée"] = pipe.predict(responses.lemma)
    nb_favorable = len(responses[responses.Opinion_estimée == "Favorable"])
    nb_defavorable = len(responses[responses.Opinion_estimée == "Défavorable"])
    logging.info(
        f"Nombre de commentaires favorables : {nb_favorable} ({nb_favorable / len(responses) * 100:.2f}%)"
    )
    logging.info(
        f"Nombre de commentaires défavorables : {nb_defavorable} ({nb_defavorable / len(responses) * 100:.2f}%)"
    )
    # Forcage des avis connus
    logging
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
    responses.to_csv(csv_file, header=True, sep=";", index=False)


if __name__ == "__main__":
    main(obj={})  # pragma: no cover
