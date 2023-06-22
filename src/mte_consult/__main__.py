"""Command-line interface."""
import csv
import hashlib
import logging

import click
import pandas as pd

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
    ctx.obj["START_COMMENT"] = start_comment
    ctx.obj["END_COMMENT"] = end_comment


@main.command()
@click.pass_context
def preprocess(ctx: click.Context) -> None:
    """Prétraitement du fichier brut contenant les commentaires."""
    consultation = ctx.obj['CONSULTATION']
    data_dir = ctx.obj["DATA_DIRECTORY"]
    logging.info(f"Prétraitement de {consultation} dans {data_dir}")
    csv_file = data_dir + "/raw/" + consultation + ".csv"
    logging.debug("Lecture %s", csv_file)
    responses = pd.read_csv(csv_file, header=0, quoting=csv.QUOTE_ALL, nrows=1000000)
    logging.info("Nombre de commentaires bruts : %d", len(responses))

    # Ajout hash-key pour assurer la traçabilité en cours de traitement
    responses["uid"] = responses.sujet.apply(
        lambda t: hashlib.sha224(t.encode("utf-8")).hexdigest()
    )

    # Découpe du sujet en éléments
    responses[["titre", "nom", "date", "heure"]] = responses.sujet.str.extract(
        "(.*), par  (.*) ,, le (.*) à (.*)", expand=True
    )
    responses = responses.drop(columns=["sujet"])
    responses = responses[
        ["titre", "nom", "date", "heure", "texte", "uid"]
    ].sort_values(by="nom", ignore_index=True)

    # Suppression des ligne dupliquées
    # responses.drop_duplicates(subset=["nom", "titre"], inplace=True)
    responses = responses.drop_duplicates(subset=["nom", "texte"])
    logging.info("Commentaires restants après déduplication : %d", len(responses))

    print(responses.head(10))
    csv_file = data_dir + "/preprocessed/" + consultation + ".csv"
    logging.debug("Ecriture dans %s", csv_file)
    responses.to_csv(csv_file, header=False, quoting=csv.QUOTE_ALL)


@main.command()
def process() -> None:
    """Traitement principal des données."""
    click.echo("Traitement principal des données")


if __name__ == "__main__":
    main(obj={})  # pragma: no cover
