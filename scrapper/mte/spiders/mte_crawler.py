"""Aspirateur pour une consultation MTE."""
from typing import Any, Generator, Self
import scrapy  # type: ignore


class MteCrawlerSpider(scrapy.Spider):
    """Méthodes pour le parcours du site."""

    name = "mte_crawler"
    allowed_domains = ["consultations-publiques.developpement-durable.gouv.fr"]

    # Modifier le nom de la consultation ci-dessous
    _start_url = (
        "http://www.consultations-publiques.developpement-durable.gouv.fr/"
        + "projet-de-plan-national-d-actions-2024-2029-sur-le-a2940.html"
    )

    # Modifier le nombre de commentaires ci-dessous
    _max_comments = 1000

    def start_requests(self: Self) -> Generator[scrapy.Request, Any, Any]:
        """Création de la liste des pages."""
        self.logger.info("Création de la liste des pages à télécharger")
        urls = [
            self._start_url,
        ]
        for page in range(20, self._max_comments, 20):
            urls.append(self._start_url + "&debut_forums=" + str(page))
        # Parcours des commentaires depuis la fin pour minimiser les doublons, au détriment des pertes
        for url in reversed(urls):
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self: Self, response: Any) -> Generator[Any, Any, Any]:
        """Analyse des pages reçues."""
        nb_com = 0
        for ligne in response.css("div.ligne-com"):
            nb_com += 1
            yield {
                "sujet": ligne.css("div.titresujet::text").getall(),
                "texte": ligne.css("div.textesujet *::text").getall(),
            }
        self.logger.info("Extraction de %d commentaires", nb_com)
