import re

import scrapy


class MteCrawlerSpider(scrapy.Spider):
    name = "mte_crawler"
    allowed_domains = ["consultations-publiques.developpement-durable.gouv.fr"]

    ## Modifier le nom de la consultation ci-dessous
    _start_url = "http://www.consultations-publiques.developpement-durable.gouv.fr/spip.php?page=article&id_article=2569"
    # "projet-de-decret-pris-en-application-de-l-article-a2569"
    ## Modifier le nombre de commentaires ci-dessous
    _max_comments = 4087

    def start_requests(self):
        ## Création de la liste des pages
        self.logger.info("Téléchargement de la première page")
        urls = [
            self._start_url,
        ]
        for page in range(20, self._max_comments, 20):
            urls.append(
                self._start_url + "&debut_forums=" + str(page) + "#pagination_forums"
            )
        print(urls[2])
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        nb_com = 0
        for ligne in response.css("div.ligne-com"):
            nb_com += 1
            yield {
                "sujet": ligne.css("div.titresujet::text").getall(),
                "texte": ligne.css("div.textesujet *::text").getall(),
            }
        self.logger.info("Extraction de %d commentaires", nb_com)
