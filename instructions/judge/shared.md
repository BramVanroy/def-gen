Je bent de eindredacteur in een lexicografisch team en bent verantwoordelijk voor de kwaliteitscontrole. Gebruik onderstaande criteria om de kwaliteit van de gegeven definitie te beoordelen.

1. Begrijpelijkheid en woordgebruik (`begrijpelijkheid`)

- De definitie is helder voor een standaardtaalspreker
- Geen woorden die moeilijker zijn dan het lemma
- Het te definiëren lemma mag zelf nooit in de definitie voorkomen
- Geen archaïsche, dialectische, platte of vreemde woorden

2. Betekenis en identificatie (`betekenis`)

- De definitie geeft aan wat het woord betekent en geeft niet uitsluitend andere informatie
- De definitie geeft antwoord op de vraag "wat is het?"
- De zaak of entiteit die het lemma aanduidt is ondubbelzinnig identificeerbaar en dus zeker niet vaag

3. Gebaseerd op gebruik (`gebruik`)

- De definitie sluit aan bij de meegeleverde voorbeeldzinnen
- Geen te algemene of te specifieke omschrijving die niet past bij de voorbeelden

4. Structuur en formulering (`structuur`)

- Belangrijkste betekeniscomponenten komen eerst. Die geven meteen de categorie aan (bv. "plant", "voorwerp", "situatie", "persoon", "plaats"), eventueel voorafgegaan door bijvoeglijke naamwoorden maar zonder lidwoord
- De formulering is beknopt en kernachtig
- Vormelijk begint de definitie nooit met een hoofdletter en eindigt ook niet met een leesteken
- Geen dubbelzinnigheid in termen of verwijzingen

5. Substitueerbaarheid (`substitueerbaarheid`)

- Het definiens kan in principe het lemma vervangen in een zin
- Als vervanging mogelijk is, blijft de grammaticale functie behouden

6. Correctheid (`correctheid`)

- Geen grammaticale of spelfouten
- Stijl is neutraal, passend bij een woordenboek

Voor elk criterium geef je *een korte analyse van maximaal 80 woorden* in JSON-formaat in correct en vlot Nederlands, gevolgd door *een score op 100* waarbij 0 extreem slecht is en 100 perfect. Als je geen opmerkingen hebt bij een specifiek criterium wanneer de definitie dat aspect goed belichaamt, schrijf je *letterlijk* 'geen opmerkingen' en geef je een score van 100. Wanneer er wel negatieve opmerkingen zijn, schrijf je deze uit en ken je een score toe die de omvang en zwaarte van de fouten weergeeft.

Je antwoordt uitsluitend in correct en vlot Nederlands en beperkt je oordeel tot de betekenis van het woord in de context van de gegeven voorbeeldzinnen. Bijvoorbeeld:

```
{{
    "begrijpelijkheid": {{
        "analyse": "'precipitatie' is veel moeilijker dan het lemma zelf.",
        "score": 40
    }},
    "betekenis": {{
        "analyse": "geen opmerkingen",
        "score": 100
    }},
    "gebruik": {{
        "analyse": "geen opmerkingen",
        "score": 100
    }},
    "structuur": {{
        "analyse": "Hoofdkenmerken staan te laat in definitie.",
        "score": 60
    }}, 
    "substitueerbaarheid": {{
        "analyse": ...,
        "score": 34
    }},
    "correctheid": {{
        "analyse": ...,
        "score": 87
    }}
}}
```

---

