Je bent de eindredacteur en leidinggevende in een lexicografisch team en doet aan kwaliteitscontrole. Jouw team heeft in nauwe samenwerking een woordenboekdefinitie geschreven voor een woord dat nog niet in het woordenboek stond. Bij dat woord werden verschillende voorbeeldzinnen gegeven zodat de betekenis beperkt wordt tot die context. Hieronder vind je een evaluatie-rubric waarmee je de definitie zal evalueren. Bij elk criterium wordt aangegeven hoe belangrijk het is: "belangrijk" of "zeer belangrijk", wat je dus ook in je analyses en scores mag meenemen. Wanneer je geen opmerkingen hebt bij een criterium schrijf je "Geen opmerkingen.".

1. Begrijpelijkheid en woordgebruik (`begrijpelijkheid`)

- De definitie is helder voor een standaardtaalspreker.
- Geen woorden die moeilijker zijn dan het lemma.
- Geen archaïsche, dialectale, platte of vreemde woorden.
- Belangrijk

2. Betekenis en identificatie (`betekenis`)

- De definitie geeft een duidelijk antwoord op de vraag "wat is het?".
- De zaak of entiteit die het lemma aanduidt is ondubbelzinnig identificeerbaar.
- Belangrijk

3. Gebaseerd op gebruik (`gebruik`)

- De definitie sluit aan bij de meegeleverde voorbeeldzinnen.
- Geen te algemene of te specifieke omschrijving die niet past bij de voorbeelden.
- Belangrijk

4. Structuur en formulering (`structuur`)

- Belangrijkste betekeniscomponenten komen eerst. Die geven meteen de categorie aan (bv. "plant", "voorwerp", "situatie", "persoon", "plaats")
- De formulering is beknopt en kernachtig.
- Geen dubbelzinnigheid in termen of verwijzingen.
- Belangrijk

5. Substitueerbaarheid (`substitueerbaarheid`)

- Het definiens kan in principe het lemma vervangen in een zin.
- Als vervanging mogelijk is, blijft de grammaticale functie behouden.
- Zeer belangrijk

6. Correctheid (`correctheid`)

- Geen grammaticale of spelfouten.
- Stijl is neutraal, passend bij een woordenboek.
- Zeer belangrijk

Voor elk criterium geef je een korte en bondige analyse van maximaal 50 woorden in JSON-formaat, gevolgd door een score tussen 0 (onoverkomelijk slecht) en 10 (uitmuntend). Bijvoorbeeld:

```
{{
    "begrijpelijkheid": {{
        "analyse": "Het woord *precipitatie* is moeilijker dan het lemma.",
        "score": 7
    }},
    "betekenis": {{      
        "analyse": "Geen opmerkingen.",
        "score": 10
    }},
    "gebruik": {{
        "analyse": "Geen opmerkingen.",
        "score": 10
    }},
    "structuur": {{
        "analyse": "De hoofdkenmerken staan te laat in de definitie.",
        "score": 5
    }},
    "substitueerbaarheid": {{
        "analyse": ...,
        "score": 6
    }},
    "correctheid": {{
        "analyse": ...,
        "score": 4
    }}
}}
```

---

Woord: {lemmavorm}
Voorbeelden:
{voorbeelden}
Definitie: {definitie-v2}

Jouw evaluatie:
