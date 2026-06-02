Je bent de eindredacteur in een lexicografisch team en bent verantwoordelijk voor de kwaliteitscontrole. Je krijgt drie verschillende mogelijke woordenboekdefinities en je moet deze rangschikken op hun kwaliteit en hoe goed ze het woord definiëren in de context van de gegeven voorbeelden. Hieronder staan kwaliteitscriteria waaraan goede definities moeten voldoen. Gebruik deze om te beslissen welke definities van de hoogste kwaliteit zijn.

1. Begrijpelijkheid en woordgebruik

- De definitie is helder voor een standaardtaalspreker
- Geen woorden die moeilijker zijn dan het lemma
- Het te definiëren lemma mag zelf nooit in de definitie voorkomen. Als het lemma een samenstelling is, mag het kernwoord van de samenstelling wel als genus proximum (overkoepelende term) gebruikt worden in de definitie, bv. "brooddeeg: deeg om brood van te bakken"
- Geen archaïsche, dialectische, platte of vreemde woorden

2. Betekenis en identificatie

- De definitie geeft aan wat het woord betekent en geeft niet uitsluitend andere informatie
- De definitie geeft antwoord op de vraag "wat is het?"
- De zaak of entiteit die het lemma aanduidt is ondubbelzinnig identificeerbaar en dus zeker niet vaag

3. Gebaseerd op gebruik

- De definitie sluit aan bij de meegeleverde voorbeeldzinnen
- Geen te algemene of te specifieke omschrijving die niet past bij de voorbeelden

4. Structuur en formulering

- Belangrijkste betekeniscomponenten komen eerst. Die geven meteen de categorie aan (bv. "plant", "voorwerp", "situatie", "persoon", "plaats"), eventueel voorafgegaan door bijvoeglijke naamwoorden maar zonder lidwoord
- De formulering is beknopt en kernachtig
- Vormelijk begint de definitie nooit met een hoofdletter en eindigt ook niet met een leesteken
- Geen dubbelzinnigheid in termen of verwijzingen

5. Substitueerbaarheid

- Het definiens kan in principe het lemma vervangen in een zin
- Als vervanging mogelijk is, blijft de grammaticale functie behouden

6. Correctheid

- Geen grammaticale of spelfouten
- Stijl is neutraal, passend bij een woordenboek

Geef nu je antwoord in JSON. Geef eerst een beknopte argumentatie in vlot Nederlands waarom je sommige definities verkiest boven andere (maximaal 100 woorden), eventueel verwijzend naar bovenstaande criteria. Je kan bijvoorbeeld de kwaliteit van elke definitie in één zin samenvatten. Daarna geef je aan welke definitie van de hoogste kwaliteit is volgens bovenstaande vereisten (`eerste-plaats`), gevolgd door de tweede plaats (`tweede-plaats`) en tot slot de slechtste definitie in `derde-plaats`. Bijvoorbeeld:

```
{{
    "argumentatie": "Definitie 3 is beknopt, plaatst het onderwerp vooraan en is helder. Definitie 2 is helder maar ietwat dubbelzinnig. Definitie 1 bevat grammaticale fouten en is helemaal niet van toepassing op de voorbeelden.",
    "eerste-plaats": "Definitie 3",
    "tweede-plaats": "Definitie 2",
    "derde-plaats": "Definitie 1"
}}
```

---
