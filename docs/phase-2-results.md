# 2. fázis – ML pipeline, MLflow és Model Registry

Ellenőrzés dátuma: 2026-10-31.

## Megvalósítás

- MLflow 3.17.2, SQLAlchemy 2.0.44, perzisztens SQLite-backend és helyi artifacttároló.
- Átlagot becslő `DummyRegressor` baseline és determinisztikus Random Forest challenger.
- Teljes, raw-input kompatibilis sklearn Pipeline a preprocessinggel és regresszorral együtt.
- Train/validation CSV-k hash-, séma- és időbeli ellenőrzése; a test halmaz kimarad a tanításból.
- Paraméterek, train és validation metrikák, input datasetek és teljes SHA-256 referenciák MLflow-ban.
- Inputpélda, signature, saját kód és verziózott környezet a mentett modellben.
- Model Registry-verzió minden jelölthöz, `champion` alias, automatikus első promóció.
- Azonos validáción újraértékelt champion; szigorú MAE-javulást és konfigurálható margót követelő kapu.
- Promóciós indok, validációs predikciók és provenance artifactok.
- Közös fájlzárral sorosított tanítások; hiba esetén `FAILED` run, sikeres eredmény nélkül.
- Aliasfeloldás minden modellbetöltéskor, a betöltött verzióhoz kötött válasszal.
- JSON kimenetű `garments-train` CLI és ténylegesen futtatott MLflow Project train entrypoint.

## Valódi futások a kiválasztott dataseten

Train: 711 sor; validation: 211 sor. A 275 soros végső test adatok nem vettek részt a folyamatban.
A Random Forest 100 fát, 8-as maximális mélységet, 3-as minimális levélméretet és 42-es seedet használ.

| Verzió | Modell | Validációs MAE | Döntés | Aktív champion |
| --- | --- | ---: | --- | --- |
| 1 | Dummy baseline | 0.1467638136 | Kezdeti promóció | 1 |
| 2 | Random Forest | 0.1131253775 | Javulás, automatikus promóció | 2 |
| 3 | Dummy baseline | 0.1467638136 | Rosszabb jelölt elutasítva | 2 |
| 4 | Random Forest, MLflow Projectből | 0.1131253775 | Azonos eredmény elutasítva | 2 |

| Verzió | MLflow run ID |
| --- | --- |
| 1 | `0972df3718be405784dd327b7fda44f0` |
| 2 | `c462cc21ecbf494cb8efa985ea89a3df` |
| 3 | `16db6b359a674a58bfbac979670b7123` |
| 4 | `6acac8f405aa4ddcb91e3dfb1261859f` |

Mind a négy modellverzió `READY`, a champion a 2-es verzió. Az MLflow felület elindult a
`http://127.0.0.1:5000` címen. A kezdőlap és a `/health` végpont HTTP 200 választ adott;
a Registry HTTP API-ján is ellenőriztük a négy verziót és az aliast.

Az állapot a Gitből kizárt `runtime/` alatt van. Új klónban a fenti run ID-k nem léteznek;
a dokumentált parancsokkal új futások és modellverziók állíthatók elő.

## Ellenőrzések

| Ellenőrzés | Eredmény |
| --- | --- |
| `make quality` a projekt interpreterével | Sikeres |
| pytest | 182 sikeres teszt |
| Src statement és branch coverage, külön mérve | 100% |
| Tesztkód statement és branch coverage, külön mérve | 100% |
| Összesített coverage | 100%, nincs coverage-kizárás |
| Flake8 | 0 hiba, 120 karakteres sorlimit |
| Standard és szelíd strict Pyright | Mindkettő 0 hiba és 0 figyelmeztetés az src alatt |
| `pip check`, offline requirements szárazpróba | Sikeres |
| Modell betöltése külön, izolált Python-folyamatból | A modell artifactjába csomagolt saját kódot használja |
| Javuló jelölt, rosszabb jelölt, azonos eredmény, konfigurált margin | Valós Registry-integrációs tesztek sikeresek |
| Tárolási hiba és aliasfrissítési hiba | Az aktív champion változatlan, a futás `FAILED` |
| Megosztott tanítási zár | A párhuzamos belépést megakadályozó teszt sikeres |
| `mlflow run . --env-manager=local ... -e train` | Sikeres valódi tanítás, 4-es modellverzió |
| DVC-adatelőkészítés az új környezettel | Mind a 11 kimenet bájtra azonos az 1. fázis eredményével |
| DVC push | A helyi remote és cache szinkronban van |

Az MLflow 3.17.2 az opcionális ResponsesAgent sémájának importálásakor egy `AnyType` warningot
adhat. A pytest csak ezt a konkrét külső üzenetet szűri, modullal és indoklással megadva;
az összes többi warning hiba marad. A saját regresszor input-signature-jét külön teszt ellenőrzi.

## Műszaki döntések és korlátok

A promóció feltétele `candidate_mae < champion_mae` és a javulás legalább `minimum_improvement`.
A küszöb abszolút MAE-különbség. A döntés ugyanazon aktuális validációs adaton újraszámolt
eredményekre épül. Ez az automatizált cserét védi; a jövőbeli teljesítményt nem garantálja.

A saját pipeline `skops` formátumban, három explicit ismert típus engedélyezésével kerül mentésre.
Nem engedélyezünk automatikusan minden ismeretlen típust. A Registry/artifacttároló megbízható
forrás, a modellhez tartozó saját kód betöltődik. A külső modellbizalom nem része a minőségi kapunak.

A `promotion_approved` tag a kapu döntését jelenti; az érvényes aktív verzió forrása a Registry-alias.
Az artifactokat és döntést az alias átállítása előtt rögzítjük. Egy hibás Registry-írás mellett
a minőségi jóváhagyás artifactja megmaradhat, de a run státusza `FAILED`, és nincs sikeres eredmény.

Minden tanítófolyamatnak azonos zárfájlt és backend-konfigurációt kell használnia. A DVC-előkészítés
a tanítás előtt fejeződjön be. Az SQLite megoldás helyi, egy írót használó fejlesztési környezetre
készült; a konténeres szolgáltatás és közös tárolás a 3. fázis feladata.

Saját train/predict REST API, Docker Compose, drift monitoring, CI és Airflow még nincs implementálva.
Ezek működését a mostani eredmények nem igazolják. Screenshotok és végső beadási csomag később készülnek.
GitHub push és távoli CI-futtatás nem történt; a minimum 50 commit a teljes projekt során teljesítendő.

## Szakmai források

- [MLflow Model Registry és aliasok](https://www.mlflow.org/docs/latest/ml/model-registry/workflow/)
- [MLflow sklearn modellek](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html)
