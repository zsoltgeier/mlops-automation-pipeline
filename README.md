# Ruházati dolgozók produktivitása – MLOps

Az UCI Garment Productivity dataset `actual_productivity` célváltozójának becslése regresszióval.
A rendszer reprodukálható adatfeldolgozást, MLflow modell-életciklust, auditált REST API-t,
Evidently/Streamlit monitoringot és Airflow-feladatláncot biztosít. Az új modell csak azonos
validációs adaton mért javulással válthatja le az aktív championt.

## Indítás Dockerrel

Előfeltétel: futó Docker Engine, Compose v2, internet az első buildhez és a
`data/garments_worker_productivity.csv` nyers adatfájl. WSL-ben a Docker Desktop
WSL-integrációja legyen bekapcsolva. Conda és előre betanított modell nem szükséges.
Új klónban a CSV DVC-ből, a kapott forrásadatokból vagy Python 3.11-gyel állítható helyre:

```bash
python3 src/garments_mlops/fetch_data.py
```

A letöltő a rögzített SHA-256 hash-t ellenőrzi, eltérő meglévő fájlt nem ír felül.
Az adatok rendelkezésre állása után a teljes rendszer egy paranccsal indul:

```bash
docker compose up
```

Háttérben, a readiness megvárásával: `docker compose up -d --wait --wait-timeout 300`.
Ellenőrzés: `docker compose ps`; naplók: `docker compose logs --tail 50 api`.
Az első induláskor az API előkészíti az adatokat, baseline-t és Random Forestet tanít,
majd betölti a championt. A dashboard szükség esetén 200 külön jelölt demórekordot küld az API-ra.

| Szolgáltatás | Elérés |
| --- | --- |
| FastAPI és Swagger | [localhost:8000/docs](http://127.0.0.1:8000/docs), [health](http://127.0.0.1:8000/health) |
| MLflow Tracking és Model Registry | [localhost:5000](http://127.0.0.1:5000) |
| Streamlit és Evidently | [localhost:8501](http://127.0.0.1:8501) |
| Airflow | [localhost:8080](http://127.0.0.1:8080) |

A portok csak loopbacken érhetők el; az opcionális `.env` az [.env.example](.env.example) alapján
állítható be. Mind a négy szolgáltatás nem root felhasználóval fut. A `mlflow-state`, `api-state`,
`prepared-data` és `airflow-state` volume-ok tárolják a modelleket, auditokat, zárakat, adatokat
és DAG-állapotot. A `stop`, `start` és `down` megőrzi ezeket; a `down --volumes` törli őket.
Meglévő champion mellett az újraindítás nem tanít új modellt. A Docker és a helyi CLI külön állapotot használ.

## Fejlesztői környezet és ellenőrzés

Linux/WSL, Git, Conda, Make és Node.js szükséges; a CI Node.js 22-t használ.
A Python verziója **3.11.14**, a közvetlen és feloldott függőségek a
[requirements.in](requirements.in) és [requirements.txt](requirements.txt) fájlban rögzítettek.

```bash
conda env create --prefix ./.conda/envs/mlops --file environment.yml
conda activate ./.conda/envs/mlops
python -m pip install --no-build-isolation --no-deps -e .
garments-fetch-data
make hooks
make quality
```

Frissítés: `python -m pip install -r requirements.txt`. A Linux x86_64 Conda-csomagok pontos
listája a [conda-linux-64.lock.txt](conda-linux-64.lock.txt) fájlban van; a VS Code beállításai verziózottak.
A Makefile a projekt interpreterét használja, ha létezik; felülírható:
`make quality PYTHON=/path/to/python`.

| Minőségi kapu | Ellenőrzött eredmény |
| --- | --- |
| pytest | 299 sikeres teszt |
| Src statement / branch coverage | 100% / 99,63%; együttesen 99,93% |
| Tesztkód statement és branch coverage | 100% |
| Flake8, 120 karakteres sorlimit | 0 hiba az `src`, `tests`, `dags` alatt |
| Pyright standard és a kiírás szerinti szelíd strict | Mindkettő 0 hiba, 0 figyelmeztetés az `src` alatt |
| `pip check` | Hibamentes |

A `make quality` minden kaput futtat, az src és a tesztkód coverage-ét külön, legalább 95%-kal
követeli meg. Nincs coverage-kizárás; a külső MLflow/Evidently figyelmeztetések szűk, indokolt
kivételei a [pyproject.toml](pyproject.toml) fájlban vannak. Külön parancsok:
`make test`, `make coverage`, `make lint`, `make typecheck`, `make docker-build`.

A telepített pre-commit hook a teljes minőségi kaput, a pre-push hook a Compose-validációt és
image buildet futtatja. A [GitHub Actions workflow](.github/workflows/ci.yml) előbb ugyanezt a
minőségi hookot, majd friss volume-okkal a teljes Docker-rendszertesztet futtatja.
Az XML-eredményeket és diagnosztikai naplókat 14 napig őrzi meg artifactként.
A távoli futások az [Actions oldalon](https://github.com/zsoltgeier/mlops-automation-pipeline/actions) követhetők.
A feature branch-ek története `--no-ff` merge-ekkel marad meg.

## Dataset és reprodukálható preprocessing

Forrás: [UCI Productivity Prediction of Garment Employees](https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees),
[DOI: 10.24432/C51S6D](https://doi.org/10.24432/C51S6D), CC BY 4.0.
A dataset 1197 rekordot, 15 oszlopot és 59 napot tartalmaz. A `wip` 506 helyen hiányzik;
a 37 darab 1 fölötti célérték és a tört létszámok megmaradnak.

A nyers CSV, a dataset-leíró PDF és a feldolgozott adatok DVC-kezelésben vannak; Gitbe a pointerek,
a pipeline és a lock kerülnek. A konfigurált remote helyi (`.dvc-remote/`), így másik gépen közös
remote vagy forrásadat szükséges. A CSV a fenti letöltővel helyreállítható; a leíró PDF-et az nem tölti le.

```bash
make prepare                  # DVC prepare stage, szükség esetén újrafuttatva
dvc status
dvc push                      # a konfigurált remote-ba
dvc pull                      # visszaállítás a hozzáférhető remote-ból
```

A preprocessing külön CLI-je `garments-prepare`; a paraméterek a [params.yaml](params.yaml) fájlban vannak.
A tisztítás rendezi a dátumot, a részlegneveket és a duplikációkat, naptári jellemzőket képez,
és ellenőrzi a sémát. A `team` kategorikus azonosító. Azonos napok mindig egy halmazban maradnak:

| Halmaz | Sorok | Időintervallum |
| --- | ---: | --- |
| Train | 711 | 2015-01-01 – 2015-02-10 |
| Validation | 211 | 2015-02-11 – 2015-02-24 |
| Test | 275 | 2015-02-25 – 2015-03-11 |

Az imputálás, hiányjelzők, skálázás és OneHotEncoder kizárólag a trainen illeszkedik;
a target nem kerül bemenetként a preprocessingbe. A test halmaz nem vesz részt modellválasztásban.
A `data/processed/` tartalmazza a tisztított és transzformált splitfájlokat, az illesztett
preprocessort, a target nélküli `reference.csv` fájlt és a SHA-256 provenance-ot rögzítő `metadata.json`-t.
A beadási `preprocessed.csv` 1197 sorból, 37 numerikus jellemzőből, targetből és splitjelölésből áll.

## Tanítás és modellminőség-védelem

A teljes tisztítás, preprocessing és regresszor egy sklearn Pipeline, amelyet MLflow naplóz.
A baseline `DummyRegressor`; a challenger Random Forest, alapból 100 fával, 8-as mélységgel,
3-as minimális levélmérettel és 42-es seeddel. A konfiguráció a [training.yaml](training.yaml).

Az első jelölt champion lesz. Később a jelöltet és az aktuális championt ugyanazon validációs
adaton értékeljük: promócióhoz szigorúan kisebb MAE és legalább `minimum_improvement` abszolút
javulás kell. Holtversenyben az alias megmarad. Minden jelölt új Registry-verziót kap;
a döntés, paraméterek, metrikák, dataset-azonosítók, forráshash-ek és artifactok visszakereshetők.
A tanításokat közös fájlzár sorosítja; tárolási hiba esetén nincs sikeres eredmény.

```bash
garments-train --model baseline
garments-train --model random_forest --minimum-improvement 0.005
```

A CLI alapértelmezett SQLite-állapota a Gitből kizárt `runtime/` alatt van.
A helyi UI: `mlflow ui --backend-store-uri sqlite:///runtime/mlflow.db --host 127.0.0.1 --port 5001 --workers 1`.
Az [MLproject](MLproject) `prepare` és `train` entrypointot biztosít; például:
`mlflow run . --env-manager=local -e train -P model=random_forest`.

## REST API és audit

| Végpont | Szerződés |
| --- | --- |
| `GET /health` | Readiness és aktív modellverzió |
| `POST /train` | `{"model":"baseline"}` vagy `{"model":"random_forest"}`; szinkron tanítás és promóciós döntés |
| `POST /predict` | 1–1000 rekord; predikciók, modellverzió, request ID és MLflow run ID |
| `POST /monitor` | `{"source":"live"}` vagy `{"source":"demo"}`; mentett driftreport összefoglalója |
| `GET /docs`, `GET /openapi.json` | Swagger és géppel olvasható API-séma |

Történeti rekord kipróbálása, demóként megjelölve:

```bash
curl --fail-with-body http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' -d '{
    "source": "demo",
    "records": [{
      "date": "2015-01-01", "quarter": "Quarter1", "department": "sewing",
      "day": "Thursday", "team": 8, "targeted_productivity": 0.8,
      "smv": 26.16, "wip": 1108, "over_time": 7080, "incentive": 98,
      "idle_time": 0, "idle_men": 0, "no_of_style_change": 0, "no_of_workers": 59
    }]
  }'

curl --fail-with-body http://127.0.0.1:8000/train \
  -H 'Content-Type: application/json' -d '{"model":"baseline"}'
```

A dátum ISO-formátumú, a numerikus mezők végesek és nemnegatívak, a `team` pozitív egész,
a `no_of_style_change` nemnegatív egész; a `wip` elhagyható vagy `null`. Az ismeretlen mezők és
a target elutasításra kerülnek. A kategóriákat a Swagger sorolja fel. A `source` alapértéke `live`.
Hibás bemenet: 422; foglalt tanítási zár: 409; hiányzó modell/tároló: 503; váratlan belső hiba: 500.

A `champion` alias minden predikciónál újra feloldódik, a kérés konkrét modellverzióhoz kötődik.
A train/predict kérések és válaszok MLflow `api/request.json`, `api/response.json` artifactokba
kerülnek. A predikciók `monitoring/current.csv` fájlja tartalmazza a validált bemeneteket és
a forrásjelölést. Az `X-Request-ID` a HTTP-válaszban, JSON-naplóban és a runban is követhető;
fejléceket és elutasított payloadokat nem naplózunk. Artifactmentési hiba mellett nincs sikeres API-válasz.

## Monitoring és Airflow

Az Evidently a DVC-verziózott train-referenciát hasonlítja a naplózott predikciós bemenetekhez.
A `live` és `demo` forgalom külön reportba kerül. Az ablak a legutóbbi 50 predikciós runból
legfeljebb 5000 sor; 15 jellemzőn Jensen–Shannon-távolságot számítunk, 0,1-es oszlopküszöbbel
és 50%-os dataset-küszöbbel. Ezek szemléltető, nem üzemi riasztási határok.
A dashboard jelzi a hiányarányokat és a 100 sornál kisebb current minta bizonytalanságát.

A natív `report.html`, `report.json`, `summary.json` és az elemzett adatok MLflow monitoringartifactok.
Azonos adat és beállítás esetén a report újrafelhasználható. A Streamlit megmutatja a championt,
a tanítási eredményeket és az interaktív Evidently reportot; frissítés és külön jelölt demó küldése
is indítható. CLI: `garments-monitor --source demo`; a Compose adataival:
`docker compose exec dashboard garments-monitor --source demo`.

Az Airflow `garment_productivity` DAG-ja az API-t hívja:
`train_challenger` → `predict_reference` → `monitor_demo`.
Random Forest jelöltet tanít, 200 történeti referenciarekordra kér demópredikciót, majd driftreportot ment.
Alapból a felületről kézzel indítható. Napi ütemezéshez `.env`: `MLOPS_AIRFLOW_SCHEDULE=@daily`,
majd `docker compose up -d --wait --wait-timeout 300 airflow`.
Az ütemezés UTC szerinti, catchup és automatikus retry nélkül, egyszerre egy DAG-futással.
A DAG vékony réteg; az üzleti logika az `src/` alatt van, a feladatválaszok XComban követhetők.

A teljes, izolált ellenőrzés: `make system-smoke`. Ez friss volume-okkal épít és indít, ellenőrzi
az API-t, MLflow artifactokat, Streamlitet, mindkét adatforrást és egy valódi Airflow-futást,
majd csak a saját konténereit és volume-jait távolítja el. A naplók a `runtime/system-smoke/` alatt maradnak.
Meglévő rendszeren, új auditált runok létrehozásával:

```bash
garments-smoke --dashboard-url http://127.0.0.1:8501 \
  --airflow-url http://127.0.0.1:8080 --output runtime/final-smoke.json
```

## Korlátok, dokumentáció és beadás

A becslés az adott napi produktivitást magyarázza; a napi túlóra és ösztönző miatt napkezdés előtti
előrejelzésként nem értelmezhető külön rendelkezésreállási vizsgálat nélkül. A validációs kapu
nem garantál jövőbeli pontosságot. A drift önmagában nem mér pontosságromlást és nem vált modellt.

A loopback portok, egy API-worker, közös fájlzár, SQLite és az Airflow standalone/LocalExecutor
konfiguráció helyi futtatásra valók. Az Airflow minden látogatót adminnak tekint; hálózati
telepítéshez hitelesítés, megfelelő adatbázis és közös koordináció szükséges.
Az MLflow/Registry és a mentett modellartifactok megbízható forrásból származzanak.
A nyers adatok és a perzisztens állapot nem érkeznek meg automatikusan egy Git-klónnal.

A [screenshotok](docs/screenshots/) valódi felületeket és terminálkimenetet mutatnak.
A részletes döntések és ellenőrzések a [megvalósítási tervben](docs/implementation-plan.md) és a
[fázisjelentésekben](docs/phase-6-results.md) találhatók; a követelményforrás a [feladatkiírás](docs/assignment.md).

A helyi, Gitből kizárt `submission/` tartalmazza a beadási ZIP-et: előfeldolgozott CSV,
egyoldalas magyar PDF, hash-manifest és a képek offline másolatai. A PDF-ben szerepelnek a
megadott személyes adatok, a repository és screenshotok helye, valamint a fontos korlátok.
A generáló segédek, tesztjeik és kiegészítő függőségeik helyben maradnak. A ZIP-et Moodle-ba kell feltölteni.

**AI-használat:** Codex segített a követelmények feldolgozásában, a kód, a tesztek és a dokumentáció
elkészítésében. Az eredményeket tényleges futtatásokkal és automatizált ellenőrzésekkel igazoltuk.
