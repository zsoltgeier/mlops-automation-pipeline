# Ruházati dolgozók produktivitása – MLOps

Az UCI Garment Productivity dataset `actual_productivity` célváltozójának becslése regresszióval.
A cél reprodukálható adatfeldolgozás és egy automatizált, tesztelt MLOps rendszer felépítése.

Elkészült az **1–3. fázis: adatelőkészítés, MLflow modell-életciklus, REST API és Docker Compose**.
Evidently/Streamlit, CI és Airflow a későbbi fázisokban készülnek.

A teljes terv és a követelmények megfeleltetése a [megvalósítási tervben](docs/implementation-plan.md)
olvasható. Az elsődleges követelményforrás a [feladatkiírás](docs/assignment.md), a fázisok rendjét
az [implementációs útmutató](docs/implementation-guidelines.md) tartalmazza.

## AI használata

Codex segít a követelmények feldolgozásában, a kód, tesztek és dokumentáció elkészítésében.
A működést tényleges helyi futtatásokkal és automatizált ellenőrzésekkel igazoljuk.

## Gyors indítás Dockerrel

Előfeltétel a futó Docker Engine és Compose v2, valamint az első image buildhez internetkapcsolat.
WSL esetén a Docker Desktopban az adott disztribúció WSL-integrációja is legyen bekapcsolva.
A repository gyökerében legyen elérhető a `data/garments_worker_productivity.csv` nyers adatfájl;
új klónban ezt DVC-ből vagy a mellékelt forrásadatokból kell helyreállítani.
Conda és előre elkészített modell nem szükséges a konténeres induláshoz.

```bash
docker compose up
```

Az image automatikusan felépül a rögzített függőségekkel. Az MLflow egészséges állapota után
az API előállítja a feldolgozott adatokat, és champion hiányában baseline-t és Random Forestet
tanít. A modell tényleges betöltése után indul a HTTP-kiszolgálás. Az első indulás ezért hosszabb.
Háttérben, a healthcheckek eredményét megvárva:

```bash
docker compose up -d --wait
docker compose ps
docker compose logs --tail 50 api
```

- [Swagger és kipróbálható API](http://127.0.0.1:8000/docs)
- [Readiness és aktív modellverzió](http://127.0.0.1:8000/health)
- [MLflow Tracking és Model Registry](http://127.0.0.1:5000)

Portütközés esetén az opcionális `.env` fájl az `.env.example` alapján állítható be.
Alapértelmezésben csak a helyi gépről érhetők el a portok. Mindkét szolgáltatás nem root
felhasználóval fut, az MLflow egyúttal HTTP-n közvetíti az artifactokat.

| Named volume | Tartalom |
| --- | --- |
| `mlflow-state` | SQLite tracking/Registry adatbázis és modell-, kérés-, predikciós artifactok |
| `api-state` | A tanítás és inicializálás közös zárfájljai |
| `prepared-data` | Feldolgozott adatok, preprocessing és referencia |

A `docker compose stop`, `start` és `down` megőrzi a volume-okat. A `down --volumes` törli
az adatbázist és modelleket is. Újraindításkor az adatelőkészítés lefut, de meglévő champion
mellett nincs automatikus újratanítás és nem keletkeznek új modellverziók.
A Docker volume-ok és a gazdagép `runtime/` könyvtára külön állapotot tartalmaznak.

## Fejlesztői környezet

Előfeltételek: Git, Conda, internet a telepítéshez; konténeres ellenőrzéshez Docker és Compose v2.
A támogatott Python-verzió **3.11.14**. A projekt saját környezetet használ, a Conda base nem módosul.

```bash
conda env create --prefix ./.conda/envs/mlops --file environment.yml
conda activate ./.conda/envs/mlops
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
```

A már naprakész környezetnél csak az aktiválás szükséges; megváltozott függőségeknél a
`python -m pip install -r requirements.txt` parancsot is futtatni kell.
A VS Code interpreter- és tesztbeállításai
a repositoryban vannak. A `requirements.in` a közvetlen, a `requirements.txt` az összes feloldott
függőség pontos verzióját tartalmazza. A pathspec 0.12.1 a DVC scmrepo-kompatibilitása miatt rögzített.

A Linux x86_64 Conda-csomagok pontos visszaállítása alternatív módon:

```bash
conda create --prefix ./.conda/envs/mlops --file conda-linux-64.lock.txt
conda activate ./.conda/envs/mlops
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps -e .
```

## Dataset és verziókezelés

Forrás: [UCI Productivity Prediction of Garment Employees](https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees),
[DOI: 10.24432/C51S6D](https://doi.org/10.24432/C51S6D). A mellékelt leírás szerint CC BY 4.0 licencű.
1197 rekord, 15 oszlop, 59 nap, 2015-01-01 és 2015-03-11 között.
Az `actual_productivity` regressziós célváltozó. A `wip` 506 helyen hiányzik;
a 37 darab 1 fölötti célértéket és a tört létszámokat megőrizzük.

A nyers CSV és az eredeti dataset-leíró PDF DVC-verziókezelésben van. Gitbe csak a `.dvc`
pointerfájlok kerülnek. A kezdeti remote a Gitből kizárt `.dvc-remote/` helyi könyvtár.

```bash
dvc pull
dvc push
dvc status
```

Új klónba a helyi remote nem érkezik meg: a kapott forrásadatokat be kell másolni, vagy közös,
hozzáférhető DVC remote-ot kell beállítani. A fejlesztői gépen a DVC belső cache helye a lokális,
Gitből kizárt konfigurációban `.cache/dvc`; más gépen a DVC alapértelmezése használható.

## Reprodukálható adatfeldolgozás

A becslés az adott napi produktivitást magyarázza. A túlóra, ösztönző és leállás időbeli
rendelkezésre állását napkezdés előtti előrejelzéshez külön tisztázni kellene.

Az aktivált Conda-környezetben:

```bash
dvc repro
dvc repro --force --no-run-cache
dvc push
```

A CLI DVC nélkül is futtatható:

```bash
garments-prepare
garments-prepare --source data/garments_worker_productivity.csv --output /tmp/garments-check --parameters params.yaml
```

A `params.yaml` állítja a teljes napokra számított train és validation arányt és az adatelőkészítés
konfigurációjában rögzített seedet. A felosztás nem véletlen; az azonos napon megfigyelt
munkacsoportok egy halmazban maradnak. A test adatok nem vehetnek részt modellválasztásban.

| Halmaz | Napok | Sorok | Időintervallum |
| --- | ---: | ---: | --- |
| Train | 35 | 711 | 2015-01-01 – 2015-02-10 |
| Validation | 11 | 211 | 2015-02-11 – 2015-02-24 |
| Test | 13 | 275 | 2015-02-25 – 2015-03-11 |

A stateless tisztítás rendezi a dátumformátumot, javítja a részlegneveket, eltávolítja a pontos
duplikációkat és hónap/nap jellemzőket képez. A `team` kategorikus azonosító. Az érvénytelen
sémát, dátumot, negatív vagy végtelen numerikus értéket és hibás targetet elutasítjuk.
A numerikus bemenetek hiányozhatnak; a target nem hiányozhat és nem kerülhet a modell bemenetébe.

A sklearn Pipeline kizárólag a train halmazon illeszti a mediánimputálást, hiányjelzőket,
StandardScalert, kategorikus leggyakoribbérték-imputálást és OneHotEncodert. Ismeretlen kategóriák
és teljesen hiányzó train oszlopok is kezelhetők. A mentett pipeline predikciókor ugyanazt a
tisztítást és transzformációt alkalmazza. A `joblib` artifact csak megbízható forrásból tölthető be.

| Fájl a `data/processed/` alatt | Tartalom |
| --- | --- |
| `cleaned.csv` | Tisztított megfigyelések, target és `split`; a hiányok a tanulásig megmaradnak |
| `train.csv`, `validation.csv`, `test.csv` | Időben elkülönített tisztított adatok |
| `*_transformed.csv` | Trainen illesztett, numerikus jellemzők és target |
| `preprocessed.csv` | A beadáshoz előállított teljes transzformált adat, target és `split` |
| `reference.csv` | Csak a train bemenetei; DVC-verziózott referencia a későbbi drift monitoringhoz |
| `preprocessor.joblib` | Illesztett sklearn pipeline |
| `metadata.json` | Bemeneti és kimeneti SHA-256 hash-ek, paraméterek, séma, hiányok, splitadatok |

A 37 numerikus jellemzőt a train adatok határozzák meg. A `metadata.json` az összes többi kimenet
hash-ét tartalmazza; a DVC lock rögzíti a kódot, környezetet, konfigurációt és a teljes eredménykönyvtárat.

## Tanítás, MLflow és Model Registry

A tanítás a `train.csv`, `validation.csv` és `metadata.json` fájlokat használja. Ellenőrzi a tartalmi
hash-eket, a sémát és az időbeli szétválasztást; a végső `test.csv` fájlt nem olvassa be.
A teljes preprocessing és regresszor egyetlen sklearn Pipeline-ként kerül MLflow-ba.

Az aktivált környezetben, a `dvc repro` után:

```bash
export MLFLOW_DISABLE_TELEMETRY=true
export MLFLOW_DISABLE_AGENT_HINT=1
garments-train --model baseline
garments-train --model random_forest
garments-train --model baseline
```

A baseline `DummyRegressor` a train targetátlagát adja vissza. A challenger Random Forest,
alapértelmezetten 100 fával, 8-as maximális mélységgel és 3-as minimális levélmérettel.
A `training.yaml` állítja a tanítás tényleges random seedjét (42), a hiperparamétereket, az
MLflow backendjét, a kísérlet- és modellneveket, valamint a promóciós küszöböt.

A CLI JSON válasza tartalmazza a run ID-t, az új modellverziót, az aktív champion verzióját,
a validációs MAE/RMSE/R² értékeket és a döntés indokát. Minden jelölt külön Registry-verziót kap,
az elutasított modell is visszakereshető marad.

Az első sikeres jelölt automatikusan champion lesz. Később az aktuális championt ugyanazon
validációs adaton újraértékeljük. Promóció akkor történik, ha a jelölt MAE-je szigorúan kisebb,
és a javulás legalább `minimum_improvement` nagyságú. A küszöb abszolút MAE-különbség;
azonos eredmény esetén az alias nem változik.

```bash
garments-train --model random_forest --minimum-improvement 0.005
garments-train --config training.yaml --tracking-uri http://127.0.0.1:5000
```

A `champion` Registry-alias az aktív verzió hiteles forrása. A `load_champion()` minden híváskor
feloldja, majd a kiválasztott verziószámhoz kötve tölti be a modellt; nincs újraindítást igénylő cache.
A közös fájlzár sorosítja a tanításokat és a promóciót. Minden írónak ugyanazt a backend-, modellnév-
és zárkonfigurációt kell használnia. Az adat-előkészítést a tanítás előtt kell befejezni.

Az MLflow rögzíti a paramétereket, train és validation metrikákat, input dataset-referenciákat,
teljes forráshash-eket, inputpéldát és modell-signature-t. Artifactként elérhetők a provenance,
a validációs predikciók és a promóciós döntés. A `promotion_approved` tag a minőségi kapu döntése;
az aktiválást a tényleges alias és a sikeresen befejezett futás igazolja. Tárolási vagy Registry-hiba
esetén a futás `FAILED`, és nem tér vissza sikeres tanítási eredmény.

A modelleket `skops` formátumban mentjük, a saját `FeatureCleaner`, a `numpy.dtype` és a saját
erdő által létrehozott `sklearn.tree._tree.Tree` típusok explicit engedélyezésével. A szükséges
saját kód a modell artifactjába kerül. A Registry és az artifacttároló csak megbízható forrás lehet.

Az MLflow metadata-adatbázisa `runtime/mlflow.db`, artifactjai a `runtime/artifacts/` alatt vannak.
Az ismételt futások megőrzik a korábbi kísérleteket és verziókat. A `runtime/` Gitből kizárt,
ezért klónozáskor nem másolódik át; a modelleket tanítással lehet előállítani. Ez a helyi CLI
alapértelmezése; a Compose az API-t a külön MLflow szolgáltatáshoz kapcsolja.

A helyi CLI futásaihoz külön UI indítható, a Docker 5000-es portja mellett:

```bash
mlflow ui --backend-store-uri sqlite:///runtime/mlflow.db --host 127.0.0.1 --port 5001 --workers 1
```

Elérés: [http://127.0.0.1:5001](http://127.0.0.1:5001). Az Experiments és Models nézetben látszanak
a futások, artifactok, modellverziók és a champion alias.

Az `MLproject` adatelőkészítő és tanító belépési pontot is tartalmaz. A tanító entrypoint kipróbált futtatása:

```bash
MLFLOW_TRACKING_URI=sqlite:///runtime/mlflow.db mlflow run . --env-manager=local --experiment-name garment-productivity-training -e train -P model=random_forest
```

A validációs kapu az automatikus modellcsere minőségét védi. Jövőbeli vagy eltérő eloszlású adatokon
nem garantál változatlan predikciós teljesítményt. A drift monitoring külön folyamat marad.

## REST API és audit

| Végpont | Bemenet / eredmény |
| --- | --- |
| `GET /health` | Registry-elérhetőség és champion alias; kész állapotban 200 és modellverzió, egyébként 503 |
| `POST /predict` | 1–1000 rekord, predikciók, tényleges modellverzió, MLflow run ID és kérésazonosító |
| `POST /train` | `{"model":"random_forest"}` vagy `{"model":"baseline"}`; a `{}` Random Forestet választ |
| `GET /docs`, `GET /openapi.json` | Swagger UI és géppel olvasható szerződés |

Predikciós példa:

```bash
curl --fail-with-body http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"records":[{"date":"2015-01-01","quarter":"Quarter1","department":"sewing","day":"Thursday","team":8,"targeted_productivity":0.8,"smv":26.16,"wip":1108,"over_time":7080,"incentive":98,"idle_time":0,"idle_men":0,"no_of_style_change":0,"no_of_workers":59}]}'
```

A dátum ISO `YYYY-MM-DD`, a kategóriák megengedett értékei a Swaggerben szerepelnek.
A `team` pozitív egész, a `no_of_style_change` nemnegatív egész; a többi numerikus mező véges,
nemnegatív szám. A tört létszám megengedett, a `wip` elhagyható vagy `null` lehet.
A részlegneveknél a forrásbeli `sweing` és a környező szóközök normalizálódnak.
A target és az ismeretlen mezők elutasításra kerülnek.

```bash
curl --fail-with-body http://127.0.0.1:8000/train \
  -H 'Content-Type: application/json' -d '{"model":"baseline"}'
```

A tanítás szinkron művelet, a válasz csak befejezés után érkezik meg. A szerver konfigurált
adatait és promóciós feltételét használja, ezeket a kliens nem írhatja felül. A válaszban
`status`, `request_id`, `run_id`, `model_version`, `model_uri`, `champion_version`, `metrics`
és `decision` szerepel. Egy gyengébb baseline új verzióként megmarad, de nem cseréli le a championt.
Minden predikció újra feloldja a champion aliast, majd a konkrét verzióval dolgozik.

A hibás bemenet 422, egy másik tanítás által foglalt zár 409, a hiányzó modell vagy nem elérhető
tároló 503 választ ad. Hibás modellkimenet és váratlan belső hiba 500. A hibaválaszok nem
tartalmaznak backendcímet, hitelesítési adatot vagy nyers elutasított bemenetet.
Minden kérés szerver által előállított `X-Request-ID` fejlécet kap; a train/predict válasz és
MLflow run ugyanazzal az azonosítóval kereshető vissza.

A train végpont a modell saját runjába menti az `api/request.json` és `api/response.json`
artifactokat. A predikciók a `garment-productivity-training-requests` kísérlet külön runjaiba
kerülnek, ugyanilyen JSON artifactokkal és `monitoring/current.csv` fájllal. Utóbbi a validált
bemeneteket, predikciókat, kérésazonosítót és modellverziót tartalmazza a következő monitoringfázishoz.
Artifactmentési hiba esetén nincs sikeres API-eredmény. A tanítás válasz-artifactját az alias
átállítása előtt mentjük; a tényleges sikert a befejezett run és az alias igazolja.

Az alkalmazás JSON naplósorokban rögzíti a kérésazonosítót, végpontot, metódust, státuszt és
futási időt. A fejlécek, query stringek és elutasított payloadok nem kerülnek ezekbe a naplókba.

Az élő Compose rendszer ellenőrzése az aktivált fejlesztői környezetből:

```bash
garments-smoke --output runtime/smoke.json
```

A smoke teszt ellenőrzi a health, Swagger, OpenAPI, predict, train és hibás batch válaszát,
majd HTTP-n letölti az MLflow artifactokat, és összeveti az API válaszaival. Három rekordra
predikciót készít és egy baseline jelöltet tanít, így valódi új runokat és modellverziót hoz létre.
A JSON eredményfájl Gitből kizárt. Eltérő portokhoz `--api-url` és `--mlflow-url` adható meg.

Docker nélküli API a helyi SQLite-konfigurációval, az aktivált környezetben:

```bash
uvicorn garments_mlops.api:create_app --factory --host 127.0.0.1 --port 8001 --workers 1 --no-access-log
```

Az `MLOPS_TRAINING_CONFIG` másik YAML-t, az `MLFLOW_TRACKING_URI` másik tracking szervert
jelölhet ki. Az `MLOPS_BOOTSTRAP=false` kikapcsolja az automatikus előkészítést és kezdeti tanítást.
Az alapértelmezett helyi rendszer egy API-workert és közös fájlzárat használ. Több gépen futó
tanítókhoz közös koordináció és más adatbázis szükséges. A szolgáltatás helyi fejlesztésre és
bemutatóra készült, autentikáció nincs beállítva.

## Minőségellenőrzés

Az aktivált környezetben `make quality`, vagy külön-külön:

```bash
python -m pytest --cov --cov-report=term-missing --cov-report=xml
python -m coverage report --include='src/*' --fail-under=95
python -m coverage report --include='tests/*' --fail-under=95
python -m flake8 src tests
python -m pyright
python -m pyright -p pyright-standard.json
python -m pip check
```

Az src és a tesztkód külön küszöböt kap, mindkettő legalább 95%. Statement és branch coverage is
mérve van, kizárás nélkül. A tesztek a warningokat hibának tekintik, egy MLflow 3.17.2 által
importált, a regresszortól független AI-séma konkrét `AnyType` figyelmeztetésének dokumentált
kivételével. A saját modell signature-jét teszt ellenőrzi. A Flake8 sorlimitje 120.
A Pyright alapkonfigurációja az assignment szelíd strict kapcsolóit használja; a másik standard.
Az [1. fázis](docs/phase-1-results.md), [2. fázis](docs/phase-2-results.md) és
[3. fázis jelentése](docs/phase-3-results.md) tartalmazza a tényleges eredményeket.
A 3. fázis végén 237 teszt sikeres, az src és a tesztkód statement és branch coverage-e 100%.

## Felépítés és következő fázisok

| Modul | Felelősség |
| --- | --- |
| `data.py` | Stateless adattisztítás és sémaellenőrzés |
| `splitting.py` | Paramétervalidáció és időrendi felosztás |
| `preprocessing.py` | Újrafelhasználható sklearn transzformációk |
| `prepare.py` | CLI, export és provenance |
| `modeling.py` | Baseline, Random Forest, metrikák és promóciós feltétel |
| `training_config.py`, `training_data.py` | Validált konfiguráció és ellenőrzött train/validation adatok |
| `tracking.py` | MLflow-kísérlet és verzióhoz kötött champion-feloldás |
| `training.py`, `train_cli.py` | Naplózott tanítás, Registry-promóció és CLI |
| `api_models.py`, `api.py` | Validált HTTP-szerződés, végpontok, korreláció és hibakezelés |
| `bootstrap.py` | Zárolt adatelőkészítés és kezdeti modell létrehozása |
| `prediction.py` | Verzióhoz kötött predikció és MLflow-audit |
| `smoke.py` | Valódi HTTP- és artifact-ellenőrzés, JSON eredményfájl |

A további architektúra vékony Streamlit/Evidently megjelenítővel bővül.
A baseline és challenger ugyanazon validációs adaton mért MAE alapján versenyeznek. A drift
monitoring külön folyamat, önmagában nem válthat modellt. Airflow a stabil alaprendszer után készül.

Monitoring dashboard, pre-commit hook és távoli CI még nincs. A valós UI screenshotok
a 6. fázisban készülnek a végleges rendszerről.
A végső ZIP és egyoldalas beadási PDF a 6. fázisban készül, a megadott személyes adatokkal.

## Git munkafolyamat

A projektalapok a `feature/project-foundation`, az adatfeldolgozás a `feature/data-preprocessing`,
a modell-életciklus a `feature/model-training`, az API és Docker a `feature/rest-api` branchen készül.
A mainre a branch-történetet
megőrző `--no-ff` merge-ekkel kerülnek a változások.
A fejlesztési lépéseket és tesztjeiket együtt, angol Conventional Commits üzenetekkel rögzítjük.

```bash
git log --graph --decorate --oneline --all
```

A minimum 50 commit követelmény több fázis tényleges fejlesztésével teljesítendő.
GitHub push, branch protection, távoli CI és a közös DVC remote beállítása még hátravan.
