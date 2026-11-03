# 3. fázis – REST API és Docker Compose

Ellenőrzés dátuma: 2026-11-04, Europe/Budapest.

## Megvalósítás

- FastAPI alkalmazás `POST /train`, `POST /predict`, `GET /health` végpontokkal és Swaggerrel.
- Pydantic-séma: 1–1000 rekord, ISO-dátum, megengedett kategóriák, véges nemnegatív számok,
  ismeretlen mezők és target elutasítása; opcionális `wip`, tört létszám támogatása.
- Szinkron baseline/Random Forest tanítás szerveroldali adatokkal és promóciós szabállyal.
- Párhuzamos tanításkor azonnali 409; validációs, backend- és modellhibák kezelése.
- Szerver által generált kérésazonosító a fejlécben, válaszban, JSON naplóban és MLflow-ban.
- Train request/response artifactok a modell runjában, predikciós audit külön kísérletben.
- Validált bemenetek, predikciók és modellverzió `monitoring/current.csv` artifactként.
- Zárolt inicializálás: adatelőkészítés, champion hiányában baseline és challenger tanítás,
  majd a mentett champion tényleges betöltése a kiszolgálás megkezdése előtt.
- Digesttel rögzített Python 3.11.14 image, pontos függőségek, nem root futtatás.
- Compose API és MLflow szolgáltatás, HTTP-n közvetített artifactok, healthcheckek,
  három perzisztens named volume és kizárólag loopbackre publikált portok.
- `garments-smoke` CLI: élő HTTP-kérések és letöltött MLflow artifactok tartalmi ellenőrzése.

Az API-ból nem módosítható az adatútvonal, tracking backend vagy minőségi küszöb.
A predikció minden kéréshez újra feloldja a champion aliast, és a betöltött verziót adja vissza.
A validációs hibákból kimarad a nyers input; a napló nem tartalmaz fejléceket vagy query stringet.

## Minőségellenőrzés

| Ellenőrzés | Eredmény |
| --- | --- |
| `make quality` a projekt interpreterével | Sikeres |
| pytest | 237 sikeres teszt |
| Src statement és branch coverage | 100%, külön ellenőrizve |
| Tesztkód statement és branch coverage | 100%, külön ellenőrizve |
| Összesített coverage | 100%, nincs coverage-kizárás |
| Flake8 | 0 hiba, 120 karakteres sorlimit |
| Standard és szelíd strict Pyright | Mindkettő 0 hiba és 0 figyelmeztetés az src alatt |
| `pip check` | Sikeres |
| DVC repro és push | Sikeres, a helyi remote szinkronban van |
| Friss Docker image teljes függőségtelepítése | Sikeres |
| `docker compose up -d --wait` | Mindkét szolgáltatás healthy |
| Konténeres felhasználó | Mindkét szolgáltatás UID 10001 |

Az API-integrációs teszt valódi MLflow SQLite-backenddel végigjárja a baseline tanítás,
predikció, Random Forest-promóció és új modellverzióval készülő predikció folyamatát.
Külön tesztek ellenőrzik a hibák redakcióját, a naplósorokat, az auditmentési hibát,
a Registry-kísérlet létrehozási versenyét, az inicializálás ismételhetőségét és a smoke ellenőrzéseit.

A Starlette 1.10.0 tesztkliense HTTPX2-t használ; a közvetlen tesztfüggőség ezért
`httpx2==2.16.0`. A tesztek warningkezelése továbbra is szigorú; csak a 2. fázisban dokumentált
külső MLflow `AnyType` warning kivétel. A fejlesztői és a frissen épített konténeres környezet
ugyanazt a rögzített Python-függőséglistát használja.

## Valódi konténeres futások

A kezdetben üres Docker volume-okkal induló rendszer automatikusan elkészítette az adatokat,
majd az alábbi modelleket. A korábbi helyi CLI SQLite-adatbázisa külön maradt.

| Verzió | Forrás | Validációs MAE | Döntés | Champion |
| --- | --- | ---: | --- | --- |
| 1 | Automatikus baseline | 0.1467638136 | Kezdeti promóció | 1 |
| 2 | Automatikus Random Forest | 0.1131253775 | Javulás, promóció | 2 |
| 3 | Első HTTP smoke baseline | 0.1467638136 | Rosszabb jelölt elutasítva | 2 |
| 4 | Újraindítás utáni HTTP smoke baseline | 0.1467638136 | Rosszabb jelölt elutasítva | 2 |

| Művelet | MLflow run ID |
| --- | --- |
| Kezdeti baseline | `ad2bfb0eccf84a9e8ad0694040aabe3a` |
| Kezdeti Random Forest | `32600643daff47b0a43a8bc0eab44e0d` |
| Első HTTP predikció | `cb89e83d93094bd1974e056f61270d6b` |
| Első HTTP baseline tanítás | `10a18a90468f4e4eb24f92c09a6e401b` |
| Újraindítás utáni HTTP predikció | `ef119d5f81434db5a0c05b616c14a003` |
| Újraindítás utáni HTTP baseline tanítás | `4fd1424e0d2f44178ea347c2c5d498fb` |

A smoke ellenőrzésekben a health, Swagger és OpenAPI 200, az üres predikciós batch 422 választ adott.
A predikció a 2-es modellverzióval három eredményt adott:
`[0.9507233004, 0.8397765269, 0.7886197509]`. A kérésazonosító egyezett a HTTP-fejlécben,
válaszban és mentett artifactban. A train és predict JSON artifactok, valamint a monitoring CSV
HTTP-n letölthetők voltak, tartalmuk megfelelt a tényleges kérésnek és válasznak.

A `docker compose stop`, majd az új image-dzsel végzett `up -d --wait` után a korábbi három
modellverzió, négy run és a 2-es champion változatlanul megmaradt. Az API indulása nem hozott
létre újabb tanítási runokat. Az ezt követő smoke új predikciós runnal és 4-es baseline
verzióval is sikeres lett. Az alkalmazás strukturált kérésnaplói a konténer stdoutján megjelentek.

Az eredményfájlok a helyi, Gitből kizárt `runtime/phase-3-smoke.json` és
`runtime/phase-3-after-restart-smoke.json` fájlokban vannak. A fenti run ID-k a Docker
`mlflow-state` volume-jához tartoznak; új klón saját futásazonosítókat kap.

## Tiszta klón ellenőrzése

Külön ideiglenes Git-klónban, Conda-környezet, `runtime/` és feldolgozott adatok nélkül,
kizárólag a mellékelt nyers CSV visszamásolásával futott a Compose. Eltérő projektnevet,
portokat és üres volume-okat használt. Az egyetlen `docker compose up -d --wait` parancs
felépítette az image-et, elindította az MLflow-t, előkészítette az adatokat és létrehozta
a kezdeti modelleket. Mindkét szolgáltatás healthy állapotba került.

A külön API-n a teljes HTTP/artifact smoke is sikeres: champion 2, három predikció, a 3-as
baseline elutasítva, Swagger/OpenAPI 200 és üres batch 422. Eredménye a helyi
`runtime/phase-3-fresh-check-smoke.json`. A teszt külön konténerei és volume-jai eltávolításra
kerültek; a fejlesztői rendszer és modelljei megmaradtak.

## Korlátok és következő fázis

A tanítás a kis datasethez illeszkedően szinkron. A konfiguráció egy API-workert és közös
fájlzárat használ; minden írónak ugyanazt a koordinációt kell követnie. Több gépes tanításhoz
elosztott koordináció és alkalmasabb adatbázis kellene. A healthcheck a Registryt és az aliast
ellenőrzi; minden predikció külön tölti be és ellenőrzi a modell kimenetét.

A promóciós artifact és az alias átállítása nem elosztott tranzakció. A döntés és tervezett
API-válasz az alias írása előtt kerül mentésre; a tényleges sikert a befejezett run és az alias
igazolja. A hibás futás nem ad sikeres HTTP-választ.

Autentikáció nincs; az alapértelmezett Compose csak a helyi gépre publikál. A DVC remote
továbbra is helyi, ezért új géphez az adatokat külön biztosítani kell. A monitoring-bemenet
már mentésre kerül, de Evidently-report és Streamlit dashboard a 4. fázisban készül.
Hookok és GitHub Actions az 5., screenshotok, Airflow és a végső beadási csomag a 6. fázis feladatai.
GitHub push nem történt; a minimum 50 commit a teljes projekt során teljesítendő.

A 3. fázis elfogadási feltétele teljesült: az addig elkészült alaprendszer egyetlen
`docker compose up` paranccsal indul, és valódi API/MLflow ellenőrzése sikeres.
