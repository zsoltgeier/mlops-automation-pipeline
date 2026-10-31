A korábban választott dataset adatain végezd el az adatok előfeldolgozását, és építs fel belőle egy machine learning modellt az általad választott célváltozóval a következő peremfeltételek mellett:

 - A machine learning modell jósága bármilyen mérőszám (kpi) esetében sem tartozik a féléves feladat körébe, miközben a féléves feladat célja a modell jósági értékeinek lehetőleg automatikus növelése vagy esetleges csökkenésének automatikus megakadályozása,
 - MLflow-val trackkeld a kísérleteidet,
 - Építsd fel a pipelineokat MLflow-val, valamint hozz létre belőle egy REST API-t, ``train`` és ``predict`` végpontokat, valamint teszteld is azokat,
 - A teszt eredményéről (Swaggert, Postmant vagy pythont alkalmazva) csatolj a képet, valamint a repositoryt is,
 - Az artifactokat mentsd MLflow-val, ``train`` és ``predict`` végpont esetében is. A beérkező kéréseket és predikciókat is logold, ezek lehetnek az Evidently drift report bemenetei,
 - Lehetőleg alkalmazz minden megoldást, amit tanultunk a félévben, többek között:
    - Git használat,
    - Git pre-commit hookok automatikus futtatása (a Github támogatja ezeket (Github actions)):
        - A tesztekre,
        - A kód coverage-re,
        - A docker image készítésrte.
    - MLflow tracking és model registry,
    - Rest api,
    - Docker konténer, docker compose,
    - EvidentlyAI, Streamlit,
    - Extra (+5%): Airflow.
 - A rendszer egyetlen docker compose up paranccsal induljon.
 - A dokumentációt a README file tartalmazza lehetőleg markdown formátumban
 - A megírt kódok legyenek production-ready állapotúak tesztekkel lefedve.
 - Python kód esetén tarts be a PEP8 irányelveket, és használd a type hinting-et,
 - A kód repository legyen git-ben (pl. github), a data(set) egy erre szolgáló verziókezelőben (pl. dvc),
 - A futtatásról készíts screenshotokat és tedd bele a git repositoryba. A screenshotokon látszódjon:
    - Az MLflow dashboard,
    - A rest api működése,
    - Docker konténer működése,
    - A létrehozott Streamlit dashboard, rajta EvidentlyAI report,
    - Airflow dashboard (ha van).

 - Amire figyelj oda, mert könnyű elrontani/elfelejteni:
    - Tiszta git main(master) branch,
    - Gyakori git commitok (utólag nehéz javítani!),
    - Python kódok tesztjei,
    - Code coverage,
    - PEP8 szabályok,
    - Type hinting,
    - Az automatizált folyamat megléte, és működése,
    - Dokumentáció,
    - Docker compose up,
    - requirements.txt pontos verziókkal
    - A hibázás nem gond, ha látszik, hogy javítottad.

Mindenkire vonatkozó (számszerű) szabályok:
 - python teszt code coverage: min 95% (coverage-el),
    - A 95% az src/ levő összes python kódra, de a Streamlit és Airflow (ha van) kód csak vékony réteg lehet, az üzleti logika a src/-ben legyen,
    - A # pragma: no cover és az omit használata csak a kód melletti indoklással alkalmazható,
    - A python teszt kódok is beleszámítanak a coverage-be, mert a le nem futó tesztkód is hiba.
 - PEP8 hibák: max 1% (az összes python kódsorban) (Flake8-cal), noqa csak indoklással a kód mellett. (line-length = 120),
 - Pyright standard módban, csak az src/ mappában: 0 hiba,
 - Extra (+5%): Pyright ,,szelíd strict'' módban (a lenti konfigurációval), csak az src/ mappában: 0 hiba.
    [tool.pyright]
    include = ["src"]
    typeCheckingMode = "strict"
    reportMissingTypeStubs = false
    reportUnknownMemberType = false
    reportUnknownArgumentType = false
    reportUnknownVariableType = false
 - Git commitok száma: min 50 db fejenként, a tárgyalt formátumban.

Egyéni feladat megvalósításnál a következő elemek extra %-ot érnek (a Github támogatja ezeket):
 - Extra (+2%): Git feature branch-ek használata a feature branch history megtartásával (min 2db)
 - Extra (+2%): A main(master) branch-re csak merge útján kerülhet commit, kivéve a typo javítását.

Csoporttagoknak a következő elemek is értékelve lesznek (a Github támogatja ezeket):
 - Git feature branch-ek használata,
 - Review-k kérése,
 - Review note-ok,
 - Pull request, és merge használata a feature branch history megtartásával,
 - Main(master) branch-re csak merge útján kerülhet commit, kivéve a typo javítását.
 - Review-k száma megegyezzen a git feature branch-ek számával.

Ajánlások:
    - MI használata erősen ajánlott. Használata esetén a dokumentációban szeretném látni röviden mire és hogyan használták,
    - Lehetőség van eltérni az itt felsorolt követelményektől, megoldásoktól, de külön indoklás kíséretében a funkció megtartásával,
    - Git repository: Github,
    - Vscode-ban a Pyright és Flake8 támogatott,
    - Extra (+5%): Demó a projekt bemutatására, ideje a zh megírása után közvetlenül (a beadási határidő előtt!),
    - A felsorolt követelmények betartását ellenőrző programok parancssorból is futtathatóak, de ajánlott ezekre megfelelő eszközt választani.

A beadás egy ZIP fájlban történik a Moodle ezen lapjára való feltöltéssel a következő file-okkal:
 - Az előfeldolgozás utáni dataset file,
 - Egyoldalas PDF file amiben a következők találhatóak:
    - Név(ek), neptun kód(ok), dátum,
    - A screenshot-ok helye a gitben,
    - Egyéb fontos megjegyzések amikről tudnom kellene. Pl: valami nem működik, miért, hogyan próbáltad megoldani (ez lehet vállalt tech. dept is), stb.,
    - A git repository linkje (ha elérhető számomra olvasási joggal), (githubhoz a felhasználónév: github@deirf.hu).
 - Ha a git repository privát és nem olvashatom, akkor a teljes projekt mappa (a .git mappával együtt!) külön ZIP fájlként.

Az értékelés folyamán az implementált folyamatnak, a használt eszközök alkalmazásának az ismeretén! van a hangsúly.