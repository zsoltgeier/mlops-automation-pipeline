# Kiegészítő implementációs utasítások a Codex coding agent számára

## 1. Szerepkör és fő cél

Senior Python Developer, ML Engineer és MLOps Engineer szemlélettel dolgozz.

A fenti egyetemi feladatkiírás minden kötelező követelményét teljesítsd. Ne csupán demonstrációs kódot, hanem egy reprodukálható, automatizált, tesztelt, Docker-alapú, end-to-end MLOps rendszert építs.

A feladatot ténylegesen implementáld, futtasd és ellenőrizd. Ne tekints egy funkciót késznek pusztán azért, mert a szükséges fájlokat létrehoztad.

A követelmények teljesítése és a rendszer működőképessége fontosabb, mint a választott ML algoritmus abszolút predikciós teljesítménye.

### Alapelv: production-ready, de ne overengineered

A production-ready szemlélet **nem jelenti azt, hogy a rendszernek indokolatlanul komplexnek kell lennie**.

Ez egy egyetemi féléves projekt, amelynek elsődleges célja az MLOps eszközök helyes alkalmazásának, az automatizált folyamatoknak és a megvalósítás mögötti mérnöki döntéseknek a bemutatása.

A rendszer ezért legyen:

- Egyszerűen érthető és karbantartható.
- Moduláris, de ne legyen túlzottan absztrakt.
- Megbízható, tesztelhető és reprodukálható.
- Minimális konfigurációval futtatható.
- Technikailag indokolt komponensekből felépítve.
- Egyetlen fejlesztő számára is könnyen átlátható.

Kövesd a KISS (Keep It Simple, Stupid), DRY és YAGNI elveket.

Ne vezess be szükségtelen microservice-eket, message broker rendszereket, Kubernetes infrastruktúrát, komplex dependency injection frameworköket, általános célú absztrakciókat vagy más túlméretezett architektúrát, ha a követelmények egyszerűbben is teljesíthetők.

Ahol több technológiai megoldás lehetséges, alapértelmezetten a legegyszerűbb, kellően robusztus, jól tesztelhető megoldást válaszd.

Ne építs olyan funkcionalitást, amelynek nincs egyértelmű szerepe a feladatkiírásban, az automatizmusokban vagy a megbízható működésben.

A production-ready itt elsősorban helyes hibakezelést, type hintinget, tesztelhetőséget, konfigurálhatóságot, strukturált naplózást, reprodukálhatóságot és átlátható kódstruktúrát jelent.

## 2. Projektfelmérés és adatkészlet

Első lépésként vizsgáld meg a meglévő repositoryt, annak fájljait, a kiválasztott datasetet, a Git historyt és a már elkészült megoldásokat.

A meglévő, működő kódot lehetőség szerint tartsd meg és bővítsd.

A dataset jellemzői alapján válassz megfelelő célváltozót, határozd meg a feladat típusát (classification/regression), és dokumentáld a választásodat.

Készíts reprodukálható előfeldolgozási pipeline-t, amely kezeli a hiányzó értékeket, a kategorikus változókat, az esetlegesen szükséges skálázást és a releváns adattisztítási lépéseket.

Kerüld el a data leakage-et: a tanulható preprocessing lépéseket kizárólag a tanítóadatokon illeszd, és ugyanazokat a transzformációkat alkalmazd predikció közben is.

Használj determinisztikus random seedet, valamint elkülönített train, validation és test adathalmazokat. A végső test adathalmaz ne vegyen részt a modellválasztásban.

A datasetet DVC-vel vagy azonos funkciót biztosító adatverzió-kezeléssel kezeld. Az előfeldolgozott dataset legyen reprodukálható, és a beadáshoz is legyen előállítva.

## 3. ML pipeline és model lifecycle

Az előfeldolgozást és a modell tanítását egységes, újrafelhasználható scikit-learn kompatibilis pipeline-ba szervezd.

Használj MLflow Trackinget a tanítási kísérletek paramétereinek, metrikáinak, dataset-referenciáinak, modelljeinek és artifactjainak naplózására.

Használd az MLflow Model Registryt modellverziózásra.

Valósíts meg champion–challenger folyamatot. Legyen legalább egy baseline modell és legalább egy új modelljelölt.

A kiválasztott, feladattípushoz illő elsődleges KPI alapján hasonlítsd össze az új jelöltet az aktuális champion modellel, ugyanazon validációs adathalmazon.

Ha az új modell teljesíti az előre definiált, konfigurálható promóciós feltételt, automatikusan váljon az aktív champion verzióvá. Ellenkező esetben az aktuális champion maradjon használatban.

A promóció vagy elutasítás okát és az összehasonlítás eredményét MLflow-ban dokumentáld. Az első sikeres tanítás gondoskodjon a kezdeti champion verzió létrehozásáról.

A REST API mindig az aktív champion modellt használja. Modellváltás után az új verziót újraindítás nélkül vagy dokumentált, automatikus újratöltéssel tegye elérhetővé.

A modell romlásának megakadályozását ne csak implementáld, hanem automatizált teszttel is bizonyítsd.

### A modellminőség-védelem pontos értelmezése

A champion–challenger folyamat a modellek összehasonlítható validációs teljesítményét vizsgálja.

Ennek célja, hogy egy előre meghatározott KPI alapján gyengébben teljesítő új modell ne válthassa fel automatikusan a jelenlegi champion modellt.

Ez nem garantálja, hogy a modell valódi, jövőbeli predikciós teljesítménye soha nem romlik.

Az automatizmusnak reprodukálhatónak, tesztelhetőnek és dokumentáltnak kell lennie.

## 4. REST API

FastAPI és Pydantic alkalmazásával hozz létre legalább `POST /train` és `POST /predict` végpontokat, valamint egy `GET /health` végpontot.

A `/train` indítsa el a konfigurált adatokon a tanítási pipeline-t, és adjon vissza értelmezhető futási státuszt, MLflow run ID-t és modellverzió-információt. Hosszú futás esetén használj egyszerű háttérfeladat-megoldást és lekérdezhető státuszt. Ne vezess be külön task queue infrastruktúrát, ha nem indokolt.

A `/predict` validált JSON bemenetet fogadjon, támogassa a több rekordos predikciót, és adja vissza a predikciókat, a használt modellverziót és a request azonosítóját.

Mindkét végpont naplózza a megfelelő eseményeket és artifactokat MLflow-ban, beleértve a releváns kérés- és válaszadatokat. A predikciós naplózás alkalmas legyen későbbi drift-elemzésre.

A requestek naplózásánál alkalmazz alapvető adatvédelmi és biztonsági szempontokat. Ne naplózz titkokat vagy indokolatlan személyes adatokat.

Valósíts meg megfelelő HTTP hibakódokat és hibaüzeneteket hibás bemenet, hiányzó modell és belső hiba esetén.

A Swagger dokumentáció legyen automatikusan elérhető.

## 5. Evidently és Streamlit

Építs Evidently-alapú data drift monitoringot.

A reference dataset legyen egy dokumentált, DVC-vel verziókezelt referencia-adathalmaz. A current dataset a REST API-n keresztül beérkezett, naplózott predikciós bemenetekből álljon.

A drift report legyen ténylegesen generálható, menthető, és artifactként MLflow-hoz kapcsolható.

Készíts Streamlit dashboardot, amely megjeleníti az Evidently reportot, a fontosabb drift-statisztikákat, az aktuális modellverziót és a tanítási kísérletek releváns eredményeit.

Gondoskodj arról, hogy friss telepítés után is demonstrálható legyen a monitoring, például determinisztikusan generált és egyértelműen megjelölt demóadatokkal.

A Streamlit csak megjelenítési réteg legyen. A monitoring üzleti logikája a `src/` könyvtárban helyezkedjen el.

### Fontos: a drift monitoring és a champion–challenger értékelés elkülönítése

A két folyamatot külön kezeld architekturális, logikai és tesztelési szempontból is.

**Champion–challenger értékelés:**

- Az újonnan tanított és a jelenleg aktív modellek teljesítményét hasonlítja össze.
- Elsődleges célja az automatikus modellpromócióról vagy elutasításról szóló döntés.
- Megfelelő validációs adatok és ismert célváltozó szükségesek hozzá.
- Az eredmények MLflow-ba kerüljenek.
- A modell életciklusának és a training pipeline-nak legyen a része.

**Drift monitoring:**

- A referenciaadatok és az aktuálisan beérkező adatok eloszlását vizsgálja.
- Elsődleges célja az adatokban bekövetkező változások észlelése és bemutatása.
- Evidently reportokat állít elő, amelyeket a Streamlit dashboard jelenít meg.
- Az eredmények legyenek elmenthetők és visszakereshetők.
- Ne módosítsa automatikusan az aktív modellt kizárólag drift észlelése alapján.

A drift önmagában nem bizonyítja a predikciós teljesítmény romlását, különösen akkor, ha nem állnak rendelkezésre valós célváltozóértékek.

A két folyamat használhat közös adatforrásokat és naplózási infrastruktúrát, de az üzleti logikájuk és felelősségi körük legyen egyértelműen elkülönítve.

Ne kapcsolj automatikus modellcserét a drift monitoringhoz, kivéve, ha ez külön indokolt, és a champion–challenger validációs folyamaton keresztül történik.

## 6. Docker és rendszerindítás

Az alkalmazás összes szükséges szolgáltatását Docker Compose segítségével indítsd.

Egy frissen klónozott repositoryból egyetlen `docker compose up` paranccsal legyen elindítható a teljes rendszer, feltéve hogy a dokumentált alapfeltételek és a szükséges dataset rendelkezésre állnak.

A rendszer biztosítsa a szükséges adatokat, adatbázisokat, MLflow Tracking Servert és artifacttárolást. Használj perzisztens Docker volume-okat.

A szolgáltatások indulási sorrendjét és readiness feltételeit healthcheckekkel és megfelelő függőségekkel kezeld.

Az első indításkor automatikusan jöjjön létre egy használható modell, ha még nem létezik. A rendszer újraindítása ne törölje a korábbi tanításokat, modellverziókat vagy naplókat.

Biztosítsd a FastAPI, MLflow és Streamlit felületek dokumentált host portokon történő elérését.

A Docker architektúra legyen a lehető legegyszerűbb, amely még teljesíti a fenti követelményeket.

Az Airflow extra funkciót csak a kötelező rendszer stabil működése után integráld.

## 7. Kódminőség és automatizált tesztek

A teljes Python-kód kövesse a PEP8 irányelveket, alkalmazzon type hintinget, és legyen moduláris, átlátható, tesztelhető.

Használj pytestet, pytest-cov/coverage.py-t, Flake8-at és Pyrightot.

A tesztek terjedjenek ki a preprocessingre, a tanításra, a modellpromócióra és elutasításra, a Registry integrációra, a predikcióra, a REST API-ra, az artifact loggingra, valamint a drift report generálására.

Legyenek pozitív és negatív tesztesetek, integration tesztek és legalább egy end-to-end smoke test.

Az `src/` teljes kódjára külön is teljesüljön a minimum 95%-os coverage. A coverage-mérés a tesztkódokat is vegye figyelembe, de a tesztkód nem kompenzálhatja az `src/` elégtelen lefedettségét.

A coverage minimumot CI-ben automatikusan ellenőrizd. Ne használj indokolatlan `pragma: no cover`, `omit` vagy `noqa` megoldásokat.

A Flake8 konfigurációban a line-length 120 legyen. Célozz 0 lint hibát.

A Pyright standard és a feladatkiírásban megadott módosított strict konfigurációja egyaránt 0 hibával fusson az `src/` könyvtáron.

A tesztelést ne halaszd a fejlesztés végére. Minden fejlesztési fázisban legyenek az új funkciókhoz tartozó tesztek.

## 8. Git és GitHub Actions

Dolgozz inkrementálisan, valódi és értelmes Git commitokkal. A commit history az implementáció tényleges fejlődését tükrözze.

Használj legalább két feature branchet, őrizd meg azok történetét, és a main branchre merge útján integráld a változtatásokat.

Ne készíts mesterséges, üres vagy megtévesztő commitokat pusztán a darabszám növelése érdekében.

A minimum 50 commit/fő követelményt tartsd szem előtt a fejlesztés során.

Konfigurálj lokális pre-commit hookokat és GitHub Actions CI workflow-kat.

A CI automatikusan futtassa a teszteket, a coverage-ellenőrzést, a Flake8-at, a Pyrightot és a Docker image buildet.

A hibás ellenőrzés sikertelen CI státuszt eredményezzen.

### Git commitok minősége és természetessége

A Git history úgy nézzen ki, mint egy tapasztalt, senior fejlesztő valós, fokozatos munkájának dokumentációja.

A commitüzenetek legyenek természetesek, szakmailag pontosak, tömörek és egyértelműek.

A commitok értelmes, logikailag összetartozó változtatásokat tartalmazzanak.

Egy commit lehetőleg egy jól körülhatárolható fejlesztési lépést, hibajavítást, refaktorálást, tesztbővítést vagy konfigurációs változtatást reprezentáljon.

Amennyiben az oktató által előírt commit formátum rendelkezésre áll, azt kövesd. Egyébként használhatsz például Conventional Commits jellegű formátumot.

Példák megfelelő commitüzenetekre:

- `feat(data): add reproducible preprocessing pipeline`
- `test(data): cover missing value handling and encoding`
- `fix(training): prevent data leakage during validation`
- `feat(mlflow): register trained models with version metadata`
- `feat(registry): implement champion-challenger promotion`
- `test(registry): reject candidate model with lower validation score`
- `feat(api): expose batch prediction endpoint`
- `fix(api): handle missing champion model gracefully`
- `feat(monitoring): generate Evidently drift reports`
- `refactor(training): separate model evaluation from promotion`
- `ci: enforce coverage threshold and static type checks`
- `docs: document Docker setup and API usage`

Kerüld az olyan semmitmondó commitüzeneteket, mint:

- `update`
- `fix stuff`
- `changes`
- `more code`
- `commit 12`
- `final version`

Ne hozz létre mesterséges commit historyt, ne hamisíts időbélyegeket, és ne bonts fel értelmetlenül apró változtatásokra egy fejlesztési lépést kizárólag a commitdarabszám teljesítése miatt.

A minimum 50 commit követelményt valódi, fokozatos fejlesztéssel teljesítsd. Ha ez nem teljesíthető hiteles módon, jelezd az eltérést.

A Git historyból legyen visszakövethető, hogy hogyan fejlődött a projekt, milyen technikai döntések születtek, milyen hibákat javítottál, és hogyan bővült a tesztlefedettség.

A feature branch-ek és merge-ek is ezt a természetes fejlesztési folyamatot támasszák alá.

## 9. Dokumentáció és beadási artifactok

### Nyelvi követelmények

A projekt dokumentációját magyar nyelven készítsd el.

- A `README.md`, a telepítési útmutatók, az architekturális magyarázatok és a beadáshoz készített PDF magyar nyelvűek legyenek.
- A Python kód, a változónevek, függvénynevek, osztálynevek és docstringek angol nyelvűek legyenek.
- A Git commitüzenetek angol nyelvűek legyenek, természetes, professzionális fejlesztői stílusban.
- A kódban szereplő kommentek és hibaüzenetek angol nyelvűek legyenek.
- A technológiai szakkifejezéseket (pl. MLflow Tracking, Model Registry, data drift, champion–challenger, pipeline) nem szükséges magyarra fordítani.
- Az automatizált tesztek nevei és leírásai angol nyelvűek legyenek.
- A felhasználónak szánt fázisvégi összefoglalókat és státuszjelentéseket magyar nyelven készítsd.

A dokumentáció legyen szakmailag pontos, természetes magyar nyelvezetű, és kerülje a túlzottan szó szerinti angol–magyar fordításokat.

A README tartalmazza a projekt célját, architektúráját, komponenseit, függőségeit, telepítési és indítási útmutatóját, API példáit, tesztelési parancsait, monitoring működését és ismert korlátait.

Dokumentáld röviden az AI coding agent használatát is.

Rögzítsd a Python verzióját, és készíts pontos, egymással kompatibilis csomagverziókat tartalmazó `requirements.txt` fájlt.

A sikeresen elindított, valós rendszerről készíts screenshotokat, és tárold őket a `docs/screenshots/` könyvtárban.

A képek mutassák be az MLflow Tracking felületét, a Model Registryt, a REST API működését, a Docker szolgáltatásokat, a Streamlit és Evidently dashboardot, illetve Airflow használata esetén az Airflow dashboardot.

Ne használj szimulált vagy nem valós működést ábrázoló screenshotokat.

Készíts elő egy beadási mappát az előfeldolgozott datasettel és az egyoldalas összefoglaló PDF-hez szükséges adatokkal. A személyes adatokat és repository hozzáférési információkat ne találd ki.

## 10. Végső elfogadási kritériumok

A munkát csak akkor tekintsd késznek, ha a következő feltételek ellenőrzése megtörtént:

1. A Docker Compose konfiguráció érvényes, és a teljes rendszer elindul.
2. A FastAPI működik, és a train/predict végpontok sikeresen tesztelhetők.
3. Az MLflow-ban tényleges kísérletek, artifactok és regisztrált modellek láthatók.
4. Az automatikus champion–challenger döntési folyamat tesztelve van.
5. Az Evidently report valós vagy egyértelműen megjelölt demóadatokból előáll, és a Streamlit megjeleníti.
6. Az `src/` coverage legalább 95%, a tesztkód mérése is megtörténik.
7. A Flake8 és a két Pyright konfiguráció teljesíti az előírásokat.
8. A GitHub Actions workflow-k definiálva vannak és a végrehajtható ellenőrzéseik sikeresek.
9. A dokumentáció és az előírt screenshotok rendelkezésre állnak.
10. A repositoryban nincsenek véletlenül commitolt titkok, fölösleges generált állományok vagy ismert, dokumentálatlan kritikus hibák.

## 11. Kötelező, fázisokra bontott munkavégzési stratégia

**Kritikus utasítás: ne implementáld a teljes projektet egyetlen körben.**

A fejlesztést az alábbi hat, szigorúan egymás után következő fázisban végezd.

Mindig kizárólag az aktuális fázison dolgozz. A következő fázist csak akkor kezdheted el, ha az előző befejeződött, annak eredményeit ellenőrizted, és én kifejezetten engedélyeztem a továbblépést.

Már a projekt elején készíts átfogó tervet és követelmény–megvalósítás megfeleltetési táblázatot, de ez nem jogosít fel arra, hogy minden fázist egyszerre implementálj.

### 1. fázis – Dataset, DVC és preprocessing

Cél: reprodukálható adatfeldolgozási alapok.

Feladatok:

- Meglévő repository és dataset felmérése.
- Célváltozó és ML feladattípus kiválasztása.
- Alap projektstruktúra kialakítása.
- Dataset verziókezelésének beállítása DVC-vel.
- Előfeldolgozási pipeline implementálása.
- Train/validation/test felosztás.
- Preprocessing unit tesztek.
- Alapvető minőségellenőrzés és dokumentáció.
- Értelmes Git commitok.

A fázis végén legyen bizonyítható, hogy az adatok reprodukálhatóan feldolgozhatók és a szükséges tesztek sikeresen futnak.

### 2. fázis – ML pipeline, MLflow és Model Registry

Cél: automatizált ML training és modell-életciklus.

Feladatok:

- ML training pipeline.
- Baseline és challenger modellek.
- MLflow Tracking integráció.
- Paraméterek, KPI-k és artifactok naplózása.
- Model Registry integráció.
- Champion–challenger értékelés.
- Automatikus promóció vagy elutasítás.
- A model lifecycle tesztelése.
- Releváns dokumentáció és Git commitok.

A fázis végén legyen működő, tesztelt modell-összehasonlítás és verziókezelés.

### 3. fázis – REST API és Docker Compose

Cél: az alap MLOps rendszer end-to-end futtatása.

Feladatok:

- FastAPI alkalmazás.
- Train, predict és health végpontok.
- Request validation és hibakezelés.
- Request és prediction logging.
- API és integrációs tesztek.
- Dockerfile-ok.
- Docker Compose konfiguráció.
- Healthcheckek, perzisztens tárolás.
- Automatikus kezdeti modell létrehozása.
- End-to-end smoke teszt.

A fázis végén a teljes addig elkészült alaprendszer legyen elindítható egyetlen `docker compose up` paranccsal.

### 4. fázis – Evidently és Streamlit

Cél: az adatok változásának monitorozása és megjelenítése.

Feladatok:

- Reference/current dataset kezelése.
- Evidently drift report generálása.
- Monitoring artifactok mentése MLflow-ba.
- Streamlit dashboard.
- Modellverzió és releváns monitoringadatok megjelenítése.
- Monitoring automatizált tesztjei.
- Docker Compose integráció bővítése.
- A champion–challenger folyamat és a drift monitoring elkülönítésének ellenőrzése.

A fázis végén legyen működő monitoring dashboard, amely a rendszer valós vagy egyértelműen megjelölt demóadatait használja.

### 5. fázis – Teljes tesztlefedettség, CI és minőségbiztosítás

Cél: az egyetemi számszerű követelmények teljesítése.

Feladatok:

- Hiányzó unit és integration tesztek pótlása.
- Minimum 95% coverage teljesítése.
- Tesztkód coverage ellenőrzése.
- Flake8 ellenőrzések.
- Pyright standard és módosított strict ellenőrzések.
- Pre-commit hookok.
- GitHub Actions workflow-k.
- Docker image build ellenőrzése.
- Meglévő kód egyszerűsítése és indokolt refaktorálása.
- Teljes end-to-end tesztelés.

A fázis végén a projekt feleljen meg minden kötelező minőségi és automatizált ellenőrzésnek.

### 6. fázis – Airflow, dokumentáció és beadás

Cél: extra funkcionalitás, végső ellenőrzés és beadásra kész projekt.

Feladatok:

- Ha indokolt és megvalósítható, Airflow integráció az extra pontokért.
- Automatizált workflow tesztelése.
- README véglegesítése.
- Architekturális és technikai döntések dokumentálása.
- AI használatának dokumentálása.
- Valós screenshotok elkészítése.
- Git history, feature branch-ek és merge-ek ellenőrzése.
- Beadáshoz szükséges artifactok összegyűjtése.
- Egyoldalas PDF elkészítése a rendelkezésre álló adatokból.
- Végső Docker Compose és teljes rendszerellenőrzés.

Az Airflow semmiképpen ne veszélyeztesse a már működő kötelező rendszert.

### Kötelező megállási pont minden fázis után

Minden fázis befejezésekor:

1. Futtasd az aktuális fázis releváns tesztjeit és ellenőrzéseit.
2. Javítsd ki a feltárt hibákat.
3. Ellenőrizd, hogy nem sérült-e korábban működő funkció.
4. Készíts természetes, értelmes Git commitokat.
5. Röviden dokumentáld az új funkciókat és a jelentős döntéseket.
6. Készíts összefoglalót az implementált változtatásokról.
7. Sorold fel a ténylegesen futtatott teszteket és azok eredményeit.
8. Jelezd a fennmaradó hibákat, korlátozásokat vagy technikai adósságokat.
9. Mutasd be, hogy a fázis elfogadási kritériumai teljesültek-e.
10. **Állj meg, és várd meg a kifejezett engedélyemet a következő fázis elkezdésére.**

Ne kezdj bele a következő fázis implementálásába automatikusan, még akkor sem, ha az aktuális fázissal sikeresen végeztél.

Az egyes fázisok közötti alapvető kompatibilitást és architekturális előkészítést megtervezheted, de a későbbi fázisok teljes funkcionalitását ne implementáld előre.

Ha egy fázis során kritikus akadályt találsz, először próbáld meg a fázison belül megoldani. Ha ez nem lehetséges, dokumentáld a problémát és a megoldási javaslatot.

## 12. Első indításkor követendő utasítás

Amikor először megkapod ezt a feladatot:

1. Olvasd el a teljes egyetemi követelményrendszert és ezt a kiegészítő specifikációt.
2. Vizsgáld meg a repositoryt, a datasetet, a Git historyt és a fejlesztési környezetet.
3. Azonosítsd a hiányzó követelményeket, a kockázatokat és az esetleges technikai akadályokat.
4. Készíts egy rövid, áttekinthető, hatfázisú megvalósítási tervet.
5. Válaszd ki a legegyszerűbb, de megfelelően robusztus architektúrát.
6. **Kezdd el kizárólag az 1. fázis tényleges megvalósítását.**
7. Az 1. fázis végén futtasd a szükséges ellenőrzéseket, készíts Git commitokat, és számolj be az eredményekről.
8. **Ezután állj meg. Ne kezdd el a 2. fázist, amíg külön nem kérem.**

A további fázisokat ugyanezzel a munkamódszerrel, egymást követően hajtsd végre.

A teljes projekt során a legfontosabb elvek: egyszerűség, működőképesség, reprodukálhatóság, automatizáltság, jó tesztlefedettség, valós fejlesztési történet és az egyetemi követelmények bizonyítható teljesítése.