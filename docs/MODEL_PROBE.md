# Tokenizer probe: `HuggingFaceTB/SmolLM2-360M`

- Vocab size: **49152**
- English tokens/word: **1.367**
- Spanish tokens/word: **2.074**
- Fertility ratio (ES/EN): **1.52**  (>1.0 means Spanish costs more tokens per word)
- Bytes/token EN: **4.55**
- Bytes/token ES: **2.96**

## Example tokenizations

### Example 1
- **EN (50 tok):** `On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each.`
  - Pieces: `On ĠMonday , Ġscientists Ġfrom Ġthe ĠStanford ĠUniversity ĠSchool Ġof ĠMedicine Ġannounced Ġthe Ġinvention Ġof Ġa Ġnew Ġdiagnostic Ġtool Ġthat Ġcan Ġsort Ġcells Ġby Ġtype : Ġa Ġtiny Ġprintable Ġchip Ġthat Ġcan Ġbe Ġmanufactured Ġusing Ġstandard Ġink jet Ġprinters Ġfor Ġpossibly Ġabout Ġone ĠU . S . Ġcent Ġeach .`
- **ES (124 tok):** `El lunes, los científicos de la facultad de medicina de la Universidad de Stanford anunciaron el invento de una nueva herramienta de diagnóstico que puede catalogar las células según su tipo: un pequeñísimo chip que se puede imprimir y fabricar con impresoras de inyección de uso corriente, por un posible costo de, aproximadamente, un centavo de dólar por cada uno.`
  - Pieces: `El Ġl unes , Ġlos Ġc ient ÃŃ f icos Ġde Ġla Ġfacult ad Ġde Ġmedic ina Ġde Ġla ĠUnivers idad Ġde ĠStanford Ġan unci aron Ġel Ġinvent o Ġde Ġuna Ġn ue va Ġher ram ient a Ġde Ġdiagn Ã³ st ico Ġque Ġp ued e Ġcatalog ar Ġlas Ġc Ã© l ulas Ġseg Ãº n Ġsu Ġtip o : Ġun Ġpe que Ã± ÃŃ sim o Ġchip Ġque Ġse Ġp ued e Ġimp rim ir Ġy Ġfabric ar Ġcon Ġimp res oras Ġde Ġin ye cci Ã³n Ġde Ġus o Ġcor ri ente , Ġpor Ġun Ġpos ible Ġcost o Ġde , Ġa prox im ad ament e , Ġun Ġcent av o Ġde Ġd Ã³ lar Ġpor Ġcada Ġun o .`

### Example 2
- **EN (42 tok):** `Lead researchers say this may bring early detection of cancer, tuberculosis, HIV and malaria to patients in low-income countries, where the survival rates for illnesses such as breast cancer can be half those of richer countries.`
  - Pieces: `Lead Ġresearchers Ġsay Ġthis Ġmay Ġbring Ġearly Ġdetection Ġof Ġcancer , Ġtuberculosis , ĠHIV Ġand Ġmalaria Ġto Ġpatients Ġin Ġlow - income Ġcountries , Ġwhere Ġthe Ġsurvival Ġrates Ġfor Ġillnesses Ġsuch Ġas Ġbreast Ġcancer Ġcan Ġbe Ġhalf Ġthose Ġof Ġricher Ġcountries .`
- **ES (101 tok):** `Los principales investigadores principales sostienen que esto puede permitir la detección precoz del cáncer, la tuberculosis, el VIH y la malaria en pacientes de países de bajos recursos, donde la tasa de supervivencia de enfermedades como el cáncer de mama puede ser la mitad de la de los países más avanzados.`
  - Pieces: `Los Ġprincip ales Ġinvestig ad ores Ġprincip ales Ġs ost ien en Ġque Ġest o Ġp ued e Ġpermit ir Ġla Ġdet e cci Ã³n Ġpre co z Ġdel Ġc Ã¡n cer , Ġla Ġtuberculosis , Ġel ĠVI H Ġy Ġla Ġmalaria Ġen Ġpac ient es Ġde Ġpa ÃŃ ses Ġde Ġb aj os Ġrecurs os , Ġd onde Ġla Ġt asa Ġde Ġsuperv iven cia Ġde Ġen f erm ed ades Ġcom o Ġel Ġc Ã¡n cer Ġde Ġm ama Ġp ued e Ġser Ġla Ġmit ad Ġde Ġla Ġde Ġlos Ġpa ÃŃ ses Ġm Ã¡s Ġav anz ados .`

### Example 3
- **EN (40 tok):** `The JAS 39C Gripen crashed onto a runway at around 9:30 am local time (0230 UTC) and exploded, closing the airport to commercial flights.`
  - Pieces: `The ĠJ AS Ġ 3 9 C ĠGri pen Ġcrashed Ġonto Ġa Ġrunway Ġat Ġaround Ġ 9 : 3 0 Ġam Ġlocal Ġtime Ġ( 0 2 3 0 ĠUTC ) Ġand Ġexploded , Ġclosing Ġthe Ġairport Ġto Ġcommercial Ġflights .`
- **ES (66 tok):** `El JAS 39C Gripen impactó contra una pista cerca de las 9:30 de la mañana hora local (0230 UTC) y explotó, lo que causó el cierre del aeropuerto para vuelos comerciales.`
  - Pieces: `El ĠJ AS Ġ 3 9 C ĠGri pen Ġimpact Ã³ Ġcontra Ġuna Ġp ista Ġc erc a Ġde Ġlas Ġ 9 : 3 0 Ġde Ġla Ġma Ã± ana Ġhor a Ġlocal Ġ( 0 2 3 0 ĠUTC ) Ġy Ġexpl ot Ã³ , Ġlo Ġque Ġcaus Ã³ Ġel Ġc ierre Ġdel Ġaer op u erto Ġpara Ġv uel os Ġcom erc ial es .`

### Example 4
- **EN (15 tok):** `The pilot was identified as Squadron Leader Dilokrit Pattavee.`
  - Pieces: `The Ġpilot Ġwas Ġidentified Ġas ĠSquadron ĠLeader ĠDil ok rit ĠP att ave e .`
- **ES (26 tok):** `Se identificó al piloto como Dilokrit Pattavee, líder de escuadrón.`
  - Pieces: `Se Ġident ific Ã³ Ġal Ġpil oto Ġcom o ĠDil ok rit ĠP att ave e , Ġl ÃŃ der Ġde Ġesc u adr Ã³n .`

### Example 5
- **EN (12 tok):** `Local media reports an airport fire vehicle rolled over while responding.`
  - Pieces: `Local Ġmedia Ġreports Ġan Ġairport Ġfire Ġvehicle Ġrolled Ġover Ġwhile Ġresponding .`
- **ES (31 tok):** `La prensa local informó que una patrulla de bomberos del aeropuerto volcó mientras prestaba servicio.`
  - Pieces: `La Ġpre ns a Ġlocal Ġinform Ã³ Ġque Ġuna Ġpat r ulla Ġde Ġbomber os Ġdel Ġaer op u erto Ġvol c Ã³ Ġm ient ras Ġprest aba Ġservic io .`
