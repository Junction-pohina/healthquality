"""Finnish clinical catalog, 49 DDXPlus pathology mappings, and exact neural network token maps.

Connects the user-facing Finnish questionnaire and municipal dashboard directly to the 982-token
DDXPlus vocabulary encoded in ``src/features.py`` and trained in ``src/train.py`` (PyTorch MLP).
"""

from __future__ import annotations

import json
from pathlib import Path

# Suomen kunnat ja niiden hyvinvointialueet (21+1 kattava otos + suurimmat kaupungit)
MUNICIPALITIES: list[dict[str, str]] = [
    {"name": "Helsinki", "county": "HUS-yhtymä (Helsinki)"},
    {"name": "Espoo", "county": "Länsi-Uusimaa"},
    {"name": "Tampere", "county": "Pirkanmaa"},
    {"name": "Vantaa", "county": "Vantaa ja Kerava"},
    {"name": "Oulu", "county": "Pohjois-Pohjanmaa"},
    {"name": "Turku", "county": "Varsinais-Suomi"},
    {"name": "Jyväskylä", "county": "Keski-Suomi"},
    {"name": "Kuopio", "county": "Pohjois-Savo"},
    {"name": "Lahti", "county": "Päijät-Häme"},
    {"name": "Pori", "county": "Satakunta"},
    {"name": "Kouvola", "county": "Kymenlaakso"},
    {"name": "Joensuu", "county": "Pohjois-Karjala"},
    {"name": "Lappeenranta", "county": "Etelä-Karjala"},
    {"name": "Hämeenlinna", "county": "Kanta-Häme"},
    {"name": "Vaasa", "county": "Pohjanmaa"},
    {"name": "Seinäjoki", "county": "Etelä-Pohjanmaa"},
    {"name": "Rovaniemi", "county": "Lappi"},
    {"name": "Mikkeli", "county": "Etelä-Savo"},
    {"name": "Porvoo", "county": "Itä-Uusimaa"},
    {"name": "Kokkola", "county": "Keski-Pohjanmaa"},
    {"name": "Kajaani", "county": "Kainuu"},
    {"name": "Hyvinkää", "county": "Keski-Uusimaa"},
    {"name": "Maarianhamina", "county": "Ahvenanmaa"},
]

# Täsmälleen samat 8 ikähaarukkaa kuin src/features.py:AGE_BANDS (DEM_AGE_@_<id>)
AGE_BANDS_UI: list[dict[str, str]] = [
    {"id": "0-17", "label": "0–17"},
    {"id": "18-29", "label": "18–29"},
    {"id": "30-39", "label": "30–39"},
    {"id": "40-49", "label": "40–49"},
    {"id": "50-59", "label": "50–59"},
    {"id": "60-69", "label": "60–69"},
    {"id": "70-79", "label": "70–79"},
    {"id": "80+", "label": "80+"},
]

# Täsmälleen DDXPlus E_204 ("Have you traveled out of the country in the last 4 weeks?") arvokoodit
TRAVEL_REGIONS_UI: list[dict[str, str]] = [
    {"id": "V_10", "label": "Ei ulkomaanmatkoja (viim. 4 vko)", "token": "E_204_@_V_10"},
    {"id": "V_9", "label": "Eurooppa", "token": "E_204_@_V_9"},
    {"id": "V_6", "label": "Aasia", "token": "E_204_@_V_6"},
    {"id": "V_7", "label": "Kaakkois-Aasia", "token": "E_204_@_V_7"},
    {"id": "V_0", "label": "Pohjois-Afrikka", "token": "E_204_@_V_0"},
    {"id": "V_1", "label": "Länsi-Afrikka", "token": "E_204_@_V_1"},
    {"id": "V_2", "label": "Eteläinen Afrikka", "token": "E_204_@_V_2"},
    {"id": "V_4", "label": "Pohjois-Amerikka", "token": "E_204_@_V_4"},
    {"id": "V_3", "label": "Väli-Amerikka", "token": "E_204_@_V_3"},
    {"id": "V_5", "label": "Etelä-Amerikka", "token": "E_204_@_V_5"},
    {"id": "V_8", "label": "Karibia", "token": "E_204_@_V_8"},
    {"id": "V_13", "label": "Oseania", "token": "E_204_@_V_13"},
]

# Ehdollisten lisäkysymysten (C / M) suorat DDXPlus-tokenikartat
PAIN_LOCATIONS_UI: list[dict] = [
    {"id": "chest_center", "label": "Rintakehän ylä- tai keskiosa", "tokens": ["E_55_@_V_101", "E_55_@_V_159", "E_55_@_V_160"]},
    {"id": "chest_lower", "label": "Alarintakehä / rintalastan alaosa", "tokens": ["E_55_@_V_29", "E_55_@_V_101"]},
    {"id": "chest_side", "label": "Rintakehän sivu / kylki", "tokens": ["E_55_@_V_55", "E_55_@_V_56"]},
    {"id": "epigastric", "label": "Ylävatsa (epigastrium)", "tokens": ["E_55_@_V_197", "E_55_@_V_103", "E_55_@_V_104"]},
    {"id": "belly", "label": "Keskivatsa tai alavatsa", "tokens": ["E_55_@_V_187", "E_55_@_V_153"]},
    {"id": "groin", "label": "Nivunen", "tokens": ["E_55_@_V_16", "E_55_@_V_17"]},
    {"id": "throat_neck", "label": "Nielu, kurkunpää tai kaula", "tokens": ["E_55_@_V_148", "E_55_@_V_53", "E_55_@_V_54"]},
    {"id": "head_sinus", "label": "Otsa, poskipäät tai ohimo", "tokens": ["E_55_@_V_89", "E_55_@_V_108", "E_55_@_V_166"]},
    {"id": "ear", "label": "Korva", "tokens": ["E_55_@_V_129", "E_55_@_V_130"]},
    {"id": "back", "label": "Yläselkä / lapojen väli", "tokens": ["E_55_@_V_170", "E_55_@_V_171", "E_55_@_V_39"]},
]

PAIN_CHARACTERS_UI: list[dict] = [
    {"id": "pressing", "label": "Puristava tai raskas painon tunne", "tokens": ["E_54_@_V_183"]},
    {"id": "sharp", "label": "Terävä tai pistävä", "tokens": ["E_54_@_V_192", "E_54_@_V_179"]},
    {"id": "burning", "label": "Polttava tai korventava", "tokens": ["E_54_@_V_181"]},
    {"id": "cramp_tug", "label": "Kouristava tai kiristävä", "tokens": ["E_54_@_V_182", "E_54_@_V_180"]},
    {"id": "dull", "label": "Jomottava, tylppä tai arka", "tokens": ["E_54_@_V_154", "E_54_@_V_161"]},
    {"id": "throbbing", "label": "Sykkivä tai kova", "tokens": ["E_54_@_V_184", "E_54_@_V_191"]},
]

PAIN_RADIATIONS_UI: list[dict] = [
    {"id": "none", "label": "Ei säteile muualle", "tokens": ["E_57_@_V_123"]},
    {"id": "arm_shoulder", "label": "Olkapäähän tai käsivarteen", "tokens": ["E_57_@_V_194", "E_57_@_V_195", "E_57_@_V_30", "E_57_@_V_31"]},
    {"id": "jaw_neck", "label": "Kaulalle tai leukaperiin", "tokens": ["E_57_@_V_121", "E_57_@_V_53", "E_57_@_V_54"]},
    {"id": "back_scapula", "label": "Selkään tai lapaluun seutuun", "tokens": ["E_57_@_V_39", "E_57_@_V_127", "E_57_@_V_128", "E_57_@_V_170"]},
    {"id": "upper_chest", "label": "Ylävatsalta ylös rintakehälle", "tokens": ["E_57_@_V_101", "E_57_@_V_29"]},
]

SWELLING_LOCATIONS_UI: list[dict] = [
    {"id": "calf", "label": "Pohje tai toinen alaraaja", "tokens": ["E_152_@_V_119", "E_152_@_V_120"]},
    {"id": "ankles", "label": "Molemmat nilkat tai jalkaterät", "tokens": ["E_152_@_V_34", "E_152_@_V_35", "E_152_@_V_72", "E_152_@_V_73"]},
    {"id": "face", "label": "Kasvot, posket, huulet tai suun alue", "tokens": ["E_152_@_V_89", "E_152_@_V_108", "E_152_@_V_109", "E_152_@_V_116"]},
    {"id": "groin", "label": "Nivusen seutu", "tokens": ["E_152_@_V_16", "E_152_@_V_17"]},
]

RASH_COLORS_UI: list[dict] = [
    {"id": "red", "label": "Punainen / punoittava", "tokens": ["E_130_@_V_157"]},
    {"id": "pink", "label": "Vaaleanpunainen", "tokens": ["E_130_@_V_156"]},
    {"id": "yellow", "label": "Kellertävä ihonsävy (ikterus)", "tokens": ["E_130_@_V_107"]},
    {"id": "pale", "label": "Vaalea / nokkosrokkomainen", "tokens": ["E_130_@_V_138"]},
]

# Profiloivat taustatekijät (Antecedents, is_antecedent=True koulutusdatassa)
ANTECEDENTS_UI: list[dict] = [
    # Elintavat ja altisteet
    {
        "id": "ant_smoking",
        "group": "ELINTAVAT JA ALTISTUKSET",
        "label_fi": "Aktiivinen tupakointi",
        "tokens": ["E_79"],
    },
    {
        "id": "ant_former_smoker",
        "group": "ELINTAVAT JA ALTISTUKSET",
        "label_fi": "Aiempi tupakointi (lopettanut)",
        "tokens": ["E_191"],
    },
    {
        "id": "ant_secondhand_smoke",
        "group": "ELINTAVAT JA ALTISTUKSET",
        "label_fi": "Päivittäinen altistus tupakansavulle",
        "tokens": ["E_222"],
    },
    {
        "id": "ant_alcohol",
        "group": "ELINTAVAT JA ALTISTUKSET",
        "label_fi": "Runsas alkoholinkäyttö",
        "tokens": ["E_78"],
    },
    {
        "id": "ant_sick_contact",
        "group": "ELINTAVAT JA ALTISTUKSET",
        "label_fi": "Lähikontakti hengitystieoireiseen / flunssa (2 vko)",
        "tokens": ["E_41", "E_116"],
    },
    {
        "id": "ant_recent_viral",
        "group": "ELINTAVAT JA ALTISTUKSET",
        "label_fi": "Äskettäin sairastettu virusinfektio",
        "tokens": ["E_0"],
    },
    {
        "id": "ant_caffeine_stim",
        "group": "ELINTAVAT JA ALTISTUKSET",
        "label_fi": "Runsas kahvin, energiajuomien tai piristeiden käyttö",
        "tokens": ["E_35", "E_60"],
    },
    # Diagnosoidut perussairaudet & hoitohistoria
    {
        "id": "ant_asthma",
        "group": "DIAGNOSOIDUT PERUSSAIRAUDET",
        "label_fi": "Astma tai avaava inhalaatiolääkitys",
        "tokens": ["E_124"],
    },
    {
        "id": "ant_copd",
        "group": "DIAGNOSOIDUT PERUSSAIRAUDET",
        "label_fi": "Keuhkoahtaumatauti (COPD)",
        "tokens": ["E_123"],
    },
    {
        "id": "ant_hypertension",
        "group": "DIAGNOSOIDUT PERUSSAIRAUDET",
        "label_fi": "Kohonnut verenpaine tai verenpainelääkitys",
        "tokens": ["E_104"],
    },
    {
        "id": "ant_diabetes",
        "group": "DIAGNOSOIDUT PERUSSAIRAUDET",
        "label_fi": "Diabetes",
        "tokens": ["E_69"],
    },
    {
        "id": "ant_overweight",
        "group": "DIAGNOSOIDUT PERUSSAIRAUDET",
        "label_fi": "Merkittävä ylipaino",
        "tokens": ["E_70"],
    },
    {
        "id": "ant_cardiac",
        "group": "DIAGNOSOIDUT PERUSSAIRAUDET",
        "label_fi": "Sepelvaltimotauti tai sydämen vajaatoiminta",
        "tokens": ["E_105", "E_106"],
    },
    {
        "id": "ant_gerd",
        "group": "DIAGNOSOIDUT PERUSSAIRAUDET",
        "label_fi": "Refluksitauti (GERD) tai palleatyrä",
        "tokens": ["E_125", "E_98"],
    },
    {
        "id": "ant_pancreatitis",
        "group": "DIAGNOSOIDUT PERUSSAIRAUDET",
        "label_fi": "Krooninen haimatulehdus",
        "tokens": ["E_6"],
    },
    {
        "id": "ant_allergy",
        "group": "DIAGNOSOIDUT PERUSSAIRAUDET",
        "label_fi": "Allergiataipumus, atopia tai ruoka-aineallergia",
        "tokens": ["E_226", "E_12"],
    },
    {
        "id": "ant_anxiety",
        "group": "DIAGNOSOIDUT PERUSSAIRAUDET",
        "label_fi": "Ahdistuneisuushäiriö tai masennus",
        "tokens": ["E_81", "E_80"],
    },
    {
        "id": "ant_recent_surgery_immob",
        "group": "DIAGNOSOIDUT PERUSSAIRAUDET",
        "label_fi": "Leikkaus (< 1 kk), pitkä vuodelepo tai aiempi laskimotukos",
        "tokens": ["E_196", "E_109", "E_110"],
    },
    {
        "id": "ant_immunosupp",
        "group": "DIAGNOSOIDUT PERUSSAIRAUDET",
        "label_fi": "Alentunut vastustuskyky tai kortisonilääkitys",
        "tokens": ["E_227", "E_44"],
    },
    {
        "id": "ant_anemia_hist",
        "group": "DIAGNOSOIDUT PERUSSAIRAUDET",
        "label_fi": "Aiemmin todettu anemia",
        "tokens": ["E_24"],
    },
    # Lähisuvun sairausrasite
    {
        "id": "ant_fam_lung_cancer",
        "group": "LÄHISUVUN SAIRAUSRASITE",
        "label_fi": "Lähisuvussa keuhkosyöpää",
        "tokens": ["E_224"],
    },
    {
        "id": "ant_fam_panc_cancer",
        "group": "LÄHISUVUN SAIRAUSRASITE",
        "label_fi": "Lähisuvussa haimasyöpää",
        "tokens": ["E_223"],
    },
    {
        "id": "ant_fam_cardio",
        "group": "LÄHISUVUN SAIRAUSRASITE",
        "label_fi": "Lähisuvussa varhainen sydänsairaus (< 50 v)",
        "tokens": ["E_225"],
    },
    {
        "id": "ant_fam_asthma_allergy",
        "group": "LÄHISUVUN SAIRAUSRASITE",
        "label_fi": "Lähisuvussa astmaa, allergiaa tai ilmarintaa",
        "tokens": ["E_87", "E_86"],
    },
]

# Profiloivat nykyiset oireet (Symptoms, is_antecedent=False koulutusdatassa)
SYMPTOMS_UI: list[dict] = [
    # Hengitystiet, kurkku & nielu
    {
        "id": "sym_cough",
        "group": "Hengitystiet, kurkku & nielu",
        "label_fi": "Yskä",
        "notable": False,
        "tokens": ["E_201"],
    },
    {
        "id": "sym_sputum",
        "group": "Hengitystiet, kurkku & nielu",
        "label_fi": "Värilliset tai runsaat yskökset",
        "notable": False,
        "tokens": ["E_77"],
    },
    {
        "id": "sym_cough_fits",
        "group": "Hengitystiet, kurkku & nielu",
        "label_fi": "Voimakkaat, puuskittaiset yskänkohtaukset",
        "notable": False,
        "tokens": ["E_203"],
    },
    {
        "id": "sym_dyspnea",
        "group": "Hengitystiet, kurkku & nielu",
        "label_fi": "Hengenahdistus tai hengityksen työläys",
        "notable": False,
        "tokens": ["E_66"],
    },
    {
        "id": "sym_wheeze",
        "group": "Hengitystiet, kurkku & nielu",
        "label_fi": "Uloshengityksen vinkuna",
        "notable": False,
        "tokens": ["E_214"],
    },
    {
        "id": "sym_sore_throat",
        "group": "Hengitystiet, kurkku & nielu",
        "label_fi": "Kurkkukipu tai nielun arkuus",
        "notable": False,
        "tokens": ["E_97"],
    },
    {
        "id": "sym_nasal",
        "group": "Hengitystiet, kurkku & nielu",
        "label_fi": "Tukkoinen nenä tai kirkas nuha",
        "notable": False,
        "tokens": ["E_181"],
    },
    {
        "id": "sym_sinus_smell",
        "group": "Hengitystiet, kurkku & nielu",
        "label_fi": "Paksu nenäerite tai hajuaistin heikkeneminen",
        "notable": False,
        "tokens": ["E_182", "E_103"],
    },
    {
        "id": "sym_itchy_nose_eyes",
        "group": "Hengitystiet, kurkku & nielu",
        "label_fi": "Nenän, nielun tai silmien kutina",
        "notable": False,
        "tokens": ["E_169", "E_170"],
    },
    {
        "id": "sym_hoarse_voice",
        "group": "Hengitystiet, kurkku & nielu",
        "label_fi": "Äänen käheys tai madaltuminen",
        "notable": False,
        "tokens": ["E_212"],
    },
    {
        "id": "sym_stridor",
        "group": "Hengitystiet, kurkku & nielu",
        "label_fi": "Vinkuva ääni sisäänhengittäessä",
        "notable": True,
        "tokens": ["E_194"],
    },
    {
        "id": "sym_dysphagia",
        "group": "Hengitystiet, kurkku & nielu",
        "label_fi": "Nielemisvaikeus tai palan tunne nielussa",
        "notable": True,
        "tokens": ["E_65"],
    },
    {
        "id": "sym_hemoptysis",
        "group": "Hengitystiet, kurkku & nielu",
        "label_fi": "Veriyskä (veriviiruja tai verta ysköksissä)",
        "notable": True,
        "tokens": ["E_45"],
    },
    # Rintakehä, kipuoireet & verenkierto
    {
        "id": "sym_pain",
        "group": "Rintakehä, kipuoireet & verenkierto",
        "label_fi": "Paikallinen kipu, särky tai paineentunne",
        "notable": False,
        "opens_panel": "pain",
        "tokens": ["E_53"],
    },
    {
        "id": "sym_chest_pain_rest",
        "group": "Rintakehä, kipuoireet & verenkierto",
        "label_fi": "Rintakehän kipu tai paine myös levossa",
        "notable": True,
        "opens_panel": "pain",
        "tokens": ["E_14", "E_53"],
    },
    {
        "id": "sym_exertion",
        "group": "Rintakehä, kipuoireet & verenkierto",
        "label_fi": "Oireet voimistuvat fyysisessä rasituksessa",
        "notable": False,
        "tokens": ["E_218"],
    },
    {
        "id": "sym_pleuritic",
        "group": "Rintakehä, kipuoireet & verenkierto",
        "label_fi": "Kipu voimistuu syvään sisäänhengittäessä",
        "notable": False,
        "opens_panel": "pain",
        "tokens": ["E_220", "E_53"],
    },
    {
        "id": "sym_orthopnea",
        "group": "Rintakehä, kipuoireet & verenkierto",
        "label_fi": "Oireet pahenevat makuulla ja helpottavat istuessa",
        "notable": False,
        "tokens": ["E_217"],
    },
    {
        "id": "sym_palpitations",
        "group": "Rintakehä, kipuoireet & verenkierto",
        "label_fi": "Sydämentykytys, tiheä tai epätasainen syke",
        "notable": False,
        "tokens": ["E_155"],
    },
    {
        "id": "sym_dizzy_syncope",
        "group": "Rintakehä, kipuoireet & verenkierto",
        "label_fi": "Huimaus tai pyörryttävä olo",
        "notable": False,
        "tokens": ["E_82", "E_76"],
    },
    {
        "id": "sym_swelling",
        "group": "Rintakehä, kipuoireet & verenkierto",
        "label_fi": "Paikallinen turvotus (esim. alaraaja tai kasvot)",
        "notable": False,
        "opens_panel": "swelling",
        "tokens": ["E_151"],
    },
    # Yleisoireet, vatsa, hermosto & iho
    {
        "id": "sym_fever",
        "group": "Yleisoireet, vatsa, hermosto & iho",
        "label_fi": "Kuume tai vilunväristykset",
        "notable": False,
        "tokens": ["E_91", "E_94"],
    },
    {
        "id": "sym_sweating",
        "group": "Yleisoireet, vatsa, hermosto & iho",
        "label_fi": "Voimakas hikoilu tai yöhikoilu",
        "notable": False,
        "tokens": ["E_50"],
    },
    {
        "id": "sym_weight_loss",
        "group": "Yleisoireet, vatsa, hermosto & iho",
        "label_fi": "Tahaton painonlasku (viimeisen 3 kk aikana)",
        "notable": True,
        "tokens": ["E_162"],
    },
    {
        "id": "sym_appetite_loss",
        "group": "Yleisoireet, vatsa, hermosto & iho",
        "label_fi": "Ruokahaluttomuus tai nopea kylläisyyden tunne",
        "notable": False,
        "tokens": ["E_161"],
    },
    {
        "id": "sym_fatigue_pallor",
        "group": "Yleisoireet, vatsa, hermosto & iho",
        "label_fi": "Pitkittynyt uupumus tai ihon kalpeus",
        "notable": False,
        "tokens": ["E_89", "E_154"],
    },
    {
        "id": "sym_muscle_ache",
        "group": "Yleisoireet, vatsa, hermosto & iho",
        "label_fi": "Laaja-alainen lihassärky",
        "notable": False,
        "tokens": ["E_144"],
    },
    {
        "id": "sym_heartburn",
        "group": "Yleisoireet, vatsa, hermosto & iho",
        "label_fi": "Närästys tai mahasta nieluun nouseva polte",
        "notable": False,
        "tokens": ["E_173"],
    },
    {
        "id": "sym_nausea_diarrhea",
        "group": "Yleisoireet, vatsa, hermosto & iho",
        "label_fi": "Pahoinvointi, oksentelu tai ripuli",
        "notable": False,
        "tokens": ["E_148", "E_51"],
    },
    {
        "id": "sym_pale_stool_dark_urine",
        "group": "Yleisoireet, vatsa, hermosto & iho",
        "label_fi": "Vaaleat ulosteet ja tumma virtsa",
        "notable": True,
        "tokens": ["E_188"],
    },
    {
        "id": "sym_lymph_nodes",
        "group": "Yleisoireet, vatsa, hermosto & iho",
        "label_fi": "Suurentuneet tai aristavat imusolmukkeet",
        "notable": False,
        "tokens": ["E_9"],
    },
    {
        "id": "sym_neuro_weakness",
        "group": "Yleisoireet, vatsa, hermosto & iho",
        "label_fi": "Raajojen lihasheikkous, puutuminen tai luomien raskaus",
        "notable": True,
        "tokens": ["E_84", "E_172", "E_177"],
    },
    {
        "id": "sym_rash",
        "group": "Yleisoireet, vatsa, hermosto & iho",
        "label_fi": "Ihottuma, ihomuutos tai ihon värimuutos",
        "notable": False,
        "opens_panel": "rash",
        "tokens": ["E_129"],
    },
]

# Kaikki 49 DDXPlus-patologiaa täsmälleen samassa aakkosmuodossa kuin src/features.py
# severity noudattaa DDXPlus release_conditions.json -tasoja (1–5)
CONDITIONS_49: dict[str, dict] = {
    "Acute COPD exacerbation / infection": {
        "name_fi": "Keuhkoahtaumataudin (COPD) pahenemisvaihe",
        "icd10": "J44.1",
        "severity": 3,
        "specialty_fi": "Keuhkosairaudet ja allergologia",
        "requires_doctor": True,
    },
    "Acute dystonic reactions": {
        "name_fi": "Akuutti dystoninen lihasreaktio",
        "icd10": "G24.0",
        "severity": 2,
        "specialty_fi": "Neurologia",
        "requires_doctor": True,
    },
    "Acute laryngitis": {
        "name_fi": "Kurkunpääntulehdus (laryngiitti)",
        "icd10": "J04.0",
        "severity": 4,
        "specialty_fi": "Omahoito / Yleislääketiede",
        "requires_doctor": False,
    },
    "Acute otitis media": {
        "name_fi": "Välikorvantulehdus",
        "icd10": "H66.9",
        "severity": 4,
        "specialty_fi": "Yleislääketiede",
        "requires_doctor": True,
    },
    "Acute pulmonary edema": {
        "name_fi": "Keuhkojen nestekertymä (keuhkopöhöepäily)",
        "icd10": "J81.0",
        "severity": 1,
        "specialty_fi": "Kardiologia / Sisätaudit",
        "requires_doctor": True,
    },
    "Acute rhinosinusitis": {
        "name_fi": "Äkillinen sivuontelotulehdus (poskiontelotulehdus)",
        "icd10": "J01",
        "severity": 4,
        "specialty_fi": "Omahoito / Yleislääketiede",
        "requires_doctor": False,
    },
    "Allergic sinusitis": {
        "name_fi": "Allerginen nuha ja sivuonteloärsytys",
        "icd10": "J30",
        "severity": 4,
        "specialty_fi": "Omahoito / Yleislääketiede",
        "requires_doctor": False,
    },
    "Anaphylaxis": {
        "name_fi": "Voimakas allerginen yleisreaktio",
        "icd10": "T78.0",
        "severity": 1,
        "specialty_fi": "Akuuttilääketiede / Sisätaudit",
        "requires_doctor": True,
    },
    "Anemia": {
        "name_fi": "Anemia (matala hemoglobiini)",
        "icd10": "D64.9",
        "severity": 4,
        "specialty_fi": "Sisätaudit / Yleislääketiede",
        "requires_doctor": True,
    },
    "Atrial fibrillation": {
        "name_fi": "Eteisvärinä (rytmihäiriöepäily)",
        "icd10": "I48.9",
        "severity": 3,
        "specialty_fi": "Kardiologia",
        "requires_doctor": True,
    },
    "Boerhaave": {
        "name_fi": "Ruokatorven äkillinen kudosvaurioepäily",
        "icd10": "K22.3",
        "severity": 2,
        "specialty_fi": "Gastroenterologinen kirurgia",
        "requires_doctor": True,
    },
    "Bronchiectasis": {
        "name_fi": "Keuhkoputkien laajentumat (bronkiektasia)",
        "icd10": "J47",
        "severity": 3,
        "specialty_fi": "Keuhkosairaudet ja allergologia",
        "requires_doctor": True,
    },
    "Bronchiolitis": {
        "name_fi": "Pienten ilmateiden tulehdus (bronkioliitti)",
        "icd10": "J21",
        "severity": 3,
        "specialty_fi": "Lastentaudit",
        "requires_doctor": True,
    },
    "Bronchitis": {
        "name_fi": "Keuhkoputkentulehdus (bronkiitti)",
        "icd10": "J40",
        "severity": 4,
        "specialty_fi": "Omahoito / Yleislääketiede",
        "requires_doctor": False,
    },
    "Bronchospasm / acute asthma exacerbation": {
        "name_fi": "Astman pahenemisvaihe / keuhkoputkien supistuminen",
        "icd10": "J45",
        "severity": 3,
        "specialty_fi": "Keuhkosairaudet ja allergologia",
        "requires_doctor": True,
    },
    "Chagas": {
        "name_fi": "Matkailijaperäinen yleisinfektioepäily (Chagas)",
        "icd10": "B57",
        "severity": 3,
        "specialty_fi": "Infektiosairaudet",
        "requires_doctor": True,
    },
    "Chronic rhinosinusitis": {
        "name_fi": "Pitkittynyt sivuontelotulehdus",
        "icd10": "J32",
        "severity": 5,
        "specialty_fi": "Korva-, nenä- ja kurkkutaudit",
        "requires_doctor": True,
    },
    "Cluster headache": {
        "name_fi": "Sarjoittainen päänsärky",
        "icd10": "G44.0",
        "severity": 3,
        "specialty_fi": "Neurologia",
        "requires_doctor": True,
    },
    "Croup": {
        "name_fi": "Kurkunpääntulehdus (krupitauti)",
        "icd10": "J05.0",
        "severity": 2,
        "specialty_fi": "Lastentaudit / Yleislääketiede",
        "requires_doctor": True,
    },
    "Ebola": {
        "name_fi": "Matkailijan kuumeinen yleisinfektioepäily",
        "icd10": "A98.4",
        "severity": 1,
        "specialty_fi": "Infektiosairaudet",
        "requires_doctor": True,
    },
    "Epiglottitis": {
        "name_fi": "Kurkunkannen tulehdus (epiglottiitti)",
        "icd10": "J05.1",
        "severity": 2,
        "specialty_fi": "Korva-, nenä- ja kurkkutaudit",
        "requires_doctor": True,
    },
    "GERD": {
        "name_fi": "Refluksitauti (GERD)",
        "icd10": "K21",
        "severity": 3,
        "specialty_fi": "Omahoito / Gastroenterologia",
        "requires_doctor": False,
    },
    "Guillain-Barré syndrome": {
        "name_fi": "Ääreishermoston tulehduksellinen heikkous (GBS-epäily)",
        "icd10": "G61.0",
        "severity": 2,
        "specialty_fi": "Neurologia",
        "requires_doctor": True,
    },
    "HIV (initial infection)": {
        "name_fi": "Virusperäinen ensi-infektioepäily (HIV)",
        "icd10": "B20",
        "severity": 3,
        "specialty_fi": "Infektiosairaudet",
        "requires_doctor": True,
    },
    "Influenza": {
        "name_fi": "Influenssa",
        "icd10": "J11.1",
        "severity": 3,
        "specialty_fi": "Omahoito / Yleislääketiede",
        "requires_doctor": False,
    },
    "Inguinal hernia": {
        "name_fi": "Nivustyrä",
        "icd10": "K40",
        "severity": 3,
        "specialty_fi": "Yleiskirurgia",
        "requires_doctor": True,
    },
    "Larygospasm": {
        "name_fi": "Kurkunpään ärsytys- tai supistustila",
        "icd10": "J38.5",
        "severity": 1,
        "specialty_fi": "Korva-, nenä- ja kurkkutaudit",
        "requires_doctor": True,
    },
    "Localized edema": {
        "name_fi": "Paikallinen kudosturvotus",
        "icd10": "R60.0",
        "severity": 4,
        "specialty_fi": "Sisätaudit / Yleislääketiede",
        "requires_doctor": True,
    },
    "Myasthenia gravis": {
        "name_fi": "Hermolihasliitoksen heikkoussairaus (Myasthenia gravis)",
        "icd10": "G70.0",
        "severity": 3,
        "specialty_fi": "Neurologia",
        "requires_doctor": True,
    },
    "Myocarditis": {
        "name_fi": "Sydänlihastulehdus (myokardiittiepäily)",
        "icd10": "I51.4",
        "severity": 2,
        "specialty_fi": "Kardiologia",
        "requires_doctor": True,
    },
    "PSVT": {
        "name_fi": "Kohtauksittainen tiheälyöntisyys (PSVT)",
        "icd10": "I47.1",
        "severity": 2,
        "specialty_fi": "Kardiologia",
        "requires_doctor": True,
    },
    "Pancreatic neoplasm": {
        "name_fi": "Haiman alueen jatkotutkimustarve (kasvainepäily)",
        "icd10": "C25",
        "severity": 3,
        "specialty_fi": "Gastroenterologia / Gastroenterologinen kirurgia",
        "requires_doctor": True,
        "is_cancer": True,
    },
    "Panic attack": {
        "name_fi": "Paniikkioireisto / autonominen vireystilareaktio",
        "icd10": "F41.0",
        "severity": 5,
        "specialty_fi": "Yleislääketiede / Psykiatria",
        "requires_doctor": False,
    },
    "Pericarditis": {
        "name_fi": "Sydänpussintulehdus (perikardiitti)",
        "icd10": "I30",
        "severity": 4,
        "specialty_fi": "Kardiologia",
        "requires_doctor": True,
    },
    "Pneumonia": {
        "name_fi": "Keuhkokuume (pneumonia)",
        "icd10": "J18",
        "severity": 3,
        "specialty_fi": "Keuhkosairaudet ja allergologia / Yleislääketiede",
        "requires_doctor": True,
    },
    "Possible NSTEMI / STEMI": {
        "name_fi": "Sepelvaltimoperäinen rintakipuepäily",
        "icd10": "I21",
        "severity": 1,
        "specialty_fi": "Kardiologia / Akuuttilääketiede",
        "requires_doctor": True,
    },
    "Pulmonary embolism": {
        "name_fi": "Keuhkoverenkierron tukosepäily (keuhkoembolia)",
        "icd10": "I26",
        "severity": 2,
        "specialty_fi": "Keuhkosairaudet / Sisätaudit",
        "requires_doctor": True,
    },
    "Pulmonary neoplasm": {
        "name_fi": "Keuhkojen jatkotutkimustarve (kasvainepäily)",
        "icd10": "C34",
        "severity": 3,
        "specialty_fi": "Keuhkosairaudet ja allergologia",
        "requires_doctor": True,
        "is_cancer": True,
    },
    "SLE": {
        "name_fi": "Systeeminen autoimmuunitulehdus (SLE-epäily)",
        "icd10": "M32",
        "severity": 4,
        "specialty_fi": "Reumatologia",
        "requires_doctor": True,
    },
    "Sarcoidosis": {
        "name_fi": "Sarkoidoosi",
        "icd10": "D86",
        "severity": 3,
        "specialty_fi": "Keuhkosairaudet ja allergologia",
        "requires_doctor": True,
    },
    "Scombroid food poisoning": {
        "name_fi": "Histamiinireaktio ruoasta (scombroid)",
        "icd10": "T61.1",
        "severity": 2,
        "specialty_fi": "Yleislääketiede",
        "requires_doctor": True,
    },
    "Spontaneous pneumothorax": {
        "name_fi": "Ilmarintaepäily (pneumothorax)",
        "icd10": "J93",
        "severity": 2,
        "specialty_fi": "Keuhkosairaudet / Kirurgia",
        "requires_doctor": True,
    },
    "Spontaneous rib fracture": {
        "name_fi": "Kylkiluun rasitus- tai yskämurtuma",
        "icd10": "S22.9",
        "severity": 3,
        "specialty_fi": "Ortopedia ja traumatologia / Yleislääketiede",
        "requires_doctor": True,
    },
    "Stable angina": {
        "name_fi": "Rasitusrintakipu (stabiili angina pectoris)",
        "icd10": "I20.9",
        "severity": 2,
        "specialty_fi": "Kardiologia",
        "requires_doctor": True,
    },
    "Tuberculosis": {
        "name_fi": "Pitkittynyt keuhkoinfektioepäily (tuberkuloosi)",
        "icd10": "A15",
        "severity": 3,
        "specialty_fi": "Keuhkosairaudet / Infektiosairaudet",
        "requires_doctor": True,
    },
    "URTI": {
        "name_fi": "Ylähengitystieinfektio (tavallinen flunssa)",
        "icd10": "J06.9",
        "severity": 5,
        "specialty_fi": "Omahoito / Yleislääketiede",
        "requires_doctor": False,
    },
    "Unstable angina": {
        "name_fi": "Sepelvaltimoperäinen rintakipu (epästabiili angina)",
        "icd10": "I20.0",
        "severity": 2,
        "specialty_fi": "Kardiologia",
        "requires_doctor": True,
    },
    "Viral pharyngitis": {
        "name_fi": "Virusperäinen nielutulehdus",
        "icd10": "J02.9",
        "severity": 4,
        "specialty_fi": "Omahoito / Yleislääketiede",
        "requires_doctor": False,
    },
    "Whooping cough": {
        "name_fi": "Pitkittynyt puuskittainen yskä (pertussis-epäily)",
        "icd10": "A37",
        "severity": 4,
        "specialty_fi": "Infektiosairaudet / Yleislääketiede",
        "requires_doctor": True,
    },
}

SORTED_LABELS: list[str] = sorted(CONDITIONS_49.keys())


def build_tokens_from_selection(
    age_band: str,
    sex: str,
    travel_id: str,
    antecedent_ids: list[str],
    symptom_ids: list[str],
    pain: dict | None = None,
    swelling: dict | None = None,
    rash: dict | None = None,
) -> list[str]:
    """Encode UI survey answers directly into the exact DDXPlus vocabulary token strings."""
    tokens: list[str] = []

    # 1. Demographics (matching src/features.py: demographic_tokens)
    if sex in ("M", "F"):
        tokens.append(f"DEM_SEX_@_{sex}")
    tokens.append(f"DEM_AGE_@_{age_band}")

    # 2. Travel history (E_204_@_V_10 when no travel, else selected V_xx)
    trav_map = {t["id"]: t["token"] for t in TRAVEL_REGIONS_UI}
    tokens.append(trav_map.get(travel_id, "E_204_@_V_10"))

    # 3. Binary Antecedents
    ant_lookup = {a["id"]: a["tokens"] for a in ANTECEDENTS_UI}
    for aid in antecedent_ids:
        tokens.extend(ant_lookup.get(aid, []))

    # 4. Binary Symptoms
    sym_lookup = {s["id"]: s["tokens"] for s in SYMPTOMS_UI}
    for sid in symptom_ids:
        tokens.extend(sym_lookup.get(sid, []))

    # 5. Hierarchical Pain parameters (parent = E_53)
    if pain and any(s in symptom_ids for s in ("sym_pain", "sym_chest_pain_rest", "sym_pleuritic")):
        tokens.append("E_53")
        loc_map = {x["id"]: x["tokens"] for x in PAIN_LOCATIONS_UI}
        char_map = {x["id"]: x["tokens"] for x in PAIN_CHARACTERS_UI}
        rad_map = {x["id"]: x["tokens"] for x in PAIN_RADIATIONS_UI}
        tokens.extend(loc_map.get(pain.get("location", "chest_center"), ["E_55_@_V_101"]))
        tokens.extend(char_map.get(pain.get("character", "pressing"), ["E_54_@_V_183"]))
        tokens.extend(rad_map.get(pain.get("radiation", "none"), ["E_57_@_V_123"]))
        vas = max(0, min(10, int(pain.get("intensity", 5))))
        onset = max(0, min(10, int(pain.get("onset", 5))))
        tokens.append(f"E_56_@_{vas}")
        tokens.append(f"E_59_@_{onset}")
        tokens.append("E_58_@_6")

    # 6. Hierarchical Swelling parameters (parent = E_151)
    if swelling and "sym_swelling" in symptom_ids:
        tokens.append("E_151")
        sw_map = {x["id"]: x["tokens"] for x in SWELLING_LOCATIONS_UI}
        tokens.extend(sw_map.get(swelling.get("location", "calf"), ["E_152_@_V_119"]))

    # 7. Hierarchical Skin / Rash parameters (parent = E_129)
    if rash and "sym_rash" in symptom_ids:
        tokens.append("E_129")
        col_map = {x["id"]: x["tokens"] for x in RASH_COLORS_UI}
        tokens.extend(col_map.get(rash.get("color", "red"), ["E_130_@_V_157"]))
        itch = max(0, min(10, int(rash.get("itch", 4))))
        swollen = 6 if rash.get("swollen", False) else 0
        tokens.append(f"E_136_@_{itch}")
        tokens.append(f"E_132_@_{swollen}")
        tokens.append("E_134_@_1")
        tokens.append("E_131_@_V_10")
        tokens.append("E_135_@_V_12")

    return sorted(set(tokens))


def resolve_evidences_from_json(
    selected_ui_ids: list[str],
    evidences_json_path: Path,
) -> list[str]:
    """Compatibility helper returning exact DDXPlus E_xxx tokens for UI IDs."""
    all_items = {item["id"]: item for item in (ANTECEDENTS_UI + SYMPTOMS_UI)}
    out: list[str] = []
    for ui_id in selected_ui_ids:
        spec = all_items.get(ui_id)
        if spec:
            out.extend(spec.get("tokens", []))
    return sorted(set(out))
