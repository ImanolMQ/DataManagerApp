# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:33:17 2024

@author: Imanol
"""

class ExtraInformation():
    
    def __init__(self):
        self.codecs_info = [
            {"Codec": "ascii", "Aliases": ["646", "us-ascii"], "Languages": ["English"]},
            {"Codec": "big5", "Aliases": ["big5-tw", "csbig5"], "Languages": ["Traditional Chinese"]},
            {"Codec": "big5hkscs", "Aliases": ["big5-hkscs", "hkscs"], "Languages": ["Traditional Chinese"]},
            {"Codec": "cp037", "Aliases": ["IBM037", "IBM039"], "Languages": ["English"]},
            {"Codec": "cp273", "Aliases": ["273", "IBM273", "csIBM273"], "Languages": ["German"]},
            {"Codec": "cp424", "Aliases": ["EBCDIC-CP-HE", "IBM424"], "Languages": ["Hebrew"]},
            {"Codec": "cp437", "Aliases": ["437", "IBM437"], "Languages": ["English"]},
            {"Codec": "cp500", "Aliases": ["EBCDIC-CP-BE", "EBCDIC-CP-CH", "IBM500"], "Languages": ["Western Europe"]},
            {"Codec": "cp720", "Aliases": [], "Languages": ["Arabic"]},
            {"Codec": "cp737", "Aliases": [], "Languages": ["Greek"]},
            {"Codec": "cp775", "Aliases": ["IBM775"], "Languages": ["Baltic languages"]},
            {"Codec": "cp850", "Aliases": ["850", "IBM850"], "Languages": ["Western Europe"]},
            {"Codec": "cp852", "Aliases": ["852", "IBM852"], "Languages": ["Central and Eastern Europe"]},
            {"Codec": "cp855", "Aliases": ["855", "IBM855"], "Languages": ["Bulgarian", "Byelorussian", "Macedonian", "Russian", "Serbian"]},
            {"Codec": "cp856", "Aliases": [], "Languages": ["Hebrew"]},
            {"Codec": "cp857", "Aliases": ["857", "IBM857"], "Languages": ["Turkish"]},
            {"Codec": "cp858", "Aliases": ["858", "IBM858"], "Languages": ["Western Europe"]},
            {"Codec": "cp860", "Aliases": ["860", "IBM860"], "Languages": ["Portuguese"]},
            {"Codec": "cp861", "Aliases": ["861", "CP-IS", "IBM861"], "Languages": ["Icelandic"]},
            {"Codec": "cp862", "Aliases": ["862", "IBM862"], "Languages": ["Hebrew"]},
            {"Codec": "cp863", "Aliases": ["863", "IBM863"], "Languages": ["Canadian"]},
            {"Codec": "cp864", "Aliases": ["IBM864"], "Languages": ["Arabic"]},
            {"Codec": "cp865", "Aliases": ["865", "IBM865"], "Languages": ["Danish", "Norwegian"]},
            {"Codec": "cp866", "Aliases": ["866", "IBM866"], "Languages": ["Russian"]},
            {"Codec": "cp869", "Aliases": ["869", "CP-GR", "IBM869"], "Languages": ["Greek"]},
            {"Codec": "cp874", "Aliases": [], "Languages": ["Thai"]},
            {"Codec": "cp875", "Aliases": [], "Languages": ["Greek"]},
            {"Codec": "cp932", "Aliases": ["932", "ms932", "mskanji", "ms-kanji", "windows-31j"], "Languages": ["Japanese"]},
            {"Codec": "cp949", "Aliases": ["949", "ms949", "uhc"], "Languages": ["Korean"]},
            {"Codec": "cp950", "Aliases": ["950", "ms950"], "Languages": ["Traditional Chinese"]},
            {"Codec": "cp1006", "Aliases": [], "Languages": ["Urdu"]},
            {"Codec": "cp1026", "Aliases": ["ibm1026"], "Languages": ["Turkish"]},
            {"Codec": "cp1125", "Aliases": ["1125", "ibm1125", "cp866u", "ruscii"], "Languages": ["Ukrainian"]},
            {"Codec": "cp1140", "Aliases": ["ibm1140"], "Languages": ["Western Europe"]},
            {"Codec": "cp1250", "Aliases": ["windows-1250"], "Languages": ["Central and Eastern Europe"]},
            {"Codec": "cp1251", "Aliases": ["windows-1251"], "Languages": ["Bulgarian", "Byelorussian", "Macedonian", "Russian", "Serbian"]},
            {"Codec": "cp1252", "Aliases": ["windows-1252"], "Languages": ["Western Europe"]},
            {"Codec": "cp1253", "Aliases": ["windows-1253"], "Languages": ["Greek"]},
            {"Codec": "cp1254", "Aliases": ["windows-1254"], "Languages": ["Turkish"]},
            {"Codec": "cp1255", "Aliases": ["windows-1255"], "Languages": ["Hebrew"]},
            {"Codec": "cp1256", "Aliases": ["windows-1256"], "Languages": ["Arabic"]},
            {"Codec": "cp1257", "Aliases": ["windows-1257"], "Languages": ["Baltic languages"]},
            {"Codec": "cp1258", "Aliases": ["windows-1258"], "Languages": ["Vietnamese"]},
            {"Codec": "euc_jp", "Aliases": ["eucjp", "ujis", "u-jis"], "Languages": ["Japanese"]},
            {"Codec": "euc_jis_2004", "Aliases": ["jisx0213", "eucjis2004"], "Languages": ["Japanese"]},
            {"Codec": "euc_jisx0213", "Aliases": ["eucjisx0213"], "Languages": ["Japanese"]},
            {"Codec": "euc_kr", "Aliases": ["euckr", "korean", "ksc5601", "ks_c-5601", "ks_c-5601-1987", "ksx1001", "ks_x-1001"], "Languages": ["Korean"]},
            {"Codec": "gb2312", "Aliases": ["chinese", "csiso58gb231280", "euc-cn", "euccn", "eucgb2312-cn", "gb2312-1980", "gb2312-80", "iso-ir-58"], "Languages": ["Simplified Chinese"]},
            {"Codec": "gbk", "Aliases": ["936", "cp936", "ms936"], "Languages": ["Unified Chinese"]},
            {"Codec": "utf_8", "Aliases": ["U8", "UTF", "utf8", "cp65001"], "Languages": ["all languages"]},
            {"Codec": "utf_8_sig", "Aliases": [], "Languages": ["all languages"]}
        ]