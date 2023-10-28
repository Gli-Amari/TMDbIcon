from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pyswip import Prolog
import matplotlib.pyplot as plt


class Processing:

    def __init__(self, df):
        self.df: pd.DataFrame = df

    def extraction(self, col_name, key):
        self.df[col_name] = self.df[col_name].apply(lambda x: [item[key] for item in x] if x else None)
        return self.df

    def replaceFeatureValue(self, col_name, valueToReplace, value):
        self.df[col_name] = self.df[col_name].replace(valueToReplace, value)
        return self.df

    def convertFeatureToInt(self, col_name):
        self.df[col_name] = self.df[col_name].astype(int)
        return self.df

    def minMaxScaler(self, col_name):
        scaler = MinMaxScaler()
        self.df[col_name] = scaler.fit_transform(self.df[[col_name]])
        return self.df

    def dropNaN(self):
        self.df.dropna(inplace=True)
        return self.df

    def replaceIso3166(self, col_name):
        iso_mapping = {
            'AF': 1, 'AX': 2, 'AL': 3, 'DZ': 4, 'AS': 5, 'AD': 6, 'AO': 7, 'AI': 8, 'AQ': 9, 'AG': 10,
            'AR': 11, 'AM': 12, 'AW': 13, 'AU': 14, 'AT': 15, 'AZ': 16, 'BS': 17, 'BH': 18, 'BD': 19, 'BB': 20,
            'BY': 21, 'BE': 22, 'BZ': 23, 'BJ': 24, 'BM': 25, 'BT': 26, 'BO': 27, 'BQ': 28, 'BA': 29, 'BW': 30,
            'BV': 31, 'BR': 32, 'IO': 33, 'BN': 34, 'BG': 35, 'BF': 36, 'BI': 37, 'KH': 38, 'CM': 39, 'CA': 40,
            'CV': 41, 'KY': 42, 'CF': 43, 'TD': 44, 'CL': 45, 'CN': 46, 'CX': 47, 'CC': 48, 'CO': 49, 'KM': 50,
            'CG': 51, 'CD': 52, 'CK': 53, 'CR': 54, 'CI': 55, 'HR': 56, 'CU': 57, 'CW': 58, 'CY': 59, 'CZ': 60,
            'DK': 61, 'DJ': 62, 'DM': 63, 'DO': 64, 'EC': 65, 'EG': 66, 'SV': 67, 'GQ': 68, 'ER': 69, 'EE': 70,
            'ET': 71, 'FK': 72, 'FO': 73, 'FJ': 74, 'FI': 75, 'FR': 76, 'GF': 77, 'PF': 78, 'TF': 79, 'GA': 80,
            'GM': 81, 'GE': 82, 'DE': 83, 'GH': 84, 'GI': 85, 'GR': 86, 'GL': 87, 'GD': 88, 'GP': 89, 'GU': 90,
            'GT': 91, 'GG': 92, 'GN': 93, 'GW': 94, 'GY': 95, 'HT': 96, 'HM': 97, 'VA': 98, 'HN': 99, 'HK': 100,
            'HU': 101, 'IS': 102, 'IN': 103, 'ID': 104, 'IR': 105, 'IQ': 106, 'IE': 107, 'IM': 108, 'IL': 109,
            'IT': 110, 'JM': 111, 'JP': 112, 'JE': 113, 'JO': 114, 'KZ': 115, 'KE': 116, 'KI': 117, 'KP': 118,
            'KR': 119, 'KW': 120, 'KG': 121, 'LA': 122, 'LV': 123, 'LB': 124, 'LS': 125, 'LR': 126, 'LY': 127,
            'LI': 128, 'LT': 129, 'LU': 130, 'MO': 131, 'MK': 132, 'MG': 133, 'MW': 134, 'MY': 135, 'MV': 136,
            'ML': 137, 'MT': 138, 'MH': 139, 'MQ': 140, 'MR': 141, 'MU': 142, 'YT': 143, 'MX': 144, 'FM': 145,
            'MD': 146, 'MC': 147, 'MN': 148, 'ME': 149, 'MS': 150, 'MA': 151, 'MZ': 152, 'MM': 153, 'NA': 154,
            'NR': 155, 'NP': 156, 'NL': 157, 'NC': 158, 'NZ': 159, 'NI': 160, 'NE': 161, 'NG': 162, 'NU': 163,
            'NF': 164, 'MP': 165, 'NO': 166, 'OM': 167, 'PK': 168, 'PW': 169, 'PS': 170, 'PA': 171, 'PG': 172,
            'PY': 173, 'PE': 174, 'PH': 175, 'PN': 176, 'PL': 177, 'PT': 178, 'PR': 179, 'QA': 180, 'RE': 181,
            'RO': 182, 'RU': 183, 'RW': 184, 'BL': 185, 'SH': 186, 'KN': 187, 'LC': 188, 'MF': 189, 'PM': 190,
            'VC': 191, 'WS': 192, 'SM': 193, 'ST': 194, 'SA': 195, 'SN': 196, 'RS': 197, 'SC': 198, 'SL': 199,
            'SG': 200, 'SX': 201, 'SK': 202, 'SI': 203, 'SB': 204, 'SO': 205, 'ZA': 206, 'GS': 207, 'SS': 208,
            'ES': 209, 'LK': 210, 'SD': 211, 'SR': 212, 'SJ': 213, 'SE': 214, 'CH': 215, 'SY': 216, 'TW': 217,
            'TJ': 218, 'TZ': 219, 'TH': 220, 'TL': 221, 'TG': 222, 'TK': 223, 'TO': 224, 'TT': 225, 'TN': 226,
            'TR': 227, 'TM': 228, 'TC': 229, 'TV': 230, 'UG': 231, 'UA': 232, 'AE': 233, 'GB': 234, 'US': 235,
            'UM': 236, 'UY': 237, 'UZ': 238, 'VU': 239, 'VE': 240, 'VN': 241, 'VG': 242, 'VI': 243, 'WF': 244,
            'EH': 245, 'YE': 246, 'ZM': 247, 'ZW': 248,
        }
        self.df[col_name] = self.df[col_name].map(iso_mapping)
        return self.df

    def replaceIso639(self, col_name):
        iso_mapping = {
            'aa': 1,
            'ab': 2,
            'af': 3,
            'am': 4,
            'ar': 5,
            'as': 6,
            'ay': 7,
            'az': 8,
            'ba': 9,
            'be': 10,
            'bg': 11,
            'bh': 12,
            'bi': 13,
            'bn': 14,
            'bo': 15,
            'br': 16,
            'bs': 17,
            'ca': 18,
            'co': 19,
            'cs': 20,
            'cy': 21,
            'da': 22,
            'de': 23,
            'dz': 24,
            'el': 25,
            'en': 26,
            'eo': 27,
            'es': 28,
            'et': 29,
            'eu': 30,
            'fa': 31,
            'fi': 32,
            'fj': 33,
            'fo': 34,
            'fr': 35,
            'fy': 36,
            'ga': 37,
            'gd': 38,
            'gl': 39,
            'gn': 40,
            'gu': 41,
            'he': 42,
            'hi': 43,
            'hr': 44,
            'ht': 45,
            'hu': 46,
            'hy': 47,
            'id': 48,
            'is': 49,
            'it': 50,
            'iu': 51,
            'ja': 52,
            'jw': 53,
            'ka': 54,
            'kk': 55,
            'km': 56,
            'kn': 57,
            'ko': 58,
            'ku': 59,
            'ky': 60,
            'la': 61,
            'ln': 62,
            'lo': 63,
            'lt': 64,
            'lv': 65,
            'mg': 66,
            'mi': 67,
            'mk': 68,
            'ml': 69,
            'mn': 70,
            'mr': 71,
            'ms': 72,
            'mt': 73,
            'my': 74,
            'ne': 75,
            'nl': 76,
            'no': 77,
            'ny': 78,
            'oc': 79,
            'om': 80,
            'or': 81,
            'pa': 82,
            'pl': 83,
            'ps': 84,
            'pt': 85,
            'qu': 86,
            'rm': 87,
            'rn': 88,
            'ro': 89,
            'ru': 90,
            'rw': 91,
            'se': 92,
            'si': 93,
            'sk': 94,
            'sl': 95,
            'sq': 96,
            'sr': 97,
            'sv': 98,
            'sw': 99,
            'ta': 100,
            'te': 101,
            'th': 102,
            'ti': 103,
            'tk': 104,
            'tr': 105,
            'ug': 106,
            'uk': 107,
            'ur': 108,
            'uz': 109,
            'vi': 110,
            'xh': 111,
            'yi': 112,
            'yo': 113,
            'zh': 114,
            'zu': 115
        }

        self.df[col_name] = self.df[col_name].map(iso_mapping)
        return self.df

    def KBInterrogation(self):
        prolog = Prolog()
        prolog.consult('./supervized_KB.pl')
        result = []
        for _, row in self.df.iterrows():
            assertz = ('gradimento(' + str(row['vote_average'])
                       + ',' + str(row['vote_count']) + ',' + str(row['popularity']) + ',' + 'Rating' + ',' + 'Res).')
            result.append(list(prolog.query(assertz))[0]['Res'])

        self.df['likeable'] = result
        return self.df

    def getFirstValueFromFeature(self, col_name):
        self.df[col_name] = self.df[col_name].astype(str)
        self.df[col_name] = self.df[col_name].str.replace(r'\[|\]|,\s*$', '', regex=True)
        self.df[col_name] = self.df[col_name].str.split(',').str.get(0)
        self.df[col_name] = self.df[col_name].str.replace("'", "")
        return self.df

    def histDataset(self):
        plt.hist(self.df, bins=6, edgecolor='black', alpha=0.7)
        plt.xlabel('Valori')
        plt.ylabel('Frequenza')
        plt.title('Esempio di Istogramma')
        return plt.show()

    def integer_to_string(self, col_name):
        self.df[col_name] = self.df[col_name].apply(lambda x: x.to_bytes((x.bit_length() + 7) // 8, 'big').decode())
        return self.df

    def string_to_integer(self, col_name):
        self.df[col_name] = self.df[col_name].apply(lambda x: int.from_bytes(x.encode(), 'big'))
        return self.df
