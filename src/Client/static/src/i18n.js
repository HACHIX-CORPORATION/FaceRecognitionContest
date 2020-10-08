import i18n from 'i18next'
import Backend from 'i18next-xhr-backend';
import {initReactI18next} from 'react-i18next';
import LanguageDetector from "i18next-browser-languagedetector";
import translationJP from "./locales/jp.json";
import translationVI from "./locales/vi.json";

i18n
    .use(LanguageDetector)
    .init({

        resources: {
            jp: {
                translations: translationJP
            },
            vi: {
                translations: translationVI
            }
        },
        fallbackLng: 'jp',
        debug: false,
        /* can have multiple namespace, in case you want to divide a huge translation into smaller pieces and load them on demand */
        // have a common namespace used around the full app
        ns: ["translations"],
        defaultNS: "translations",

        keySeparator: false, // we use content as keys

        interpolation: {
            escapeValue: false, // not needed for react!!
            formatSeparator: ","
        },

        react: {
            wait: true
        }
    })

export default i18n;
