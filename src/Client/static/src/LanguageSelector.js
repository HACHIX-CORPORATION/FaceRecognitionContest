import React from 'react'
import {makeStyles} from "@material-ui/core/styles";
import i18next from "./i18n";

const useStyles = makeStyles(theme => ({
    button: {
        margin: theme.spacing(1),
        padding: theme.spacing(1),
        background: "0 0",
        textDecoration: "none",
        color:"white"
    },
}));

const LanguageSelector = (props) => {
    const classes = useStyles();
    const changeLanguage = (lang) => {
        i18next.changeLanguage(lang, (err, t) => {
            if (err) return console.log('something went wrong loading', err);
            console.log("language changed to " + lang)
        });
    }
    return (
        <div>
            <button onClick={() => changeLanguage("jp")} className={classes.button}>JP</button>
            <button onClick={() => changeLanguage("vi")} className={classes.button}>VN</button>
        </div>
    )
}

export default LanguageSelector;